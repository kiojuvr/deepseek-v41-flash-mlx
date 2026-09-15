"""External oMLX trace harness: no oMLX source is modified.

Loads the pinned oMLX DeepSeek-V4.1 model, monkeypatches Block / Gate / hc_pre /
project_logits in the loaded module namespace, feeds the fixed token sequence one
token at a time (matching the native token-serial schedule) and saves the same
boundary schema as dsv41-text-trace. This is an Apple-Silicon runtime reference,
not an official-kernel oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def logical_dtype(a):
    if a.dtype == mx.bfloat16:
        return 'bfloat16'
    if a.dtype == mx.float32:
        return 'float32'
    if a.dtype == mx.uint16:
        return 'uint16'
    if a.dtype == mx.uint8:
        return 'uint8'
    return 'other'


class Accumulator:
    """Per-token arrays concatenated along axis 0 to match the native [tokens, ...] layout."""

    def __init__(self):
        self.parts = {}

    def add(self, name, a):
        mx.eval(a)
        self.parts.setdefault(name, []).append(a)

    def joined(self, name):
        parts = self.parts[name]
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    def finalize(self, root):
        arrays = []
        for name in sorted(self.parts):
            value = self.joined(name)
            mx.eval(value)
            mx.save(str(root / (name + '.npy')), value)
            arrays.append({'name': name, 'dtype': logical_dtype(value), 'shape': list(value.shape)})
        return arrays


def install_hooks(language, acc, state):
    block_call = language.Block.__call__
    gate_call = language.Gate.__call__
    hc_pre = language.hc_pre
    hc_pre_norm = language.hc_pre_norm
    hc_post = language.hc_post
    attention_call = language.Attention.__call__
    moe_call = language.MoE.__call__
    project_logits = language.project_logits
    rmsnorm_call = language.RMSNorm.__call__
    rope_call = language.rope
    qmm_call = language.QuantizedProjection.__call__

    def sub_name(suffix):
        idx = state['layer']
        prefix = 'encoder.layer' if idx < 20 else 'decoder.layer'
        return f'{prefix}{idx}.{suffix}'

    def traced_qmm(self, x, indices=None, sorted_indices=False, block_plan=None):
        out = qmm_call(self, x, indices, sorted_indices, block_plan)
        target = state.get('qmm_targets', {}).get(id(self))
        if target is not None:
            acc.add(sub_name(target), out[0])
        return out

    def traced_rmsnorm(self, x):
        out = rmsnorm_call(self, x)
        target = state.get('norm_targets', {}).get(id(self))
        if target is not None:
            acc.add(sub_name(target), out[0])
        return out

    def traced_rope(x, positions, config, compressed, inverse=False):
        out = rope_call(x, positions, config, compressed, inverse)
        if state.get('layer', -1) < 2 and not inverse:
            call = state.get('rope_calls', 0)
            if call == 0:
                acc.add(sub_name('attn_q'), out[0])
            elif call == 1:
                acc.add(sub_name('attn_kv'), out[0])
            state['rope_calls'] = call + 1
        return out

    def traced_block(self, h, pre, cache, shared, start, image_mask):
        idx = self._dsv41_layer
        state['layer'] = idx
        state['pre_calls'] = 0
        state['post_calls'] = 0
        if idx == 0:
            acc.add('encoder.entry.hidden', h[0])
            acc.add('encoder.entry.pre_mix', pre[0])
        out_h, out_pre = block_call(self, h, pre, cache, shared, start, image_mask)
        prefix = 'encoder' if idx < 20 else 'decoder'
        acc.add(f'{prefix}.layer{idx}.hidden', out_h[0])
        acc.add(f'{prefix}.layer{idx}.pre_mix', out_pre[0])
        return out_h, out_pre

    def traced_pre_norm(x, pre, weight, eps):
        out = hc_pre_norm(x, pre, weight, eps)
        if state.get('pre_calls') == 0:
            acc.add(sub_name('attn_in'), out[0])
        elif state.get('pre_calls') == 1:
            acc.add(sub_name('ffn_in'), out[0])
        state['pre_calls'] = state.get('pre_calls', 0) + 1
        return out

    def traced_post(x, residual, post, comb):
        out = hc_post(x, residual, post, comb)
        if state.get('post_calls') == 0:
            acc.add(sub_name('post_attn'), out[0])
        state['post_calls'] = state.get('post_calls', 0) + 1
        return out

    def traced_attention(self, x, cache, shared, start):
        state['rope_calls'] = 0
        out = attention_call(self, x, cache, shared, start)
        acc.add(sub_name('attn_out'), out[0])
        return out

    def traced_moe(self, x, image_mask):
        out = moe_call(self, x, image_mask)
        acc.add(sub_name('moe_out'), out[0])
        return out

    def traced_gate(self, x, image_mask):
        idx, weights = gate_call(self, x, image_mask)
        c = self._config
        raw = (x.astype(mx.float32) @ self.weight.astype(mx.float32).T) / c.gate_temp
        scores = (
            mx.softmax(raw, -1)
            if c.score_func == 'softmax'
            else (mx.sigmoid(raw) if c.score_func == 'sigmoid' else mx.sqrt(mx.logaddexp(raw, 0)))
        )
        bias = self.bias
        if image_mask is not None and 'bias_vl' in self:
            bias = mx.where(image_mask[..., None], self.bias_vl, bias)
        corrected = (scores + bias).reshape(-1).astype(mx.float32)
        mx.eval(corrected)
        values = np.array(corrected)
        order = np.argsort(-values, kind='stable')
        if values[order[5]] == values[order[6]]:
            mx.eval(idx)
            state['ties'].append({
                'layer': int(self._dsv41_layer),
                'token': int(state['pos']),
                'sixth_id': int(order[5]),
                'seventh_id': int(order[6]),
                'sixth_score': float(values[order[5]]),
                'seventh_score': float(values[order[6]]),
                'sixth_score_bits': int(values[order[5]].view(np.uint32)),
                'seventh_score_bits': int(values[order[6]].view(np.uint32)),
                'olmx_ids': [int(v) for v in np.array(idx).reshape(-1)[:6]],
            })
        return idx, weights

    def traced_hc_pre(x, pre):
        out = hc_pre(x, pre)
        if x.ndim == 4 and x.shape[-2] == 4:
            acc.add('final.collapsed', out[0])
        return out

    def traced_project_logits(x, weight):
        out = project_logits(x, weight)
        if x.shape[-1] == 5120:
            acc.add('final.norm', x[0])
            acc.add('logits', out[0])
        return out

    language.Block.__call__ = traced_block
    language.Gate.__call__ = traced_gate
    language.hc_pre = traced_hc_pre
    language.hc_pre_norm = traced_pre_norm
    language.hc_post = traced_post
    language.Attention.__call__ = traced_attention
    language.MoE.__call__ = traced_moe
    language.RMSNorm.__call__ = traced_rmsnorm
    language.rope = traced_rope
    language.QuantizedProjection.__call__ = traced_qmm
    language.project_logits = traced_project_logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, nargs='+', default=[0, 42, 1000, 42])
    p.add_argument('--tokens-file', type=Path, default=None)
    a = p.parse_args()
    if a.output.resolve().is_relative_to(a.checkpoint.resolve()):
        raise ValueError('checkpoint read-only')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.tokens_file is not None:
        a.tokens = [int(x) for x in a.tokens_file.read_text().split()]
    if not a.tokens or len(a.tokens) > 256:
        raise ValueError('token list must contain 1..256 text tokens')

    from omlx.patches.deepseek_v41.loading import load
    import omlx.patches.deepseek_v41.language as language

    native = json.loads(a.metadata.read_text())
    native_map = native['token_map']

    print('Loading pinned oMLX DeepSeek-V4.1 (Engram SSD offload, no MTP)', flush=True)
    model, _processor = load(str(a.checkpoint), engram_ssd_offload=True, preserve_mtp=False)
    lm = model.language_model

    hasher = getattr(lm, '_hasher', None)
    if hasher is None:
        raise RuntimeError('oMLX did not build the Engram token map')
    token_map_match = (
        list(hasher.token_map) == list(native_map)
        and int(hasher.token_map.max()) + 1 == native['compressed_vocab_size']
    )

    for i, layer in enumerate(lm.layers):
        layer._dsv41_layer = i
        layer.ffn.gate._dsv41_layer = i

    acc = Accumulator()
    state = {'pos': 0, 'ties': [], 'norm_targets': {}, 'qmm_targets': {}}
    # Record the pure-SWA q/kv norms and the wq_b projection so the attention boundaries are comparable.
    for i in (0, 1):
        state['norm_targets'][id(lm.layers[i].attn.q_norm)] = 'attn_qr'
        state['norm_targets'][id(lm.layers[i].attn.kv_norm)] = 'attn_kv_norm'
        state['qmm_targets'][id(lm.layers[i].attn.wq_b)] = 'attn_qb'
    install_hooks(language, acc, state)

    mx.set_default_device(mx.gpu)
    cache = lm.make_cache()
    for pos, tok in enumerate(a.tokens):
        state['pos'] = pos
        out = model(mx.array([[tok]], mx.int32), cache=cache)
        mx.eval(out)
    # encoder.out / decoder.out alias the last layer of each half.
    for out_name, layer_name in [('encoder.out', 'encoder.layer19'), ('decoder.out', 'decoder.layer39')]:
        for suffix in ('hidden', 'pre_mix'):
            if f'{layer_name}.{suffix}' in acc.parts:
                acc.parts[f'{out_name}.{suffix}'] = list(acc.parts[f'{layer_name}.{suffix}'])

    arrays = acc.finalize(a.output)
    logits_np = np.array(mx.astype(acc.joined('logits'), mx.float32)).reshape(-1, 129280)
    argmax = [int(x) for x in np.argmax(logits_np, axis=-1)]
    manifest = {
        'schema_version': 1,
        'token_ids': a.tokens,
        'logits_argmax': argmax,
        'route_tie_count': len(state['ties']),
        'route_ties': state['ties'],
        'arrays': arrays,
        'token_map_matches_native': bool(token_map_match),
        'checkpoint': str(a.checkpoint),
        'scope': 'oMLX v0.7.0.dev2 external hook trace, token-serial; Apple Silicon reference, not an official-kernel oracle.',
    }
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'PASS: wrote {len(arrays)} boundary arrays and {len(state["ties"])} route ties to {a.output}')
    print(f'token_map_matches_native={token_map_match}')


if __name__ == '__main__':
    main()
