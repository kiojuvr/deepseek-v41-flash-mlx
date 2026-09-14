"""Isolate Q differences using one CPU sparse-attention implementation and fixed KV.

Stops before inverse RoPE/output projection; no logits or routing qualification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cpu_reference import Weights, sparse_attention, sha
from compare_traces import checked_array, validate_pair, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--cpu', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or any(a.output.resolve().is_relative_to(d.resolve()) for d in (a.checkpoint, a.native, a.cpu)):
        p.error('fresh output outside inputs required')
    torch.set_num_threads(4)
    manifests = [json.loads((d/'manifest.json').read_text()) for d in (a.native, a.cpu)]
    descriptors = validate_pair(*manifests)
    n = len(manifests[0]['token_ids'])
    if not 1 <= n <= 128:
        raise ValueError('requires 1..128 position-zero tokens')
    input_hashes = {}
    def read(which, suffix, shape):
        name = 'encoder.layer0.' + suffix
        d = descriptors[which][name]
        if d['dtype'] != 'bfloat16' or d['shape'] != shape:
            raise ValueError('invalid attention descriptor')
        root = (a.native, a.cpu)[which]
        raw = checked_array(root, d)
        input_hashes[str(root/(name+'.npy'))] = sha(root/(name+'.npy'))
        result = torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16)
        if not torch.isfinite(result).all():
            raise ValueError('nonfinite input')
        return result
    queries = [read(i, 'attn_q', [n,64,512]) for i in range(2)]
    keys = [read(i, 'attn_kv_quant', [n,512]) for i in range(2)]
    if not torch.equal(keys[0].view(torch.int16), keys[1].view(torch.int16)):
        raise ValueError('Q-only isolation requires identical quantized KV')
    recorded = [read(i, 'attn_o_raw', [n,64,512]) for i in range(2)]
    weights = Weights(a.checkpoint, 'artifacts/checkpoint/summary.json', 'artifacts/checkpoint/verification.json')
    sink = weights.tensor('layers.0.attn.attn_sink')[0].float()
    outputs = [[], []]
    for t in range(n):
        pad = 0 if t == 0 else 128-(t+1)
        kv = torch.cat([torch.zeros(pad,512,dtype=torch.bfloat16), keys[0][:t+1]])
        valid = torch.arange(kv.shape[0]) >= pad
        for i in range(2):
            outputs[i].append(sparse_attention(queries[i][t], kv, sink, valid))
    outputs = [torch.stack(rows) for rows in outputs]
    def compare(x,y):
        def raw(t): return t.contiguous().view(torch.uint16).numpy().view('|V2')
        m = metrics(raw(x), 'bfloat16', raw(y), 'bfloat16')
        if m['nonfinite_native'] or m['nonfinite_oracle']:
            raise ValueError('nonfinite output')
        return m
    replay = compare(recorded[1], outputs[1])
    if not replay['exact_bits']:
        raise ValueError('CPU baseline replay differs from recorded trace')
    result = {
        'scope': 'Q substitution on common CPU sparse attention, identical quantized KV; no output projection, routing or logits test',
        'cpu_baseline_replay': replay,
        'q_effect_on_common_cpu_attention': compare(outputs[0], outputs[1]),
        'native_attention_vs_cpu_on_native_q': compare(recorded[0], outputs[0]),
        'original_native_vs_cpu_attention': compare(recorded[0], recorded[1]),
        'per_token_q_effect': [compare(outputs[0][t],outputs[1][t]) for t in range(n)],
        'token_ids': manifests[0]['token_ids'], 'input_sha256': input_hashes,
        'manifest_sha256': {str(d/'manifest.json'): sha(d/'manifest.json') for d in (a.native,a.cpu)},
        'source_sha256': weights.sources, 'tensor_sha256': weights.tensor_hashes,
        'script_sha256': {str(f): sha(f) for f in (Path(__file__),Path(__file__).with_name('cpu_reference.py'),Path(__file__).with_name('compare_traces.py'))},
        'torch_version': torch.__version__, 'full_model_qualified': False,
    }
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for name in ('cpu_baseline_replay','q_effect_on_common_cpu_attention','native_attention_vs_cpu_on_native_q','original_native_vs_cpu_attention'):
        m=result[name]; print(name,m['bit_mismatch_count'],m['max_abs_diff'])


if __name__ == '__main__':
    main()
