"""Compare CPU RMSNorm stages on frozen native Q; substitutions isolate differences."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cpu_reference import Weights, sha
from compare_traces import checked_array, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native-q', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.checkpoint.resolve()):
        p.error('output must be fresh and outside checkpoint')
    torch.set_num_threads(4)
    m = json.loads((a.native_q / 'manifest.json').read_text())
    desc = {d['name']: d for d in m['arrays']}
    def read(suffix):
        d = desc['encoder.layer0.' + suffix]
        raw = checked_array(a.native_q, d)
        if d['dtype'] == 'bfloat16':
            return torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16)
        return torch.from_numpy(raw.copy())
    q, qr = read('attn_qa'), read('attn_qr')
    if q.shape != qr.shape or q.ndim != 2 or q.shape[1] != 1280 or not 1 <= q.shape[0] <= 128:
        raise ValueError('invalid Q shape')
    weights = Weights(a.checkpoint, 'artifacts/checkpoint/summary.json', 'artifacts/checkpoint/verification.json')
    if m['revision'] != weights.summary['revision']:
        raise ValueError('revision mismatch')
    w = weights.tensor('layers.0.attn.q_norm.weight')[0].float()
    native = {s: read('qnorm_' + s) for s in ('variance', 'adjusted', 'inv', 'normalized', 'weighted')}
    h = q.float()
    # Match the native token-serial execution schedule.
    variance = torch.cat([row[None].square().mean(-1, keepdim=True) for row in h])
    adjusted = variance + 1e-20
    inv = torch.rsqrt(adjusted)
    cpu = {'variance': variance, 'adjusted': adjusted, 'inv': inv,
           'normalized': h * inv, 'weighted': w * (h * inv)}
    def compare(x, y):
        dtype = 'bfloat16' if x.dtype == torch.bfloat16 else 'float32'
        def raw(t):
            return t.contiguous().view(torch.uint16).numpy().view('|V2') if dtype == 'bfloat16' else t.numpy()
        result = metrics(raw(x), dtype, raw(y), dtype)
        if result['nonfinite_native'] or result['nonfinite_oracle']:
            raise ValueError('nonfinite stage')
        denominator = torch.maximum(x.double().abs(), y.double().abs())
        relative = torch.where(denominator > 0, (x.double() - y.double()).abs() / denominator, 0.)
        result['max_symmetric_relative_diff'] = float(relative.max())
        return result
    stages = {s: compare(native[s], cpu[s]) for s in native}
    substitutions = {
        'inv_rsqrt_on_native_adjusted': compare(native['inv'], torch.rsqrt(native['adjusted'])),
        'inv_sqrt_on_native_adjusted': compare(native['inv'], 1 / torch.sqrt(native['adjusted'])),
        'normalized_on_native_inv': compare(native['normalized'], h * native['inv']),
        'weighted_on_native_normalized': compare(native['weighted'], w * native['normalized']),
        'cast_native_weighted_vs_native_qr': compare(qr, native['weighted'].to(torch.bfloat16)),
        'cpu_output_vs_native_qr': compare(qr, cpu['weighted'].to(torch.bfloat16)),
        'cpu_output_on_native_inv_vs_native_qr': compare(qr, (w * (h * native['inv'])).to(torch.bfloat16)),
    }
    differing = torch.nonzero(qr != cpu['weighted'].to(torch.bfloat16)).tolist()
    examples = [{'token': t, 'channel': c, 'native_bf16': float(qr[t,c]),
                 'cpu_bf16': float(cpu['weighted'].to(torch.bfloat16)[t,c]),
                 'native_weighted_fp32': float(native['weighted'][t,c]),
                 'cpu_weighted_fp32': float(cpu['weighted'][t,c])} for t,c in differing[:16]]
    output = cpu['weighted'].to(torch.bfloat16)
    def ordered_bf16(t):
        bits = t.contiguous().view(torch.uint16).numpy().astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)
    ulp = np.abs(ordered_bf16(qr) - ordered_bf16(output))
    labels = m.get('sample_ids', [str(i) for i in range(q.shape[0])])
    per_sample = [{'sample': labels[i], 'bit_mismatches': int((qr[i].view(torch.int16) != output[i].view(torch.int16)).sum()),
                   'max_bf16_ulp': int(ulp[i].max()), 'max_abs': float((qr[i].float() - output[i].float()).abs().max())}
                  for i in range(q.shape[0])]
    report = {'scope': 'CPU RMSNorm on frozen native Q; no routing/logits/CUDA qualification',
              'stages': stages, 'substitutions': substitutions, 'examples': examples,
              'per_sample': per_sample, 'max_bf16_ulp': int(ulp.max()),
              'input_sha256': {f.name: sha(f) for f in a.native_q.iterdir() if f.suffix in ('.npy', '.json')},
              'source_sha256': weights.sources, 'tensor_sha256': weights.tensor_hashes,
              'script_sha256': {str(f): sha(f) for f in (Path(__file__), Path(__file__).with_name('cpu_reference.py'), Path(__file__).with_name('compare_traces.py'))},
              'torch_version': torch.__version__, 'full_model_qualified': False}
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    for group in (stages, substitutions):
        for s,v in group.items():
            print(s, v['bit_mismatch_count'], v['max_abs_diff'])
    print(examples)


if __name__ == '__main__':
    main()
