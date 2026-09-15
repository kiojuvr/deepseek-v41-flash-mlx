"""Small CPU-only layer-0 Q projection/norm diagnostic; no CUDA oracle claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cpu_reference import Weights, Linear, activation, norm, sha
from compare_traces import checked_array, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-q', type=Path, help='optional dsv41-q-projection output')
    a = p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.checkpoint.resolve()):
        p.error('output must be fresh and outside checkpoint')
    torch.set_num_threads(4)
    manifest = json.loads((a.native / 'manifest.json').read_text())
    descriptors = {v['name']: v for v in manifest['arrays']}
    def read(suffix):
        desc = descriptors['encoder.layer0.' + suffix]
        if desc['dtype'] != 'bfloat16':
            raise ValueError('BF16 required')
        raw = checked_array(a.native, desc)
        return torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16)
    x, target = read('attn_in'), read('attn_qr')
    if x.ndim != 2 or x.shape[1] != 5120 or not 1 <= x.shape[0] <= 128:
        raise ValueError('requires position-zero 1..128 token layer-0 input')
    weights = Weights(a.checkpoint, 'artifacts/checkpoint/summary.json',
                      'artifacts/checkpoint/verification.json')
    linear = Linear(weights, 'layers.0.attn.wq_a')
    w = weights.tensor('layers.0.attn.q_norm.weight')[0]
    # Keep the trace's token-serial schedule; FP64 is a sensitivity probe only.
    def double_projection(row):
        if linear.scale is None:
            return F.linear(row.double(), linear.raw.double()).to(torch.bfloat16)
        q, scale, _ = activation(row)
        out = torch.zeros(1, linear.weight.shape[0], dtype=torch.float64)
        for b in range(q.shape[1]):
            partial = F.linear(q[:, b].double(), linear.weight[:, b*32:(b+1)*32].double())
            out += partial * scale[:, b:b+1].double() * linear.scale[:, b].double()
        return out.to(torch.bfloat16)
    projections = {
        'fp32': torch.cat([linear(row[None]) for row in x]),
        'fp64': torch.cat([double_projection(row[None]) for row in x]),
    }
    def raw(v):
        return v.contiguous().view(torch.uint16).numpy().view('|V2')
    results = {}
    for kind, projected in projections.items():
        f = projected.float()
        variance = f.square().mean(-1, keepdim=True) + 1e-20
        for method, value in (
                ('rsqrt', norm(projected, w)),
                ('inv_sqrt', (f * (1 / torch.sqrt(variance)) * w.float()).to(torch.bfloat16))):
            results[kind + '_' + method] = metrics(raw(target), 'bfloat16', raw(value), 'bfloat16')
    result = {
        'scope': 'CPU projection/norm sensitivity; missing native pre-norm Q prevents definitive attribution',
        'native_manifest_sha256': sha(a.native / 'manifest.json'),
        'input_sha256': {s: sha(a.native / ('encoder.layer0.' + s + '.npy')) for s in ('attn_in', 'attn_qr')},
        'source_sha256': weights.sources, 'tensor_sha256': weights.tensor_hashes,
        'script_sha256': {str(f): sha(f) for f in (Path(__file__), Path(__file__).with_name('cpu_reference.py'), Path(__file__).with_name('compare_traces.py'))},
        'torch_version': torch.__version__, 'variants': results,
        'projection_fp32_vs_fp64': metrics(raw(projections['fp32']), 'bfloat16', raw(projections['fp64']), 'bfloat16'),
        'full_model_qualified': False,
    }
    if a.native_q:
        qm = json.loads((a.native_q / 'manifest.json').read_text())
        if qm['token_ids'] != manifest['token_ids'] or qm['revision'] != weights.summary['revision']:
            raise ValueError('native Q trace identity mismatch')
        qdesc = {v['name']: v for v in qm['arrays']}
        def qread(suffix):
            return torch.from_numpy(checked_array(a.native_q, qdesc['encoder.layer0.' + suffix]).view(np.uint16).copy()).view(torch.bfloat16)
        native_qa, native_qr = qread('attn_qa'), qread('attn_qr')
        def metric(a, b):
            return metrics(raw(a), 'bfloat16', raw(b), 'bfloat16')
        result['native_q'] = {
            'manifest_sha256': sha(a.native_q / 'manifest.json'),
            'array_sha256': {s: sha(a.native_q / ('encoder.layer0.' + s + '.npy')) for s in ('attn_qa', 'attn_qr')},
            'reproduced_qr_vs_original_trace': metric(target, native_qr),
            'qa_vs_cpu_fp32': metric(native_qa, projections['fp32']),
            'qa_vs_cpu_fp64': metric(native_qa, projections['fp64']),
            'cpu_norm_on_native_qa_vs_native_qr': metric(native_qr, norm(native_qa, w)),
        }
        result['scope'] = 'Native pre-norm Q vs CPU projection; CPU norm on frozen native Q; no CUDA qualification'
        for name, m in result['native_q'].items():
            if isinstance(m, dict) and 'bit_mismatch_count' in m:
                print(name, m['bit_mismatch_count'], m['max_abs_diff'])
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, m in results.items():
        print(name, m['bit_mismatch_count'], m['max_abs_diff'])


if __name__ == '__main__':
    main()
