"""Offline layer-0 attention CPU formula trace from frozen native attn_in."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cpu_reference import Weights, Layer0Attention, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summary', type=Path, default=Path('artifacts/checkpoint/summary.json'))
    p.add_argument('--verification', type=Path, default=Path('artifacts/checkpoint/verification.json'))
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or out.is_relative_to(a.checkpoint.resolve()) or out.is_relative_to(a.native.resolve()):
        p.error('use a fresh output directory outside checkpoint/input')
    manifest = json.loads((a.native / 'manifest.json').read_text())
    name = 'encoder.layer0.attn_in'
    desc = next(v for v in manifest['arrays'] if v['name'] == name)
    path = a.native / (name + '.npy')
    raw = np.load(path, allow_pickle=False)
    n = len(manifest['token_ids'])
    if not 1 <= n <= 128 or desc['dtype'] != 'bfloat16' or desc['shape'] != [n, 5120] or raw.shape != (n, 5120) or raw.dtype.itemsize != 2:
        raise ValueError('invalid native attention input')
    x = torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16)
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite input')
    torch.set_num_threads(4)
    weights = Weights(a.checkpoint, a.summary, a.verification)
    print('Loading CPU layer-0 attention weights', flush=True)
    layer = Layer0Attention(weights)
    tensors = {}
    def emit(token, name, value):
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite: ' + name)
        tensors.setdefault(name, []).append(value.detach().clone())
        if name.endswith('.attn_out'):
            print(f'Completed token {token+1}/{n}', flush=True)
    with torch.inference_mode():
        layer.forward(x, emit)
    out.mkdir(parents=True)
    arrays = []
    for name, rows in tensors.items():
        value = torch.cat(rows)
        data = value.contiguous().view(torch.uint16).numpy().view('|V2')
        np.save(out / (name + '.npy'), data, allow_pickle=False)
        arrays.append({'name': name, 'dtype': 'bfloat16', 'shape': list(value.shape)})
    result = {'schema_version': 1, 'token_ids': manifest['token_ids'], 'arrays': arrays,
              'scope': 'CPU formula layer0 attention on frozen native attn_in; token-serial schedule, not full-model/CUDA oracle',
              'official_cuda_executed': False, 'full_model_qualified': False,
              'revision': weights.summary['revision'], 'source_sha256': weights.sources,
              'tensor_sha256': weights.tensor_hashes, 'torch_version': torch.__version__,
              'input_manifest_sha256': sha(a.native / 'manifest.json'), 'input_sha256': sha(path),
              'script_sha256': {str(f): sha(f) for f in [Path(__file__), Path(__file__).with_name('cpu_reference.py')]},
              'output_sha256': {v['name']: sha(out / (v['name'] + '.npy')) for v in arrays}}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Completed CPU attention trace. Independent comparison still required.', flush=True)


if __name__ == '__main__':
    main()
