"""Fixed synthetic exploration set, independent of the observed coding trace.

This is diagnostic/calibration data, not a held-out qualification set.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cpu_reference import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('use a fresh output directory')
    rng = np.random.default_rng(20260915)
    rows, labels = [], []
    def add(name, row):
        rows.append(torch.from_numpy(np.asarray(row, dtype=np.float32)).to(torch.bfloat16))
        labels.append(name)
    add('zero', np.zeros(1280))
    for exponent in (-60, -34, -33, -32, -10, 0, 10, 30):
        scale = 2.**exponent
        add(f'constant/{exponent}', np.full(1280, scale))
        add(f'alternating/{exponent}', np.tile([scale, -scale], 640))
        sparse = np.zeros(1280); sparse[469] = scale
        add(f'sparse/{exponent}', sparse)
        for j in range(6):
            add(f'normal/{exponent}/{j}', rng.standard_normal(1280) * scale)
    # Adjacent BF16 input perturbations explore rounding sensitivity without
    # selecting cases based on either backend's output discrepancy.
    base = torch.from_numpy(rng.uniform(.5, 2., 1280).astype(np.float32)).to(torch.bfloat16)
    for direction in (-1, 0, 1):
        bits = base.view(torch.int16).clone()
        bits[::3] += direction
        add(f'neighbor/{direction}', bits.view(torch.bfloat16).float().numpy())
    data = torch.stack(rows).view(torch.uint16).numpy().view('|V2')
    a.output.mkdir(parents=True)
    path = a.output / 'encoder.layer0.attn_qa.npy'
    np.save(path, data, allow_pickle=False)
    (a.output / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'sample_ids': labels, 'seed': 20260915,
        'scope': 'Synthetic calibration Q inputs; not token IDs or qualification data',
        'arrays': [{'name': 'encoder.layer0.attn_qa', 'shape': list(data.shape), 'dtype': 'bfloat16'}],
        'sha256': {path.name: sha(path)}, 'script_sha256': sha(__file__),
        'numpy_version': np.__version__, 'torch_version': torch.__version__,
    }, indent=2) + '\n')
    print(f'Wrote {len(rows)} independent synthetic Q rows')


if __name__ == '__main__':
    main()
