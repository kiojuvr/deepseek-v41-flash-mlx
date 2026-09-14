"""Execute pinned official RoPE functions on CPU with frozen native inputs.

This is a local boundary oracle, not a CPU full-model trace or CUDA oracle.
"""
import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verification', type=Path, default=Path('artifacts/checkpoint/verification.json'))
    a = p.parse_args()
    out = a.output.resolve()
    if out.is_relative_to(a.checkpoint.resolve()) or out.is_relative_to(a.native.resolve()):
        p.error('output must be outside checkpoint and input trace')
    if out.exists():
        p.error('output exists; use a fresh directory')
    verification = json.loads(a.verification.read_text())
    source = a.checkpoint / 'inference/model.py'
    expected = next(x['sha256'] for x in verification['files'] if x['path'] == 'inference/model.py')
    if verification['status'] != 'verified' or sha(source) != expected:
        raise ValueError('official source identity mismatch')
    names = {'precompute_freqs_cis', 'apply_rotary_emb'}
    nodes = [x for x in ast.parse(source.read_text()).body
             if isinstance(x, ast.FunctionDef) and x.name in names]
    if {x.name for x in nodes} != names:
        raise ValueError('official RoPE functions missing')
    env = {'torch': torch, 'math': math, 'lru_cache': lru_cache}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
    manifest = json.loads((a.native / 'manifest.json').read_text())
    n = len(manifest['token_ids'])
    if not 1 <= n <= 128:
        raise ValueError('boundary probe supports 1..128 tokens starting at zero')
    torch.set_num_threads(4)
    freq = env['precompute_freqs_cis'](64, n, 0, 10000, 16, 32, 1)
    arrays, input_hashes = [], {}
    cases = [('attn_qb', 'attn_q', (1, n, 64, 512)),
             ('attn_kv_norm', 'attn_kv', (1, n, 512))]
    outputs = []
    for src, dest, shape in cases:
        name = 'encoder.layer0.' + src
        path = a.native / (name + '.npy')
        desc = next(x for x in manifest['arrays'] if x['name'] == name)
        raw = np.load(path, allow_pickle=False)
        if desc['dtype'] != 'bfloat16' or list(raw.shape) != desc['shape'] or raw.dtype.itemsize != 2:
            raise ValueError('invalid BF16 input trace')
        x = torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16).reshape(shape)
        env['apply_rotary_emb'](x[..., -64:], freq)
        data = x.reshape(shape[1:]).contiguous().view(torch.uint16).numpy().view('|V2')
        target = 'encoder.layer0.' + dest
        outputs.append((target, data))
        arrays.append({'name': target, 'dtype': 'bfloat16', 'shape': list(data.shape)})
        input_hashes[name] = sha(path)
    out.mkdir(parents=True)
    for name, data in outputs:
        np.save(out / (name + '.npy'), data, allow_pickle=False)
    result = {'schema_version': 1, 'token_ids': manifest['token_ids'], 'arrays': arrays,
              'scope': 'Official RoPE functions executed on CPU with frozen native layer0 inputs; not full-model/CUDA oracle',
              'official_cuda_executed': False, 'full_model_qualified': False,
              'source_sha256': expected, 'script_sha256': sha(__file__),
              'input_manifest_sha256': sha(a.native / 'manifest.json'), 'input_sha256': input_hashes,
              'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'output_sha256': {name: sha(out / (name + '.npy')) for name, _ in outputs}}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Wrote official-function CPU RoPE boundary trace:', out)


if __name__ == '__main__':
    main()
