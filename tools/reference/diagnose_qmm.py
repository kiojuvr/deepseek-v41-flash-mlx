"""Diagnose whether the layer-0 wq_b gap is a 2D-per-row vs 3D-batched quantized_matmul difference.

Native `PackedLinearReference` calls `mx.quantized_matmul` on a 2D row; oMLX calls it on the
whole `[1, seqlen, K]` activation. This runs both shapes on the exact traced `attn_qr` input
with the pinned oMLX repack of the real `wq_b` weight and compares the outputs.
"""
import argparse
import ast
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

PIN = 'b390b31e0c6831225fed0f24d278eb1db7fcb68b'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_bf16(path):
    raw = np.load(path, allow_pickle=False)
    u = raw.view(np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def read_raw(checkpoint, name):
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    path = checkpoint / index[name]
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        spec = json.loads(f.read(length))[name]
        f.seek(8 + length + spec['data_offsets'][0])
        data = f.read(spec['data_offsets'][1] - spec['data_offsets'][0])
    return np.frombuffer(data, dtype=np.uint8).reshape(spec['shape']), spec['dtype']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--omlx', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--prefix', default='attn.wq_b')
    a = p.parse_args()

    qr = load_bf16(a.native / f'encoder.layer{a.layer}.attn_qr.npy')  # [tokens,1280]
    wr, wd = read_raw(a.checkpoint, f'layers.{a.layer}.{a.prefix}.weight')
    sr, sd = read_raw(a.checkpoint, f'layers.{a.layer}.{a.prefix}.scale')

    convert = 'omlx/patches/deepseek_v41/convert.py'
    pinned = subprocess.check_output(['git', '-C', str(a.omlx), 'show', PIN + ':' + convert])
    tree = ast.parse(pinned)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'repack_weight')
    ns = {'mx': mx, 'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(a.omlx / convert), 'exec'), ns)
    weights, config = ns['repack_weight'](wr, wd, sr, sd)
    mx.eval(weights['weight'], weights['scales'])

    x2 = mx.array(qr.astype(np.float32)).astype(mx.bfloat16)
    x3 = x2.reshape(1, x2.shape[0], x2.shape[1])
    out2 = mx.quantized_matmul(x2, weights['weight'], weights['scales'], group_size=32, **config)
    out3 = mx.quantized_matmul(x3, weights['weight'], weights['scales'], group_size=32, **config)
    mx.eval(out2, out3)
    b2 = np.array(out2.view(mx.uint16))
    b3 = np.array(out3.view(mx.uint16)).reshape(b2.shape)
    report = {
        'layer': a.layer, 'prefix': a.prefix,
        'config': config,
        'per_row_2d_shape': list(out2.shape),
        'batched_3d_shape': list(out3.shape),
        'bits_mismatch_2d_vs_3d': int(np.count_nonzero(b2 != b3)),
        'elements': int(b2.size),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
