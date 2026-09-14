"""Offline CPU formula reference. No CUDA execution and no native output fitting."""
import ast
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Weights:
    def __init__(self, root, summary, verification):
        self.root = Path(root).resolve()
        self.summary = json.loads(Path(summary).read_text())
        proof = json.loads(Path(verification).read_text())
        if self.summary['status'] != 'verified' or proof['status'] != 'verified' or self.summary['revision'] != proof['revision']:
            raise ValueError('M1 identity mismatch')
        self.sources = {}
        for name in ('inference/model.py', 'inference/kernel.py', 'inference/config.json'):
            digest = next(v['sha256'] for v in proof['files'] if v['path'] == name)
            if sha(self.root / name) != digest:
                raise ValueError('official source changed: ' + name)
            self.sources[name] = digest
        self.mapping = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.tensor_hashes = {}

    def tensor(self, name):
        shard = self.mapping[name]
        if Path(shard).name != shard:
            raise ValueError('unsafe shard name')
        with (self.root / shard).open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 64 * 1024**2:
                raise ValueError('invalid header size')
            header = f.read(size)
            if hashlib.sha256(header).hexdigest() != self.summary['shards'][shard]['header_sha256']:
                raise ValueError('checkpoint header changed')
            desc = json.loads(header)[name]
            begin, end = desc['data_offsets']
            if begin < 0 or end < begin or end - begin > 256 * 1024**2:
                raise ValueError('unsupported tensor size')
            f.seek(8 + size + begin)
            data = f.read(end - begin)
        if len(data) != end - begin:
            raise ValueError('short checkpoint read')
        self.tensor_hashes[name] = hashlib.sha256(data).hexdigest()
        dtype = {'BF16': torch.bfloat16, 'F32': torch.float32,
                 'F8_E4M3': torch.float8_e4m3fn, 'F8_E8M0': torch.float8_e8m0fnu,
                 'I8': torch.uint8, 'U8': torch.uint8}[desc['dtype']]
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(desc['shape']), desc['dtype']


def official_rope(root):
    names = {'precompute_freqs_cis', 'apply_rotary_emb'}
    path = Path(root) / 'inference/model.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('missing official RoPE functions')
    env = {'torch': torch, 'math': math, 'lru_cache': lru_cache}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env['precompute_freqs_cis'], env['apply_rotary_emb']


def activation(x):
    blocks = x.float().reshape(x.shape[0], -1, 32)
    maximum = blocks.abs().amax(-1).clamp_min(1e-4)
    bits = (maximum * (1.0 / 448.0)).view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    q = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    decoded = (q.float() * scale.unsqueeze(-1)).reshape_as(x).to(torch.bfloat16)
    return q, scale, decoded


def norm(x, weight):
    # Official RMSNorm: FP32 variance, then multiply weight, BF16 output.
    f = x.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-20) * weight.float()).to(x.dtype)


class Linear:
    def __init__(self, weights, prefix):
        w, self.kind = weights.tensor(prefix + '.weight')
        self.raw = w
        self.scale = None
        if self.kind in ('F8_E4M3', 'I8'):
            s, kind = weights.tensor(prefix + '.scale')
            if kind != 'F8_E8M0':
                raise ValueError('unsupported linear scale')
            self.scale = s.float() if self.kind == 'I8' else s.float().repeat_interleave(32, 0)[:w.shape[0]]
        elif self.kind == 'I8':
            self.bits = 4
        elif self.kind != 'BF16':
            raise ValueError('attention CPU linear requires BF16 or FP8')
        if self.kind == 'I8':
            lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
            codes = torch.stack((w & 15, w >> 4), -1).reshape(w.shape[0], -1).long()
            self.weight = lut[codes]
        else:
            self.weight = w.float()

    def decoded_weight(self):
        if self.kind == 'I8':
            return (self.weight * self.scale.repeat_interleave(32, 1)).to(torch.bfloat16)
        if self.scale is None:
            return self.raw
        return (self.weight * self.scale.repeat_interleave(32, 1)).to(torch.bfloat16)

    def __call__(self, x):
        if self.kind == 'BF16':
            return F.linear(x.float(), self.weight).to(x.dtype)
        q, scale, _ = activation(x)
        if self.kind == 'I8':
            # E2M1 values are already decoded in self.weight; use the same
            # explicit ascending 32-channel accumulation as the FP8 path.
            out = torch.zeros(x.shape[0], self.weight.shape[0], dtype=torch.float32)
            for b in range(q.shape[1]):
                part = F.linear(q[:, b].float(), self.weight[:, b*32:(b+1)*32])
                out = out + (part * scale[:, b:b+1]) * self.scale[:, b]
            return out.to(torch.bfloat16)
        out = torch.zeros(x.shape[0], self.weight.shape[0], dtype=torch.float32)
        # Explicit ascending 32-channel block sum; not CUDA reduction order.
        for b in range(q.shape[1]):
            part = F.linear(q[:, b].float(), self.weight[:, b*32:(b+1)*32])
            out = out + (part * scale[:, b:b+1]) * self.scale[:, b]
        return out.to(torch.bfloat16)


def sparse_attention(q, kv, sink, valid=None):
    """One token, official ordered slots; fixed 64-row tiled formula."""
    if valid is None:
        valid = torch.ones(kv.shape[0], dtype=torch.bool)
    maximum = torch.full((q.shape[0], 1), -1e30)
    denominator = torch.zeros_like(maximum)
    acc = torch.zeros_like(q, dtype=torch.float32)
    for first in range(0, kv.shape[0], 64):
        mask = valid[first:first+64]
        keys = torch.where(mask[:, None], kv[first:first+64].float(), 0.)
        score = F.linear(q.float(), keys) * (512**-0.5)
        score = score.masked_fill(~mask[None, :], -torch.inf)
        next_max = torch.maximum(maximum, score.amax(-1, keepdim=True))
        rescale = torch.exp(maximum - next_max)
        exponent = torch.exp(score - next_max)
        denominator = denominator * rescale + exponent.sum(-1, keepdim=True)
        acc = acc * rescale + exponent.to(torch.bfloat16).float() @ keys
        maximum = next_max
    return (acc / (denominator + torch.exp(sink[:, None] - maximum))).to(torch.bfloat16)


class Layer0Attention:
    def __init__(self, weights):
        self.qa = Linear(weights, 'layers.0.attn.wq_a')
        self.qb = Linear(weights, 'layers.0.attn.wq_b')
        self.kv = Linear(weights, 'layers.0.attn.wkv')
        self.wo = Linear(weights, 'layers.0.attn.wo_b')
        self.grouped = Linear(weights, 'layers.0.attn.wo_a').decoded_weight().reshape(8, 1024, 4096)
        self.qnorm = weights.tensor('layers.0.attn.q_norm.weight')[0]
        self.kvnorm = weights.tensor('layers.0.attn.kv_norm.weight')[0]
        self.sink = weights.tensor('layers.0.attn.attn_sink')[0].float()
        self.freqs, self.rope = official_rope(weights.root)

    def forward(self, x, emit):
        n = x.shape[0]
        freqs = self.freqs(64, n, 0, 10000, 16, 32, 1)
        window = []
        for t in range(n):
            h = x[t:t+1]
            qr = norm(self.qa(h), self.qnorm)
            qb = self.qb(qr)
            q = qb.reshape(1, 1, 64, 512).clone()
            self.rope(q[..., -64:], freqs[t:t+1])
            kn = norm(self.kv(h), self.kvnorm)
            k = kn.reshape(1, 1, 512).clone()
            self.rope(k[..., -64:], freqs[t:t+1])
            quant = activation(k.reshape(1, 512))[2]
            window.append(quant)
            keys = torch.cat(window[-128:])
            pad = 0 if t == 0 else 128 - keys.shape[0]
            keys = torch.cat([torch.zeros(pad, 512, dtype=torch.bfloat16), keys])
            valid = torch.arange(keys.shape[0]) >= pad
            raw = sparse_attention(q.reshape(64, 512), keys, self.sink, valid)
            o = raw.reshape(1, 1, 64, 512).clone()
            self.rope(o[..., -64:], freqs[t:t+1], True)
            groups = o.reshape(8, 1, 4096).float()
            projected = torch.bmm(groups, self.grouped.float().transpose(1, 2)).to(torch.bfloat16)
            out = self.wo(projected.reshape(1, 8192))
            for name, value in [('attn_in', h), ('attn_qr', qr), ('attn_qb', qb),
                                ('attn_q', q.reshape(1, 64, 512)), ('attn_kv_norm', kn),
                                ('attn_kv', k.reshape(1, 512)), ('attn_kv_quant', quant),
                                ('attn_o_raw', raw.reshape(1, 64, 512)), ('attn_o', o.reshape(1, 64, 512)),
                                ('attn_out', out)]:
                emit(t, 'encoder.layer0.' + name, value)
