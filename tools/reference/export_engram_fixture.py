"""Offline fixtures from pinned official Python; never imported by the native runtime."""
import argparse
import ast
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=2048)
    args = p.parse_args()
    if not 32 <= args.tokens <= 32768:
        p.error('--tokens must be 32..32768')
    root, out = args.checkpoint.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        p.error('output must be outside checkpoint')
    verify = json.loads(args.verification.read_text())
    assert verify['status'] == 'verified'
    pins = {f['path']: f for f in verify['files']}
    inputs = ('config.json', 'inference/config.json', 'inference/engram.py',
              'inference/model.py', 'tokenizer.json')
    for name in inputs:
        assert pins[name]['status'] == 'verified' and sha(root/name) == pins[name]['sha256'], name
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((root/'inference/config.json').read_text())
    cfg = SimpleNamespace(**config, max_batch_size=1, max_seq_len=args.tokens+8)
    spec = importlib.util.spec_from_file_location('dsv41_official_engram', root/'inference/engram.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))

    class Wrapper:
        backend_tokenizer = tokenizer
        def __len__(self):
            return tokenizer.get_vocab_size(with_added_tokens=True)

    layout = module.EngramLayout.from_args(cfg)
    oracle = module.NgramHashState(cfg, layout, Wrapper())
    metadata = {
        'schema_version': 1, 'revision': verify['revision'],
        'source_sha256': {name: sha(root/name) for name in inputs},
        'token_map': oracle.token_map.tolist(), 'compressed_vocab_size': cfg.engram_compressed_vocab_size,
        'pad_id': oracle.pad_id, 'layer_ids': list(layout.layer_ids),
        'head_dim': layout.head_dim, 'n_heads': layout.n_heads,
        'max_ngram_size': layout.max_ngram_size, 'num_embeddings': list(layout.num_embeddings),
        'primes': oracle.primes.tolist(), 'offsets': oracle.offsets.tolist(),
        'multipliers': oracle.multipliers.tolist(),
        'derivation': 'Official EngramLayout and NgramHashState; tokenizers normalization and NumPy RNG run offline.'}
    (out/'metadata.json').write_text(json.dumps(metadata, separators=(',', ':'))+'\n')

    # Synthetic coding-like fixture: repeat identifiers and protocol fragments while varying code.
    text = '\n'.join(
        f'<tool_result>src/cache_{i % 13}.cpp: test_session_{i % 29} passed</tool_result>\n'
        f'void update_cache_{i % 13}(Session& session) {{\n'
        f'  const auto token_id = session.tokens[{i % 37}];\n'
        f'  if (token_id != 0) session.append(token_id);\n}}\n'
        'Please inspect the diff, preserve ownership, and run the affected tests.\n'
        for i in range(args.tokens))
    anchor = tokenizer.encode(text, add_special_tokens=False).ids[:args.tokens]
    rng = np.random.default_rng(4112026)
    unrelated = rng.integers(0, len(Wrapper()), size=args.tokens, dtype=np.uint32).tolist()
    cases = []
    traces = {}
    for name, ids in [('coding', anchor), ('random_ids', unrelated)]:
        mask = [1]*len(ids)
        if name == 'coding':
            for start in range(97, len(ids), 257):
                mask[start:start+3] = [0]*min(3, len(ids)-start)
        tensor = torch.tensor([ids], dtype=torch.int64)
        live = torch.tensor([mask], dtype=torch.bool)
        hashes = oracle(tensor, 0, live).numpy().astype('<u8')
        oracle.cache.fill_(-1)
        parts = []
        pos = 0
        for chunk in [1, 2, 3, 127, 128, 129, len(ids)]:
            end = min(pos+chunk, len(ids))
            if end > pos:
                parts.append(oracle(tensor[:,pos:end], pos, live[:,pos:end]))
            pos = end
        assert np.array_equal(torch.cat(parts, dim=1).numpy(), hashes)
        (out/f'{name}.rows.u64').write_bytes(hashes.tobytes())
        cases.append({'name':name,'token_ids':ids,'token_mask':mask,
                      'expected_rows_file':f'{name}.rows.u64'})
        traces[name] = hashes.reshape(-1,len(layout.layer_ids),24)
    (out/'traces.json').write_text(json.dumps({'schema_version':1,'cases':cases}, separators=(',', ':'))+'\n')

    # Execute the official class alone; avoid importing GPU-only TileLang/model initialization.
    tree = ast.parse((root/'inference/model.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ParallelEngramEmbedding')
    namespace = {'torch':torch, 'nn':torch.nn, 'F':F, 'world_size':1, 'rank':0,
                 'fp8_block_size':32, 'scale_dtype':torch.float8_e8m0fnu}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root/'inference/model.py'), 'exec'), namespace)
    Embedding = namespace['ParallelEngramEmbedding']
    e = Embedding(256, 256)
    with torch.no_grad():
        e.weight.copy_(torch.arange(256,dtype=torch.uint8).repeat(256,1).view(torch.float8_e4m3fn))
        e.scale.copy_(torch.arange(256,dtype=torch.uint8)[:,None].repeat(1,8).view(torch.float8_e8m0fnu))
        allbits = e(torch.arange(256)).view(torch.uint16).numpy().astype('<u2')
    (out/'dequantize-all-bits.bf16').write_bytes(allbits.tobytes())

    # Independent positional reads supply real checkpoint rows to that same official class.
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    samples = []
    expected = []
    import struct
    for li, layer in enumerate(layout.layer_ids):
        wn, sn = f'layers.{layer}.engram.embed.weight', f'layers.{layer}.engram.embed.scale'
        shard = root/index[wn]
        assert index[wn] == index[sn]
        fd = os.open(shard, os.O_RDONLY)
        try:
            n = struct.unpack('<Q',os.pread(fd,8,0))[0]
            header = json.loads(os.pread(fd,n,8))
            row_ids = [0,1,layout.num_embeddings[li]-1] + traces['coding'][:24,li,:].reshape(-1).tolist()
            w = b''.join(os.pread(fd,256,8+n+header[wn]['data_offsets'][0]+r*256) for r in row_ids)
            s = b''.join(os.pread(fd,8,8+n+header[sn]['data_offsets'][0]+r*8) for r in row_ids)
            assert len(w)==len(row_ids)*256 and len(s)==len(row_ids)*8
            e = Embedding(len(row_ids),256)
            with torch.no_grad():
                e.weight.copy_(torch.frombuffer(bytearray(w),dtype=torch.float8_e4m3fn).reshape(-1,256))
                e.scale.copy_(torch.frombuffer(bytearray(s),dtype=torch.float8_e8m0fnu).reshape(-1,8))
                y = e(torch.arange(len(row_ids))).view(torch.uint16).numpy().astype('<u2')
            samples.append({'layer_id':layer,'row_ids':row_ids})
            expected.append(y.tobytes())
        finally:
            os.close(fd)
    (out/'samples.json').write_text(json.dumps(samples,separators=(',', ':'))+'\n')
    (out/'samples.bf16').write_bytes(b''.join(expected))
    names = ['metadata.json','traces.json','coding.rows.u64','random_ids.rows.u64',
             'dequantize-all-bits.bf16','samples.json','samples.bf16']
    provenance = {'revision':verify['revision'],'source_sha256':metadata['source_sha256'],
                  'exporter_sha256':sha(Path(__file__)),
                  'versions':{'python':sys.version.split()[0], 'torch':torch.__version__,
                              'numpy':np.__version__,
                              'tokenizers':importlib.metadata.version('tokenizers'),
                              'sympy':importlib.metadata.version('sympy')},
                  'fixture_sha256':{name:sha(out/name) for name in names},
                  'tokens_per_trace':args.tokens,
                  'workload':'synthetic coding-like transcript and seeded random token IDs; not a real agent session',
                  'official_oracle_scope':'NgramHashState and AST-extracted ParallelEngramEmbedding on CPU; no full-model oracle'}
    (out/'fixture-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Exported {len(metadata["token_map"])} token IDs, {cfg.engram_compressed_vocab_size} compressed IDs; '
          f'{2*args.tokens*48} row IDs, 65536 dequantization combinations, {sum(len(s["row_ids"]) for s in samples)} real rows.')


if __name__ == '__main__':
    main()
