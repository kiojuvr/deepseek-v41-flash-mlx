"""Adversarial small checkpoints; no model, network, MLX, or Python runtime dependency."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())
SHARD = 'model-00001-of-00001.safetensors'
REV = 'a' * 40


class AtlasContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'model'
        self.root.mkdir()
        self.out = self.base / 'out'
        self.manifest = self.base / 'manifest.json'
        self.header = {}
        offset = 0
        specs = [('norm.weight', 'BF16', [32]),
                 ('layers.1.engram.embed.weight', 'F8_E4M3', [2, 32]),
                 ('layers.1.engram.embed.scale', 'F8_E8M0', [2, 1]),
                 ('layers.1.attn.wo_a.weight', 'F8_E4M3', [32, 32]),
                 ('layers.1.attn.wo_a.scale', 'F8_E8M0', [1, 1])]
        for w in ('w1', 'w2', 'w3'):
            specs += [(f'layers.0.ffn.experts.0.{w}.weight', 'I8', [32, 16]),
                      (f'layers.0.ffn.experts.0.{w}.scale', 'F8_E8M0', [32, 1])]
        for name, dtype, shape in specs:
            size = (2 if dtype == 'BF16' else 1)
            for d in shape:
                size *= d
            self.header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset+size])
            offset += size
        self.payload = bytes(offset)
        cfg = {'model_type': 'deepseek_v41', 'text_config': {
            'num_hidden_layers': 2, 'num_nextn_predict_layers': 1,
            'n_routed_experts': 1, 'dspark_n_routed_experts': 1,
            'hidden_size': 32, 'moe_intermediate_size': 32,
            'engram_layer_ids': [1], 'engram_num_embeddings': [2], 'engram_head_dim': 32}}
        (self.root/'config.json').write_text(json.dumps(cfg))
        self.write_shard()
        self.write_index()
        self.refresh_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_shard(self, raw=None, payload=None):
        data = json.dumps(self.header).encode() if raw is None else raw
        (self.root/SHARD).write_bytes(struct.pack('<Q', len(data)) + data +
                                     (self.payload if payload is None else payload))

    def write_index(self):
        (self.root/'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': len(self.payload)},
            'weight_map': {k: SHARD for k in self.header}}))

    def refresh_manifest(self):
        siblings = []
        for p in sorted(self.root.iterdir()):
            if not p.is_file():
                continue
            data = p.read_bytes()
            entry = {'rfilename': p.name, 'size': len(data),
                     'blobId': hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()}
            if p.suffix == '.safetensors':
                entry['lfs'] = {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            siblings.append(entry)
        self.manifest.write_text(json.dumps({'sha': REV, 'siblings': siblings}))

    def run_atlas(self, code=0, verify=True, output=None):
        cmd = [BINARY, '--checkpoint', str(self.root), '--output', str(output or self.out),
               '--manifest', str(self.manifest)]
        if verify:
            cmd += ['--verify-payload']
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return json.loads((self.out/'summary.json').read_text()) if code != 1 else None

    def reject_structure(self, phrase):
        self.write_shard()
        self.refresh_manifest()
        s = self.run_atlas(2)
        self.assertTrue(any(phrase in i for i in s['issues']), s['issues'])

    def test_verified_accounting(self):
        s = self.run_atlas()
        self.assertEqual(s['status'], 'verified')
        self.assertEqual(s['payload_bytes'], len(self.payload))
        self.assertEqual(s['by_owner']['engram_backing']['storage_bytes'], 66)
        self.assertEqual(s['by_component']['moe_routed']['logical_parameters'], 3*32*32)
        self.assertEqual(s['by_component']['moe_routed']['storage_bytes'], 3*(512+32))
        rows = {r['name']: r for r in map(json.loads, (self.out/'atlas.jsonl').read_text().splitlines())}
        row = rows['layers.0.ffn.experts.0.w1.weight']
        self.assertEqual(row['logical_dtype'], 'FP4_E2M1')
        self.assertEqual(row['logical_shape'], [32, 32])
        self.assertEqual(row['group_size'], [1, 32])
        self.assertEqual(row['digest_scope'], 'whole_shard')
        self.assertIsNone(row['runtime_bytes'])
        self.assertTrue(all(r['completeness'] == 'verified' for r in rows.values()))

    def test_header_only_not_verified(self):
        s = self.run_atlas(verify=False)
        self.assertEqual(s['status'], 'header-valid')
        rows = list(map(json.loads, (self.out/'atlas.jsonl').read_text().splitlines()))
        self.assertTrue(all(r['digest'] is None for r in rows))

    def test_payload_corruption(self):
        data = bytearray((self.root/SHARD).read_bytes())
        data[-1] ^= 1
        (self.root/SHARD).write_bytes(data)
        s = self.run_atlas(2)
        self.assertTrue(any('digest mismatch' in i for i in s['issues']))
        rows = list(map(json.loads, (self.out/'atlas.jsonl').read_text().splitlines()))
        self.assertTrue(all(r['completeness'] == 'failed' for r in rows))

    def test_auxiliary_corruption(self):
        p = self.root/'config.json'
        p.write_text(p.read_text().replace('"hidden_size": 32', '"hidden_size": 64'))
        s = self.run_atlas(2)
        self.assertTrue(any('config.json: upstream digest mismatch' in i for i in s['issues']))

    def test_duplicate_header_keys(self):
        raw = json.dumps(self.header).encode()
        raw = raw[:-1] + b',"norm.weight":' + json.dumps(self.header['norm.weight']).encode() + b'}'
        self.write_shard(raw=raw)
        self.refresh_manifest()
        self.assertTrue(any('duplicate JSON key' in i for i in self.run_atlas(2)['issues']))

    def test_duplicate_index_keys(self):
        p = self.root/'model.safetensors.index.json'
        p.write_text('{"weight_map":{},"weight_map":{}}')
        self.refresh_manifest()
        self.run_atlas(1)

    def test_shape_overflow(self):
        self.header['norm.weight']['shape'] = [2**63, 4]
        self.reject_structure('overflow')

    def test_negative_float_bool_dimensions(self):
        for d in (-1, 32.0, True):
            with self.subTest(d=d):
                self.header['norm.weight']['shape'] = [d]
                self.reject_structure('unsigned JSON integer')

    def test_shape_byte_mismatch(self):
        self.header['norm.weight']['shape'] = [31]
        self.reject_structure('shape/dtype/payload')

    def test_offset_overlap(self):
        self.header['layers.1.engram.embed.weight']['data_offsets'] = [0, 64]
        self.reject_structure('gap or overlap')

    def test_offset_bounds(self):
        self.header['norm.weight']['data_offsets'] = [0, 2**64-1]
        self.reject_structure('out of bounds')

    def test_trailing_payload(self):
        self.write_shard(payload=self.payload+b'x')
        self.refresh_manifest()
        self.assertTrue(any('trailing payload' in i for i in self.run_atlas(2)['issues']))

    def test_truncated_header(self):
        (self.root/SHARD).write_bytes(struct.pack('<Q', 1000)+b'{}')
        self.refresh_manifest()
        self.assertTrue(any('header length' in i for i in self.run_atlas(2)['issues']))

    def test_unsupported_dtype(self):
        self.header['norm.weight']['dtype'] = 'F16'
        self.reject_structure('unsupported checkpoint storage dtype')

    def test_missing_shard(self):
        (self.root/SHARD).rename(self.root/(SHARD+'.incomplete'))
        s = self.run_atlas(2)
        self.assertEqual(s['valid_headers'], 0)
        self.assertEqual(s['status'], 'partial_or_invalid')

    def test_index_mismatch(self):
        del self.header['norm.weight']
        self.write_shard()
        self.refresh_manifest()
        self.run_atlas(2)

    def test_unknown_owner(self):
        self.header['alien.weight'] = self.header.pop('norm.weight')
        self.write_index()
        self.reject_structure('unknown semantic owner')

    def test_orphan_scale(self):
        self.header['layers.1.attn.wo_b.scale'] = self.header.pop('layers.1.attn.wo_a.scale')
        self.write_index()
        self.reject_structure('orphan scale')

    def test_bad_scale_geometry(self):
        self.header['layers.0.ffn.experts.0.w1.scale']['shape'] = [1, 32]
        self.reject_structure('scale group/shape mismatch')

    def test_extra_shard(self):
        (self.root/'extra.safetensors').write_bytes((self.root/SHARD).read_bytes())
        s = self.run_atlas(2)
        self.assertTrue(any('unexpected shard' in i for i in s['issues']))

    def test_unsafe_manifest_path(self):
        j = json.loads(self.manifest.read_text())
        j['siblings'][0]['rfilename'] = '../outside'
        self.manifest.write_text(json.dumps(j))
        self.run_atlas(1)

    def test_symlink_shard(self):
        target = self.base/'outside.safetensors'
        (self.root/SHARD).rename(target)
        (self.root/SHARD).symlink_to(target)
        s = self.run_atlas(2)
        self.assertTrue(any('symlink rejected' in i for i in s['issues']))

    def test_output_cannot_touch_checkpoint(self):
        self.run_atlas(1, output=self.root/'reports')


if __name__ == '__main__':
    unittest.main()
