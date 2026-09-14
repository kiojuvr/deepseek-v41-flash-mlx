"""Fast regression tests: report ordering and mismatch semantics, no model load."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np

spec = importlib.util.spec_from_file_location('compare_traces',
    Path(__file__).resolve().parents[2] / 'tools/reference/compare_traces.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class TraceComparisonTests(unittest.TestCase):
    def test_incompatible_trace_identity(self):
        valid = {'token_ids': [0, 42], 'arrays': [{'name': 'logits'}]}
        for other in ({'token_ids': [42, 0], 'arrays': [{'name': 'logits'}]},
                      {'token_ids': [0, 42], 'arrays': [{'name': 'other'}]},
                      {'token_ids': [0, 42], 'arrays': [{'name': 'logits'}, {'name': 'logits'}]}):
            with self.assertRaises(ValueError):
                compare.validate_pair(valid, other)

    def test_manifest_storage_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / 'logits.npy', np.zeros((2, 3), np.float32))
            for shape, dtype in (([3, 2], 'float32'), ([2, 3], 'bfloat16')):
                with self.assertRaises(ValueError):
                    compare.checked_array(root, {'name': 'logits', 'shape': shape, 'dtype': dtype})
            self.assertEqual(compare.checked_array(root, {'name': 'logits', 'shape': [2, 3], 'dtype': 'float32'}).shape, (2, 3))

    def test_rope_precedes_attention_output(self):
        names = ['encoder.layer0.attn_out', 'encoder.layer0.attn_q',
                 'encoder.layer0.attn_qb', 'encoder.layer0.post_attn']
        self.assertEqual(sorted(names, key=compare.boundary_order), [
            'encoder.layer0.attn_qb', 'encoder.layer0.attn_q',
            'encoder.layer0.attn_out', 'encoder.layer0.post_attn'])

    def test_oracle_only_nonfinite_is_failure(self):
        m = compare.metrics(np.array([1.], np.float32), 'float32',
                            np.array([np.nan], np.float32), 'float32')
        self.assertTrue(compare.diverges(m, 1e-6))
        self.assertIsNone(m['argmax_equal'])

    def test_logical_dtype_matters(self):
        raw = np.array([0], np.uint16)
        m = compare.metrics(raw, 'bfloat16', raw, 'float16')
        self.assertFalse(m['exact_bits'])
        self.assertTrue(compare.diverges(m, 0))

    def test_signed_zero_bits_separate_from_numerical_tolerance(self):
        m = compare.metrics(np.array([0., 1.], np.float32), 'float32',
                            np.array([-0., 1.], np.float32), 'float32')
        self.assertEqual(m['bit_mismatch_count'], 1)
        self.assertFalse(m['exact_bits'])
        self.assertFalse(compare.diverges(m, 0))

    def test_bf16_raw_storage(self):
        a = np.array([0x3f80, 0x3f80], np.uint16).view('|V2')
        b = np.array([0x3f80, 0x3f81], np.uint16).view('|V2')
        m = compare.metrics(a, 'bfloat16', b, 'bfloat16')
        self.assertEqual(m['bit_mismatch_count'], 1)
        self.assertEqual(m['max_abs_diff'], 1/128)


if __name__ == '__main__':
    unittest.main()
