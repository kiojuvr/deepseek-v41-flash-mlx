"""Small CPU arithmetic checks, no checkpoint load."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/reference'))
from cpu_reference import activation, sparse_attention, norm


class CPUReferenceTests(unittest.TestCase):
    def test_activation_floor(self):
        x = torch.full((1, 32), 2**-60, dtype=torch.bfloat16)
        q, scale, decoded = activation(x)
        self.assertTrue(torch.all(q.float() == 0))
        self.assertTrue(torch.all(decoded == 0))
        self.assertEqual(float(scale[0, 0]), 2**-22)

    def test_sink_with_empty_first_tile(self):
        torch.set_num_threads(2)
        for live in (1, 63, 64, 65, 127, 128):
            y = sparse_attention(torch.zeros(64, 512, dtype=torch.bfloat16),
                                 torch.ones(128, 512, dtype=torch.bfloat16),
                                 torch.zeros(64), torch.arange(128) >= 128-live)
            expected = torch.full_like(y, live/(live+1))
            self.assertTrue(torch.equal(y, expected), live)

    def test_zero_norm(self):
        self.assertTrue(torch.equal(norm(torch.zeros(1, 512, dtype=torch.bfloat16),
                                         torch.ones(512, dtype=torch.bfloat16)),
                                    torch.zeros(1, 512, dtype=torch.bfloat16)))


if __name__ == '__main__':
    unittest.main()
