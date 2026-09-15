#!/usr/bin/env python3
"""Compare native fixed-Exp sampling decisions with the official FP32 formula."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--native-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be fresh')

    verification = json.loads(args.verification.read_text())
    entry = next(item for item in verification['files'] if item['path'] == 'inference/model.py')
    source_sha = sha(args.official_source)
    if entry['status'] != 'verified' or source_sha != entry['sha256']:
        raise ValueError('official sampling source identity mismatch')
    source = args.official_source.read_text()
    required = ('def sample(logits, temperature: float = 1.0):',
                'logits = logits / max(temperature, 1e-5)',
                'probs = torch.softmax(logits, dim=-1, dtype=torch.float32)',
                'return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)')
    if not all(fragment in source for fragment in required):
        raise ValueError('official sampling formula changed')

    logits = torch.tensor([0.25, -1.0, 2.0, 0.5], dtype=torch.float32)
    exponentials = torch.tensor([0.8, 0.2, 100.0, 0.01], dtype=torch.float32)
    cases = []
    for name, temperature in [('fixed', 0.7), ('floor', 1e-8)]:
        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
        scores = probs / exponentials
        cases.append({'name': name, 'temperature': temperature, 'selected': int(scores.argmax()),
                      'probabilities': probs.tolist(), 'scores': scores.tolist()})

    native_text = args.native_log.read_text()
    match = re.search(r'fixed=(\d+) floor=(\d+)', native_text)
    if not match:
        raise ValueError('native sampling decisions missing')
    native = {'fixed': int(match.group(1)), 'floor': int(match.group(2))}
    expected = {case['name']: case['selected'] for case in cases}
    if native != expected:
        raise ValueError(f'native sampling decision mismatch: {native} != {expected}')
    result = {'schema_version': 1, 'status': 'pass', 'official_source': str(args.official_source),
              'official_source_sha256': source_sha, 'official_revision': entry['revision'],
              'verification_sha256': sha(args.verification), 'native_log_sha256': sha(args.native_log),
              'script_sha256': sha(Path(__file__)), 'dtype': 'float32',
              'logits': logits.tolist(), 'exponentials': exponentials.tolist(),
              'cases': cases, 'native_selected': native,
              'scope': 'fixed exponential variates and CPU formula; not official CUDA RNG or full generation'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: native fixed-exponential decisions match official FP32 formula', flush=True)


if __name__ == '__main__':
    main()
