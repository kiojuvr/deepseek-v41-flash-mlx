#!/usr/bin/env python3
"""Offline validation helper: cycle a text-token pattern to an exact length."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be fresh')
    if not 1 <= args.length <= 262144:
        parser.error('length must be in 1..262144')
    try:
        tokens = [int(value) for value in args.input.read_text().split()]
    except ValueError as error:
        parser.error(f'pattern contains a non-integer: {error}')
    if not tokens or any(token < 0 or token >= 129280 or token == 129264 for token in tokens):
        parser.error('pattern must contain valid text token IDs')
    expanded = [tokens[index % len(tokens)] for index in range(args.length)]
    args.output.write_text(' '.join(map(str, expanded)) + '\n')
    print(f'expanded {len(tokens)}-token pattern to {len(expanded)} tokens')


if __name__ == '__main__':
    main()
