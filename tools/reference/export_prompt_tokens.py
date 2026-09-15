"""Tokenize a fixed teacher-forced prompt with the checkpoint tokenizer.

Writes a whitespace-separated token-id file usable by dsv41-text-trace --tokens-file and
tools/reference/trace_omlx.py --tokens-file. The text is a fixed coding-agent-style prompt;
it is a token-ID fixture, not a rendered chat template.
"""
import argparse
import hashlib
import json
from pathlib import Path

PROMPT = (
    "You are a coding assistant. Fix the following Python function so it returns the sum of "
    "squares of a list of integers.\n\n"
    "def sum_squares(values):\n"
    "    total = 0\n"
    "    for value in values:\n"
    "        total += value\n"
    "    return total\n\n"
    "Show the corrected function and explain the bug."
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=128)
    a = p.parse_args()
    if a.output.resolve().is_relative_to(a.checkpoint.resolve()):
        raise ValueError('checkpoint read-only')

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.checkpoint / 'tokenizer.json'))
    ids = tokenizer.encode(PROMPT, add_special_tokens=False).ids[: a.max_tokens]
    if not ids:
        raise ValueError('empty tokenization')
    a.output.write_text(' '.join(str(i) for i in ids) + '\n')
    manifest = {
        'schema_version': 1,
        'prompt': PROMPT,
        'token_count': len(ids),
        'token_ids': ids,
        'tokenizer_sha256': sha(a.checkpoint / 'tokenizer.json'),
        'output_sha256': sha(a.output),
        'scope': 'Fixed teacher-forced coding-agent-style prompt token IDs; no chat template rendering.',
    }
    (a.output.parent / (a.output.name + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'token_count': len(ids), 'first_ids': ids[:16]}, indent=2))


if __name__ == '__main__':
    main()
