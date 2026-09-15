#!/usr/bin/env python3
"""Compare native and external-reference MoE tie observations without promoting policy."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--external', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be fresh')
    native_path = args.native / 'manifest.json'
    external_path = args.external / 'manifest.json'
    native = json.loads(native_path.read_text())
    external = json.loads(external_path.read_text())
    if native.get('token_ids') != external.get('token_ids'):
        raise ValueError('trace token IDs differ')
    required = {'layer', 'token', 'sixth_id', 'seventh_id', 'sixth_score_bits', 'seventh_score_bits'}
    for manifest in (native, external):
        if any(not required <= set(record) for record in manifest.get('route_ties', [])):
            raise ValueError('tie record lacks exact comparison fields')
    native_records = native.get('route_ties', [])
    external_records = external.get('route_ties', [])
    native_by = {(record['layer'], record['token']): record for record in native_records}
    external_by = {(record['layer'], record['token']): record for record in external_records}
    if len(native_by) != len(native_records) or len(external_by) != len(external_records):
        raise ValueError('duplicate layer/token tie record')
    observations = []
    for key in sorted(set(native_by) | set(external_by)):
        local = native_by.get(key)
        other = external_by.get(key)
        same_boundary = local is not None and other is not None
        same_pair = same_boundary and all(local[name] == other[name] for name in
                                          ('sixth_id', 'seventh_id', 'sixth_score_bits', 'seventh_score_bits'))
        external_selected_local_sixth = (same_boundary and
            local['sixth_id'] in other.get('olmx_ids', []))
        observations.append({
            'layer': key[0], 'token': key[1], 'native': local, 'external': other,
            'same_boundary': same_boundary, 'same_id_and_score_bits': same_pair,
            'external_selected_native_sixth': external_selected_local_sixth,
        })
    report = {
        'schema_version': 1,
        'status': 'observations_recorded',
        'native_manifest_sha256': sha(native_path),
        'external_manifest_sha256': sha(external_path),
        'token_count': len(native['token_ids']),
        'native_tie_count': len(native_by),
        'external_tie_count': len(external_by),
        'observations': observations,
        'scope': ('Tie observations from local native MLX and pinned oMLX. '
                  'Official torch.topk does not specify a tie-break policy here; '
                  'agreement is diagnostic and does not qualify the lowest-ID policy.'),
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'native_ties': len(native_by),
                      'external_ties': len(external_by), 'observations': len(observations)}))


if __name__ == '__main__':
    main()
