"""Compare two backbone traces (native / oracle) boundary by boundary.

Both inputs are directories with manifest.json and <name>.npy arrays following the
dsv41-text-trace schema. MLX saves bfloat16 as raw 2-byte storage, so the logical
dtype from the manifest drives decoding. Reports shape, bit mismatch count, max/mean
absolute difference, nonfinite count and argmax agreement per boundary, and names the
first diverging boundary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_manifest(root):
    return json.loads((root / 'manifest.json').read_text())


def load_array(root, name):
    return np.load(root / (name + '.npy'), allow_pickle=False)


def validate_pair(native, oracle):
    tokens = native.get('token_ids')
    if (not isinstance(tokens, list) or not tokens or
            any(type(t) is not int or t < 0 for t in tokens) or
            tokens != oracle.get('token_ids')):
        raise ValueError('traces require identical nonempty token_ids')
    indexed = []
    for manifest in (native, oracle):
        arrays = {}
        for desc in manifest['arrays']:
            name = desc['name']
            if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in arrays:
                raise ValueError('invalid or duplicate boundary name')
            arrays[name] = desc
        indexed.append(arrays)
    if not set(indexed[0]) & set(indexed[1]):
        raise ValueError('no common boundaries')
    return indexed


def checked_array(root, desc):
    raw = load_array(root, desc['name'])
    if list(raw.shape) != desc['shape']:
        raise ValueError('array shape differs from manifest: ' + desc['name'])
    dtype = desc['dtype']
    if dtype == 'bfloat16':
        valid = raw.dtype == np.dtype('|V2') or raw.dtype == np.dtype('<u2')
    else:
        valid = dtype in ('float32', 'float16', 'uint8', 'uint16', 'uint32', 'int32') and raw.dtype == np.dtype(dtype)
    if not valid:
        raise ValueError('storage differs from logical dtype: ' + desc['name'])
    return raw


def boundary_order(name):
    """Chronological execution order of the traced boundaries."""
    suffix = {name: i for i, name in enumerate((
        'attn_in', 'attn_qr', 'attn_qb', 'attn_q', 'attn_kv_norm',
        'attn_kv', 'attn_kv_quant', 'attn_o_raw', 'attn_o',
        'attn_out', 'post_attn', 'ffn_in', 'moe_out',
        'hidden', 'pre_mix'))}
    parts = name.rsplit('.', 2)
    if len(parts) == 3 and parts[0] in ('encoder', 'decoder') and parts[1].startswith('layer'):
        layer = int(parts[1][len('layer'):])
        base = (1 + layer) if parts[0] == 'encoder' else (200 + layer)
        return (base, suffix.get(parts[2], 99), name)
    if len(parts) == 2 and name.endswith('.pre_mix'):
        return (boundary_order(name[:-len('.pre_mix')] + '.hidden')[0], suffix['pre_mix'])
    if name.startswith('encoder.entry'):
        return (0, 0 if name.endswith('hidden') else 1)
    if name == 'encoder.out.hidden':
        return (100, 0)
    if name == 'encoder.out.pre_mix':
        return (100, suffix['pre_mix'])
    if name == 'decoder.out.hidden':
        return (300, 0)
    if name == 'decoder.out.pre_mix':
        return (300, suffix['pre_mix'])
    if name == 'final.collapsed':
        return (400, 0)
    if name == 'final.norm':
        return (401, 0)
    if name == 'logits':
        return (402, 0)
    return (999, 0)


def as_float(raw, logical_dtype):
    """Decode MLX bfloat16 raw storage or plain floats to float32 for diffing."""
    if logical_dtype == 'bfloat16':
        u = raw.view(np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32)
    if logical_dtype in ('float32', 'float16'):
        return raw.astype(np.float32)
    if logical_dtype in ('uint8', 'uint16', 'uint32', 'int32'):
        return raw.astype(np.float32)
    raise ValueError(f'unsupported logical dtype {logical_dtype}')


def storage(raw, logical_dtype):
    """Canonical 1-D byte view for exact bit comparison."""
    return raw.tobytes()


def metrics(a_raw, a_dtype, b_raw, b_dtype):
    if a_raw.shape != b_raw.shape:
        return {'shape_mismatch': True, 'native_shape': list(a_raw.shape), 'oracle_shape': list(b_raw.shape)}
    out = {
        'shape': list(a_raw.shape),
        'native_dtype': a_dtype,
        'oracle_dtype': b_dtype,
        'dtype_mismatch': a_dtype != b_dtype,
        'exact_bits': a_dtype == b_dtype and storage(a_raw, a_dtype) == storage(b_raw, b_dtype),
    }
    # Element-bit differences, distinct from numerical differences such as +0/-0.
    out['bit_mismatch_count'] = None
    if a_dtype == b_dtype and a_raw.dtype.itemsize == b_raw.dtype.itemsize:
        width = a_raw.dtype.itemsize
        ab = np.frombuffer(a_raw.tobytes(), dtype=np.uint8).reshape(-1, width)
        bb = np.frombuffer(b_raw.tobytes(), dtype=np.uint8).reshape(-1, width)
        out['bit_mismatch_count'] = int(np.count_nonzero(np.any(ab != bb, axis=1)))
    af = as_float(a_raw, a_dtype).ravel()
    bf = as_float(b_raw, b_dtype).ravel()
    finite = np.isfinite(af) & np.isfinite(bf)
    diff = np.abs(af[finite] - bf[finite]) if finite.any() else np.array([0.0])
    out['max_abs_diff'] = float(diff.max())
    out['mean_abs_diff'] = float(diff.mean())
    out['nonfinite_native'] = int(np.count_nonzero(~np.isfinite(af)))
    out['nonfinite_oracle'] = int(np.count_nonzero(~np.isfinite(bf)))
    if af.size and finite.all():
        out['argmax_equal'] = bool(int(np.argmax(af)) == int(np.argmax(bf)))
    else:
        out['argmax_equal'] = None
    return out


def diverges(m, tolerance):
    return bool(m.get('shape_mismatch') or m.get('dtype_mismatch') or
                m.get('nonfinite_native') or m.get('nonfinite_oracle') or
                m.get('max_abs_diff', 0.0) > tolerance)


def compare_ties(native_ties, oracle_ties):
    """Match routing ties by (layer, token) and report whether the chosen sixth expert differs."""
    native_by = {(t.get('layer'), t.get('token')): t for t in native_ties}
    oracle_by = {(t.get('layer'), t.get('token')): t for t in oracle_ties}
    rows = []
    for key in sorted(set(native_by) | set(oracle_by)):
        n, o = native_by.get(key), oracle_by.get(key)
        rows.append({
            'layer': key[0],
            'token': key[1],
            'native_sixth_id': None if n is None else n.get('sixth_id'),
            'oracle_sixth_id': None if o is None else o.get('sixth_id'),
            'native_seventh_id': None if n is None else n.get('seventh_id'),
            'oracle_seventh_id': None if o is None else o.get('seventh_id'),
            'sixth_agrees': (n is not None and o is not None and n.get('sixth_id') == o.get('sixth_id')),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=0.0,
                   help='max_abs_diff above this marks a boundary as diverging')
    a = p.parse_args()
    if not np.isfinite(a.tolerance) or a.tolerance < 0:
        p.error('--tolerance must be finite and nonnegative')

    native = load_manifest(a.native)
    oracle = load_manifest(a.oracle)
    native_arrays, oracle_arrays = validate_pair(native, oracle)
    common = sorted(set(native_arrays) & set(oracle_arrays), key=boundary_order)

    boundaries = {}
    first_mismatch = None
    first_bit_mismatch = None
    for name in common:
        m = metrics(checked_array(a.native, native_arrays[name]), native_arrays[name]['dtype'],
                    checked_array(a.oracle, oracle_arrays[name]), oracle_arrays[name]['dtype'])
        m['native_array_sha256'] = sha(a.native / (name + '.npy'))
        m['oracle_array_sha256'] = sha(a.oracle / (name + '.npy'))
        boundaries[name] = m
        if not m.get('exact_bits', False) and first_bit_mismatch is None:
            first_bit_mismatch = name
        if diverges(m, a.tolerance) and first_mismatch is None:
            first_mismatch = name

    logits_summary = None
    if ('logits' in boundaries and not boundaries['logits'].get('shape_mismatch')
            and not boundaries['logits'].get('dtype_mismatch')
            and not boundaries['logits'].get('nonfinite_native')
            and not boundaries['logits'].get('nonfinite_oracle')):
        nl = as_float(load_array(a.native, 'logits'), native_arrays['logits']['dtype'])
        ol = as_float(load_array(a.oracle, 'logits'), oracle_arrays['logits']['dtype'])
        nl = nl.reshape(-1, nl.shape[-1]) if nl.ndim > 1 else nl.reshape(1, -1)
        ol = ol.reshape(-1, ol.shape[-1]) if ol.ndim > 1 else ol.reshape(1, -1)
        per_position = []
        argmax_equal = 0
        for t in range(nl.shape[0]):
            row_diff = np.abs(nl[t] - ol[t])
            agree = int(np.argmax(nl[t])) == int(np.argmax(ol[t]))
            argmax_equal += int(agree)
            per_position.append({'token': t, 'max_abs': float(row_diff.max()), 'mean_abs': float(row_diff.mean()), 'argmax_equal': agree})
        logits_summary = {
            'tokens': int(nl.shape[0]),
            'argmax_equal_tokens': argmax_equal,
            'max_abs_over_positions': float(max(p['max_abs'] for p in per_position)),
            'mean_abs_over_positions': float(np.mean([p['mean_abs'] for p in per_position])),
            'per_position': per_position,
        }

    report = {
        'schema_version': 1,
        'native': str(a.native),
        'oracle': str(a.oracle),
        'native_manifest_sha256': sha(a.native / 'manifest.json'),
        'oracle_manifest_sha256': sha(a.oracle / 'manifest.json'),
        'tolerance': a.tolerance,
        'common_boundaries': common,
        'native_only': sorted(set(native_arrays) - set(oracle_arrays)),
        'oracle_only': sorted(set(oracle_arrays) - set(native_arrays)),
        'first_diverging_boundary': first_mismatch,
        'first_bit_mismatching_boundary': first_bit_mismatch,
        'boundaries': boundaries,
        'native_route_tie_count': native.get('route_tie_count'),
        'oracle_route_tie_count': oracle.get('route_tie_count'),
        'route_tie_comparison': compare_ties(native.get('route_ties', []), oracle.get('route_ties', [])),
        'native_logits_argmax': native.get('logits_argmax'),
        'oracle_logits_argmax': oracle.get('logits_argmax'),
        'logits_summary': logits_summary,
        'scope': 'Boundary comparison of two traces; bit equality is not claimed as full-model qualification.',
    }
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'first_diverging_boundary': first_mismatch,
                      'boundaries': len(common),
                      'native_only': report['native_only'],
                      'oracle_only': report['oracle_only'],
                      'native_route_tie_count': report['native_route_tie_count'],
                      'oracle_route_tie_count': report['oracle_route_tie_count'],
                      'logits_summary': None if logits_summary is None else {
                          k: logits_summary[k] for k in ('tokens', 'argmax_equal_tokens', 'max_abs_over_positions', 'mean_abs_over_positions')}}, indent=2))
    for row in report['route_tie_comparison']:
        if not row['sixth_agrees']:
            print(f"  tie layer={row['layer']} token={row['token']}: native_sixth={row['native_sixth_id']} oracle_sixth={row['oracle_sixth_id']}")
    for name in common:
        m = boundaries[name]
        if m.get('shape_mismatch'):
            print(f'  {name}: SHAPE MISMATCH native={m["native_shape"]} oracle={m["oracle_shape"]}')
        elif not m['exact_bits']:
            print(f'  {name}: exact_bits=False max_abs={m["max_abs_diff"]:.6g} argmax={m.get("argmax_equal")}')


if __name__ == '__main__':
    main()
