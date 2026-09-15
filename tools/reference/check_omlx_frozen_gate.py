"""Replay one pinned oMLX Gate on a native frozen FFN input.

Only the gate weight and bias tensors are read from the checkpoint.  This
separates routing selection semantics from earlier full-model numerical drift;
it is a diagnostic, not an official torch.topk oracle or M2 qualification.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native-trace', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--token', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.resolve().is_relative_to(a.checkpoint.resolve()):
        raise ValueError('checkpoint is read-only')

    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.convert import mapped
    from omlx.patches.deepseek_v41.language import Gate
    from omlx.patches.deepseek_v41.storage import TensorFile, decode_array
    import omlx.patches.deepseek_v41.language as language

    native_manifest_path = a.native_trace / 'manifest.json'
    native = json.loads(native_manifest_path.read_text())
    if not 0 <= a.token < len(native['token_ids']):
        raise ValueError('token position outside native trace')
    tie = next((r for r in native.get('route_ties', [])
                if r['layer'] == a.layer and r['token'] == a.token), None)
    if tie is None:
        raise ValueError('native trace has no tie at requested layer/token')

    input_path = a.native_trace / f'encoder.layer{a.layer}.ffn_in.npy'
    if not input_path.is_file():
        input_path = a.native_trace / f'decoder.layer{a.layer}.ffn_in.npy'
    x_all = mx.load(str(input_path))
    if x_all.ndim != 2 or a.token >= x_all.shape[0]:
        raise ValueError('unexpected frozen FFN input shape')
    x = x_all[a.token:a.token + 1]

    raw_config = json.loads((a.checkpoint / 'config.json').read_text())
    config = ModelConfig.from_dict(raw_config)
    mapping_path = a.checkpoint / 'model.safetensors.index.json'
    mapping = json.loads(mapping_path.read_text())['weight_map']
    prefix = f'layers.{a.layer}.ffn.gate'
    keys = [prefix + '.weight', prefix + '.bias']
    if any(key not in mapping for key in keys):
        raise ValueError('gate tensor missing from checkpoint index')

    readers = {}
    try:
        values = {}
        for key in keys:
            filename = mapping[key]
            reader = readers.setdefault(filename, TensorFile(a.checkpoint / filename))
            raw, dtype = reader.read(key)
            values[key] = decode_array(raw, dtype)
    finally:
        for reader in readers.values():
            reader.close()

    gate = Gate(config)
    gate.weight = values[prefix + '.weight']
    gate.bias = values[prefix + '.bias']
    idx, weights = gate(x, None)
    raw = (x.astype(mx.float32) @ gate.weight.astype(mx.float32).T) / config.gate_temp
    scores = (
        mx.softmax(raw, -1) if config.score_func == 'softmax'
        else mx.sigmoid(raw) if config.score_func == 'sigmoid'
        else mx.sqrt(mx.logaddexp(raw, 0))
    )
    corrected = (scores + gate.bias).reshape(-1).astype(mx.float32)
    mx.eval(idx, weights, corrected)
    corrected_np = np.array(corrected)
    order = np.argsort(-corrected_np, kind='stable')
    selected = [int(v) for v in np.array(idx).reshape(-1)]
    sixth, seventh = int(order[5]), int(order[6])
    sixth_bits = int(corrected_np[sixth].view(np.uint32))
    seventh_bits = int(corrected_np[seventh].view(np.uint32))
    native_pair = [int(tie['sixth_id']), int(tie['seventh_id'])]

    language_path = Path(inspect.getfile(language))
    result = {
        'schema_version': 1,
        'status': 'observation_recorded',
        'scope': ('Pinned oMLX Gate expression on the native frozen FFN input; only gate tensors '
                  'were read. This is not an official torch.topk oracle or M2 qualification.'),
        'layer': a.layer,
        'token': a.token,
        'token_id': int(native['token_ids'][a.token]),
        'native_tie': tie,
        'omlx_frozen_input': {
            'selected_ids': selected,
            'sixth_id_stable_score_order': sixth,
            'seventh_id_stable_score_order': seventh,
            'sixth_score': float(corrected_np[sixth]),
            'seventh_score': float(corrected_np[seventh]),
            'sixth_score_bits': sixth_bits,
            'seventh_score_bits': seventh_bits,
            'boundary_is_exact_tie': sixth_bits == seventh_bits,
            'native_pair_is_boundary': [sixth, seventh] == native_pair,
            'selected_native_sixth': native_pair[0] in selected,
            'selected_native_seventh': native_pair[1] in selected,
            'top8_stable_score_order': [int(v) for v in order[:8]],
        },
        'identity': {
            'native_manifest_sha256': sha(native_manifest_path),
            'native_input_sha256': sha(input_path),
            'checkpoint_config_sha256': sha(a.checkpoint / 'config.json'),
            'checkpoint_index_sha256': sha(mapping_path),
            'gate_shards': {name: sha(a.checkpoint / name) for name in sorted(readers)},
            'omlx_language_path': str(language_path),
            'omlx_language_sha256': sha(language_path),
            'script_sha256': sha(Path(__file__)),
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': result['status'],
        'exact_tie': result['omlx_frozen_input']['boundary_is_exact_tie'],
        'boundary': [sixth, seventh],
        'selected_ids': selected,
    }))


if __name__ == '__main__':
    main()
