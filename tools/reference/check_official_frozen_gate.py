"""Run the official Gate.forward on one native frozen FFN input using torch CPU.

The Gate class is AST-extracted unchanged from checkpoint inference/model.py.
This records one torch CPU topk observation; tie ordering remains unspecified by
the torch API and the result is not a CUDA oracle or M2 qualification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


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

    manifest_path = a.native_trace / 'manifest.json'
    native = json.loads(manifest_path.read_text())
    tie = next((r for r in native.get('route_ties', [])
                if r['layer'] == a.layer and r['token'] == a.token), None)
    if tie is None:
        raise ValueError('native trace has no tie at requested layer/token')
    input_path = a.native_trace / f'encoder.layer{a.layer}.ffn_in.npy'
    if not input_path.is_file():
        input_path = a.native_trace / f'decoder.layer{a.layer}.ffn_in.npy'
    raw_input = np.load(input_path, allow_pickle=False)
    if raw_input.ndim != 2 or not 0 <= a.token < raw_input.shape[0]:
        raise ValueError('unexpected frozen FFN input shape')
    x = torch.from_numpy(raw_input.view(np.uint16)[a.token:a.token + 1].copy()).view(torch.bfloat16)

    mapping_path = a.checkpoint / 'model.safetensors.index.json'
    mapping = json.loads(mapping_path.read_text())['weight_map']
    prefix = f'layers.{a.layer}.ffn.gate'
    keys = [prefix + '.weight', prefix + '.bias']
    if any(key not in mapping for key in keys):
        raise ValueError('gate tensor missing from checkpoint index')
    tensors = {}
    for filename in sorted({mapping[key] for key in keys}):
        with safe_open(a.checkpoint / filename, framework='pt', device='cpu') as reader:
            for key in keys:
                if mapping[key] == filename:
                    tensors[key] = reader.get_tensor(key)

    source = a.checkpoint / 'inference/model.py'
    tree = ast.parse(source.read_text())
    gate_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Gate')
    module = ast.Module(body=[gate_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'torch': torch, 'nn': torch.nn, 'F': F, 'linear': F.linear,
                 'ModelArgs': object}
    exec(compile(module, str(source), 'exec'), namespace)

    n_routed, dim = tensors[prefix + '.weight'].shape
    config_json = json.loads((a.checkpoint / 'config.json').read_text())
    text_config = config_json.get('text_config', config_json)
    topk = int(text_config['num_experts_per_tok'])
    args = SimpleNamespace(
        dim=int(dim), score_func=text_config['scoring_func'],
        gate_temp=float(text_config.get('gate_temp', 1.0)),
        norm_topk_prob=bool(text_config.get('norm_topk_prob', True)),
        route_scale=float(text_config['routed_scaling_factor']),
        vision_enabled=False,
        get_moe_config=lambda _layer: (int(n_routed), topk),
    )
    gate = namespace['Gate'](a.layer, args)
    gate.weight = torch.nn.Parameter(tensors[prefix + '.weight'], requires_grad=False)
    gate.bias = torch.nn.Parameter(tensors[prefix + '.bias'], requires_grad=False)
    with torch.inference_mode():
        _weights, indices = gate(x, None)
        raw = F.linear(x.float(), gate.weight.float()) / args.gate_temp
        scores = (raw.softmax(-1) if args.score_func == 'softmax' else
                  raw.sigmoid() if args.score_func == 'sigmoid' else F.softplus(raw).sqrt())
        corrected = (scores + gate.bias).reshape(-1).float()
    corrected_np = corrected.numpy()
    order = np.argsort(-corrected_np, kind='stable')
    selected = [int(v) for v in indices.reshape(-1).tolist()]
    sixth, seventh = int(order[5]), int(order[6])
    sixth_bits = int(corrected_np[sixth].view(np.uint32))
    seventh_bits = int(corrected_np[seventh].view(np.uint32))
    native_pair = [int(tie['sixth_id']), int(tie['seventh_id'])]

    result = {
        'schema_version': 1,
        'status': 'observation_recorded',
        'scope': ('Official checkpoint Gate.forward, AST-extracted unchanged, on native frozen '
                  'FFN input using torch CPU. torch topk tie ordering is unspecified; this is not '
                  'a CUDA oracle or M2 qualification.'),
        'layer': a.layer, 'token': a.token,
        'token_id': int(native['token_ids'][a.token]),
        'native_tie': tie,
        'official_torch_cpu_frozen_input': {
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
            'native_manifest_sha256': sha(manifest_path),
            'native_input_sha256': sha(input_path),
            'official_model_source_sha256': sha(source),
            'checkpoint_config_sha256': sha(a.checkpoint / 'config.json'),
            'checkpoint_index_sha256': sha(mapping_path),
            'gate_shards': {name: sha(a.checkpoint / name) for name in sorted({mapping[k] for k in keys})},
            'script_sha256': sha(Path(__file__)),
            'torch_version': torch.__version__,
            'torch_device': 'cpu',
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'exact_tie': sixth_bits == seventh_bits,
                      'boundary': [sixth, seventh], 'selected_ids': selected}))


if __name__ == '__main__':
    main()
