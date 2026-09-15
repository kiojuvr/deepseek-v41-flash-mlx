"""Offline CPU oracle diagnostics for the native gate trace; no model load or promotion."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    a = torch.as_tensor(actual, dtype=torch.float32).flatten()
    e = torch.as_tensor(expected, dtype=torch.float32).flatten()
    if a.shape != e.shape or not torch.isfinite(a).all() or not torch.isfinite(e).all():
        raise ValueError('invalid comparison')
    ab, eb = a.view(torch.int32).to(torch.int64), e.view(torch.int32).to(torch.int64)
    def ordered(b):
        u = b & 0xffffffff
        return torch.where(u & 0x80000000 != 0, 0x80000000 - (u & 0x7fffffff), 0x80000000 + u)
    return {'bit_mismatches': int((ab != eb).sum()),
            'max_abs': float((a-e).abs().max()),
            'max_fp32_ulp': int((ordered(ab)-ordered(eb)).abs().max()),
            'actual': a.tolist(), 'expected': e.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--native-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError('checkpoint is read-only')
    manifest_path = args.fixtures/'manifest.json'
    m = json.loads(manifest_path.read_text())
    native = json.loads(args.native_result.read_text())
    if native['manifest_sha256'] != sha(manifest_path) or native['revision'] != m['revision']:
        raise ValueError('native/fixture identity mismatch')
    for name, digest in m['fixture_sha256'].items():
        if Path(name).name != name or sha(args.fixtures/name) != digest:
            raise ValueError('fixture identity mismatch')
    source = args.checkpoint/'inference/model.py'
    if sha(source) != m['source_sha256']['inference/model.py']:
        raise ValueError('official source changed')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engram')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    if not isinstance(forward.body[-1], ast.Return):
        raise ValueError('official forward structure changed')
    forward.body[-1].value = ast.Tuple(elts=[forward.body[-1].value, ast.Call(
        func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], ctx=ast.Load())
    ast.fix_missing_locations(cls)
    ns = {'torch': torch, 'nn': torch.nn, 'ModelArgs': object, 'EngramLayout': object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), ns)
    torch.set_num_threads(4)
    reports = []
    with torch.inference_mode():
        for case in m['cases']:
            tag, n = case['tag'], case['tokens']
            def read(suffix, dtype, shape):
                return torch.frombuffer(bytearray((args.fixtures/(tag+'-'+suffix)).read_bytes()),
                                        dtype=dtype).reshape(shape)
            h = read('hidden.bf16', torch.bfloat16, (n,4,5120))
            kv = read('kv.bf16', torch.bfloat16, (n,25600))
            q = read('q.bf16', torch.bfloat16, (4,5120))
            k = read('k.bf16', torch.bfloat16, (4,5120))
            dot_fixture = read('dot.f32', torch.float32, (n,4))
            gate_fixture = read('gate.f32', torch.float32, (n,4))
            frozen = SimpleNamespace(dim=5120, hc_mult=4, eps=m['norm_eps'], clamp_value=1e-6,
                                     q_weight=q, k_weight=k, embed=lambda _:torch.zeros(n,24,256),
                                     wkv=lambda _:kv)
            output, loc = ns['Engram'].forward(frozen, h, None, torch.tensor(case['mask']))
            for actual, expected in [(loc['dot'],dot_fixture),(loc['gate'],gate_fixture)]:
                if compare(actual,expected)['bit_mismatches']:
                    raise ValueError('official replay differs from frozen fixture')
            if not torch.equal(output.view(torch.int16),read('expected.bf16',torch.bfloat16,(n,4,5120)).view(torch.int16)):
                raise ValueError('official residual replay differs')
            hm, km = loc['h'].square().mean(-1), loc['key'].square().mean(-1)
            hr, kr = torch.rsqrt(hm+m['norm_eps']), torch.rsqrt(km+m['norm_eps'])
            weighted = loc['h']*loc['weight']*loc['key']
            root = torch.copysign(loc['dot'].abs().clamp_min(1e-6).sqrt(),loc['dot'])
            stages = {'hidden_mean_square':hm, 'key_mean_square':km,
                      'hidden_rsqrt':hr, 'key_rsqrt':kr, 'rstd':loc['rstd'],
                      'weighted_sum':weighted.sum(-1), 'dot':loc['dot'], 'signed_root':root,
                      'gate_unmasked':torch.sigmoid(root), 'oracle_dot_signed_root':root,
                      'oracle_dot_gate':torch.sigmoid(root)}
            for run in native['runs']:
                if run['layer'] != case['layer_id']:
                    continue
                trace = run['gate_trace']
                comparisons = {name:compare(trace[name],expected) for name,expected in stages.items()}
                # Inject native inputs to separate local reduction errors from rsqrt/sigmoid errors.
                for prefix in ('hidden','key'):
                    native_mean=torch.tensor(trace[prefix+'_mean_square']).reshape(n,4)
                    comparisons[prefix+'_rsqrt_given_native_mean']=compare(
                        trace[prefix+'_rsqrt'],torch.rsqrt(native_mean+m['norm_eps']))
                native_root=torch.tensor(trace['signed_root']).reshape(n,4)
                comparisons['sigmoid_given_native_root']=compare(trace['gate_unmasked'],torch.sigmoid(native_root))
                reports.append({'layer':run['layer'],'mode':run['mode'],'stages':comparisons})
    result = {'status':'diagnostic_complete','full_model_qualified':False,'optimized_path_promoted':False,
              'scope':'Official Engram.forward CPU replay with frozen projection; intermediate CPU reductions and boundary injections. CUDA not executed. omlx not executed.',
              'manifest_sha256':sha(manifest_path),'native_result_sha256':sha(args.native_result),
              'script_sha256':sha(Path(__file__)),'torch_version':torch.__version__,'torch_threads':4,
              'runs':reports}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in reports:
        print(r['layer'],r['mode'],{k:v['max_fp32_ulp'] for k,v in r['stages'].items()})


if __name__ == '__main__':
    main()
