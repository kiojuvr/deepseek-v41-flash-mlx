"""Offline fixed-omlx Engram gate replay with frozen projection; never a runtime dependency."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import mlx.core as mx
import numpy as np

PIN = 'b390b31e0c6831225fed0f24d278eb1db7fcb68b'
SOURCE = 'omlx/patches/deepseek_v41/engram.py'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(a, e, dtype):
    a, e = np.asarray(a, dtype=dtype).reshape(-1), np.asarray(e, dtype=dtype).reshape(-1)
    if a.shape != e.shape:
        raise ValueError('shape mismatch')
    if dtype == np.float32:
        if not np.isfinite(a).all() or not np.isfinite(e).all():
            raise ValueError('nonfinite result')
        ab, eb = a.view(np.uint32), e.view(np.uint32)
    else:
        ab, eb = a, e
    mismatch = np.flatnonzero(ab != eb)
    first = int(mismatch[0]) if mismatch.size else None
    return {'elements':int(a.size),'bit_mismatches':int(mismatch.size),
            'first_mismatch':None if first is None else {'index':first,'actual_bits':int(ab[first]),'expected_bits':int(eb[first])},
            'max_abs':float(np.max(np.abs(a.astype(np.float64)-e.astype(np.float64)))) if dtype == np.float32 else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--omlx',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--native-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    source = args.omlx/SOURCE
    pinned = subprocess.check_output(['git','-C',str(args.omlx),'show',PIN+':'+SOURCE])
    if source.read_bytes() != pinned:
        raise ValueError('omlx source differs from pinned commit')
    m = json.loads((args.fixtures/'manifest.json').read_text())
    native = json.loads(args.native_result.read_text())
    if native['manifest_sha256'] != sha(args.fixtures/'manifest.json'):
        raise ValueError('native fixture mismatch')
    for name,digest in m['fixture_sha256'].items():
        if Path(name).name != name or sha(args.fixtures/name) != digest:
            raise ValueError('fixture mismatch')
    tree = ast.parse(pinned)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engram')
    fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    if not isinstance(fn.body[-1],ast.Return):
        raise ValueError('omlx return structure changed')
    fn.body[-1].value = ast.Tuple(elts=[fn.body[-1].value,ast.Name(id='dot',ctx=ast.Load()),
                                      ast.Name(id='gate',ctx=ast.Load())],ctx=ast.Load())
    ast.fix_missing_locations(fn)
    ns = {'mx':mx}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    mx.set_default_device(mx.gpu)
    runs=[]
    for c in m['cases']:
        n,tag=c['tokens'],c['tag']
        def raw(suffix,dtype):
            return np.frombuffer((args.fixtures/(tag+'-'+suffix)).read_bytes(),dtype=dtype).copy()
        def bf16(suffix,shape):
            return mx.array(raw(suffix,np.uint16).reshape(shape)).view(mx.bfloat16)
        h=bf16('hidden.bf16',(n,4,5120)); kv=bf16('kv.bf16',(n,25600))
        frozen=SimpleNamespace(_config=SimpleNamespace(dim=5120,hc_mult=4,norm_eps=m['norm_eps']),
            q_weight=bf16('q.bf16',(4,5120)),k_weight=bf16('k.bf16',(4,5120)),
            embed=lambda _:mx.zeros((n,24,256)),wkv=lambda _:kv)
        output,dot,gate=ns['__call__'](frozen,h,np.zeros((n,24),dtype=np.int64),
                                     mx.array([not v for v in c['mask']]))
        mx.eval(output,dot,gate)
        dot_np=np.array(dot); gate_np=np.array(gate)
        residual=np.array(output.view(mx.uint16))
        native_run=next(r for r in native['runs'] if r['layer']==c['layer_id'] and r['mode']=='mmap')
        runs.append({'layer':c['layer_id'],
            'official_dot':compare(dot_np,raw('dot.f32',np.float32),np.float32),
            'official_gate':compare(gate_np,raw('gate.f32',np.float32),np.float32),
            'official_residual':compare(residual,raw('expected.bf16',np.uint16),np.uint16),
            'native_dot':compare(dot_np,native_run['official_dot_fp32']['actual'],np.float32),
            'native_gate':compare(gate_np,native_run['official_gate_fp32']['actual'],np.float32),
            'dot':dot_np.reshape(-1).tolist(),'gate':gate_np.reshape(-1).tolist()})
    result={'status':'diagnostic_complete','scope':'AST-extracted omlx Engram.__call__, frozen CPU-transcribed projection, synthetic hidden. No model load or full-path qualification.',
            'omlx_commit':PIN,'omlx_source_sha256':sha(source),'mlx_version':mx.__version__,
            'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(args.fixtures/'manifest.json'),
            'native_result_sha256':sha(args.native_result),'runs':runs,'full_model_qualified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in runs:
        print(r['layer'],{k:v for k,v in r.items() if isinstance(v,dict)})


if __name__=='__main__':
    main()
