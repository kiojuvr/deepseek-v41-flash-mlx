"""Pinned oMLX repack/QMM reference, plus a separate activation-floor diagnostic."""
import argparse, ast, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.dont_write_bytecode=True
PIN='b390b31e0c6831225fed0f24d278eb1db7fcb68b'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pinned(root,name):
    data=subprocess.check_output(['git','-C',str(root),'show',PIN+':'+name])
    if (root/name).read_bytes()!=data: raise ValueError('modified oMLX source')
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--omlx',type=Path,required=True); p.add_argument('--fixtures',type=Path,required=True)
    a=p.parse_args(); root=a.fixtures; m=json.loads((root/'manifest.json').read_text())
    for name,h in m['fixture_sha256'].items():
        if Path(name).name!=name or sha(root/name)!=h: raise ValueError('fixture changed')
    convert='omlx/patches/deepseek_v41/convert.py'; activation='omlx/patches/deepseek_v41/activation.py'
    tree=ast.parse(pinned(a.omlx,convert)); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='repack_weight')
    ns={'mx':mx,'np':np}; exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.omlx/convert),'exec'),ns)
    pinned(a.omlx,activation)
    spec=importlib.util.spec_from_file_location('dsv41_pinned_activation',a.omlx/activation)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    mx.set_default_device(mx.gpu); results=[]; generated={}
    for c in m['cases']:
        tag=c['tag']; n,k,t=c['output_dims'],c['input_dims'],c['tokens']
        def raw(suffix,dtype,shape): return np.frombuffer((root/(tag+'-'+suffix)).read_bytes(),dtype=dtype).copy().reshape(shape)
        wr=raw('weight.raw',np.uint8,c['storage_shape']); sr=raw('scale.raw',np.uint8,c['scale_shape'])
        weights,config=ns['repack_weight'](wr,c['storage_dtype'],sr,'F8_E8M0')
        mx.eval(weights)
        for field,expected in [('weight',c['packed_weight_sha256']),('scales',c['expanded_scale_sha256'])]:
            if hashlib.sha256(np.array(weights[field]).tobytes()).hexdigest()!=expected: raise ValueError('repack mismatch')
        decoded=mx.array(raw('decoded.bf16',np.uint16,(t,k))).view(mx.bfloat16)
        x=mx.array(raw('input.bf16',np.uint16,(t,k))).view(mx.bfloat16)
        original_activation=module.quantize_fp8_activation(x)
        output=mx.quantized_matmul(decoded,weights['weight'],weights['scales'],group_size=32,**config)
        original_output=mx.quantized_matmul(original_activation,weights['weight'],weights['scales'],group_size=32,**config)
        mx.eval(output,original_activation,original_output)
        expected=np.array(output.view(mx.uint16)); old=np.array(original_activation.view(mx.uint16))
        ref=np.array(decoded.view(mx.uint16)); old_output=np.array(original_output.view(mx.uint16))
        path=root/(tag+'-omlx-qmm.bf16'); path.write_bytes(expected.tobytes()); generated[path.name]=sha(path)
        cpu=raw('cpu-projection.bf16',np.uint16,(t,n))
        results.append({'prefix':c['prefix'],'repack_exact':True,
            'qmm_vs_cpu_block_mismatches':int(np.count_nonzero(expected!=cpu)),
            'omlx_vs_official_activation_mismatches_by_token':np.count_nonzero(old!=ref,axis=1).tolist(),
            'omlx_vs_common_input_qmm_mismatches_by_token':np.count_nonzero(old_output!=expected,axis=1).tolist()})
    report={'status':'diagnostic_complete','omlx_commit':PIN,'mlx_version':mx.__version__,
        'source_sha256':{s:sha(a.omlx/s) for s in [convert,activation]},'script_sha256':sha(__file__),
        'manifest_sha256':sha(root/'manifest.json'),'generated_sha256':generated,'runs':results,
        'scope':'Pinned oMLX repack; MLX QMM with identical official-quantized input. Separate original oMLX activation diagnostic. No full model/CUDA execution.'}
    (root/'omlx-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
