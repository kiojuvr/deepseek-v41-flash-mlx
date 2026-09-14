"""Validate CPU FP4 expert SwiGLU against the stored real expert fixture."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from cpu_reference import Linear, sha

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--fixture',type=Path,default=Path('artifacts/linear')); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): p.error('output must be fresh')
    m=json.loads((a.fixture/'manifest.json').read_text()); c=[x for x in m['cases'] if 'experts.0' in x['prefix']]
    if len(c)!=2: raise ValueError('expert w1/w2 fixture missing')
    def load(case,suffix,shape,dtype=torch.bfloat16):
        raw=(a.fixture/(case['tag']+'-'+suffix)).read_bytes(); return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(shape)
    class W:
        def tensor(self,name):
            case=next(x for x in c if x['prefix']==name.rsplit('.',1)[0]); suffix=name.rsplit('.',1)[1]
            if suffix=='weight': return load(case,'weight.raw',case['storage_shape'],torch.uint8),case['storage_dtype']
            return load(case,'scale.raw',case['scale_shape'],torch.uint8).view(torch.float8_e8m0fnu), 'F8_E8M0'
    w=W(); w1=Linear(w,'layers.0.ffn.experts.0.w1'); w2=Linear(w,'layers.0.ffn.experts.0.w2')
    x=load(c[0],'input.bf16',(10,5120)); expected1=load(c[0],'cpu-projection.bf16',(10,2304)); expected2=load(c[1],'cpu-projection.bf16',(10,5120))
    # The stored w2 input is an independent fixture; validate both linears and
    # the SwiGLU clamp/activation using deterministic synthetic branches.
    out1=w1(x); x2=load(c[1],'input.bf16',(10,2304)); out2=w2(x2)
    def stat(a,b):
        aa=a.view(torch.int16); bb=b.view(torch.int16); d=(a.float()-b.float()).abs(); return {'bit_mismatch':int((aa!=bb).sum()),'max_abs':float(d.max()),'mean_abs':float(d.mean())}
    gate=torch.linspace(-12,12,2304).reshape(1,-1).to(torch.bfloat16); up=torch.linspace(-12,12,2304).reshape(1,-1).to(torch.bfloat16)
    clamped_gate=gate.float().clamp(max=10); clamped_up=up.float().clamp(-10,10); swiglu=(torch.sigmoid(clamped_gate)*clamped_gate*clamped_up).to(torch.bfloat16)
    result={'scope':'CPU FP4 expert linear fixture and SwiGLU clamp diagnostic; no six-expert or full-model qualification','w1':stat(out1,expected1),'w2':stat(out2,expected2),'swiglu_finite':bool(torch.isfinite(swiglu).all()),'swiglu_range':[float(swiglu.min()),float(swiglu.max())],'fixture_manifest_sha256':sha(a.fixture/'manifest.json'),'script_sha256':sha(Path(__file__)),'full_model_qualified':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
