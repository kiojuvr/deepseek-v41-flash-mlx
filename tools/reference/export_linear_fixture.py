"""Small real-checkpoint linear fixtures; CPU transcription is not a CUDA oracle."""
import argparse, hashlib, json, struct
from pathlib import Path
import torch
import torch.nn.functional as F


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok: raise ValueError(msg)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','m1-summary','verification','entry-fixtures','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); root=a.checkpoint.resolve(); out=a.output.resolve()
    require(not out.is_relative_to(root),'checkpoint read-only')
    summary=json.loads(a.m1_summary.read_text()); verified=json.loads(a.verification.read_text())
    require(summary['status']==verified['status']=='verified' and summary['revision']==verified['revision'],'M1 mismatch')
    sources=['inference/model.py','inference/kernel.py']
    for name in sources:
        pin=next(r for r in verified['files'] if r['path']==name)
        require(sha(root/name)==pin['sha256'],'source changed')
    entry=json.loads((a.entry_fixtures/'manifest.json').read_text())
    require(entry['revision']==summary['revision'],'entry revision mismatch')
    require(sha(a.entry_fixtures/'norm-input.bf16')==entry['fixture_sha256']['norm-input.bf16'],'input fixture changed')
    raw_input=torch.frombuffer(bytearray((a.entry_fixtures/'norm-input.bf16').read_bytes()),dtype=torch.bfloat16).reshape(10,5120)
    mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    def tensor(name):
        shard=mapping[name]; require(Path(shard).name==shard,'unsafe shard')
        with (root/shard).open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]; require(n<64*1024*1024,'header too large')
            data=f.read(n); require(hashlib.sha256(data).hexdigest()==summary['shards'][shard]['header_sha256'],'header changed')
            h=json.loads(data)[name]; f.seek(8+n+h['data_offsets'][0]); raw=f.read(h['data_offsets'][1]-h['data_offsets'][0])
            require(len(raw)==h['data_offsets'][1]-h['data_offsets'][0],'short tensor')
        return raw,h
    out.mkdir(parents=True,exist_ok=True); files=[]; cases=[]; torch.set_num_threads(4)
    def save(name,data):
        if isinstance(data,torch.Tensor): data=data.contiguous().view(torch.uint8).numpy().tobytes()
        (out/name).write_bytes(data); files.append(name)
    for i,prefix in enumerate(['layers.0.attn.wq_a','layers.0.ffn.experts.0.w1','layers.0.ffn.experts.0.w2']):
        wr,wh=tensor(prefix+'.weight'); sr,sh=tensor(prefix+'.scale')
        bits=8 if wh['dtype']=='F8_E4M3' else 4
        require(wh['dtype'] in ['F8_E4M3','I8'] and sh['dtype']=='F8_E8M0','unsupported dtype')
        n,stored=wh['shape']; k=stored*(8//bits); tag='case-'+str(i)
        x=raw_input[:,:k].contiguous(); blocks=x.float().reshape(10,k//32,32)
        maxima=blocks.abs().amax(-1).clamp_min(1e-4)
        b=(maxima*(1.0/448.0)).view(torch.int32)
        exponent=((b>>23)&255)-127+((b&0x7fffff)!=0).int()
        scale=torch.ldexp(torch.ones_like(maxima),exponent)
        quant=(blocks/scale.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
        decoded=(quant.float()*scale.unsqueeze(-1)).reshape(10,k).to(torch.bfloat16)
        w=torch.frombuffer(bytearray(wr),dtype=torch.uint8).reshape(n,stored)
        if bits==8: wf=w.view(torch.float8_e4m3fn).float()
        else:
            lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
            codes=torch.stack([w&15,w>>4],-1).reshape(n,k).long(); wf=lut[codes]
        scales=torch.frombuffer(bytearray(sr),dtype=torch.float8_e8m0fnu).reshape(sh['shape'])
        expanded=scales.view(torch.uint8).repeat_interleave(32,0)[:n] if bits==8 else scales.view(torch.uint8)
        sf=expanded.view(torch.float8_e8m0fnu).float()
        accum=torch.zeros(10,n)
        for block in range(k//32):
            partial=F.linear(quant[:,block,:].float(),wf[:,block*32:(block+1)*32])
            accum=accum+(partial*scale[:,block:block+1])*sf[:,block]
        expected=accum.to(torch.bfloat16)
        require(torch.isfinite(expected).all().item(),'nonfinite expected projection')
        for suffix,data in [('weight.raw',wr),('scale.raw',sr),('input.bf16',x),('quant.u8',quant.view(torch.uint8)),
                            ('activation-scale.u8',scale.to(torch.float8_e8m0fnu).view(torch.uint8)),
                            ('decoded.bf16',decoded),('cpu-projection.bf16',expected)]: save(tag+'-'+suffix,data)
        cases.append({'tag':tag,'prefix':prefix,'bits':bits,'tokens':10,'input_dims':k,'output_dims':n,
                      'storage_shape':wh['shape'],'scale_shape':sh['shape'],'storage_dtype':wh['dtype'],
                      'packed_weight_sha256':hashlib.sha256(wr).hexdigest(),
                      'expanded_scale_sha256':hashlib.sha256(expanded.contiguous().numpy().tobytes()).hexdigest()})
    manifest={'schema_version':1,'revision':summary['revision'],'source_sha256':{name:sha(root/name) for name in sources},
              'exporter_sha256':sha(__file__),'torch_version':torch.__version__,
              'entry_manifest_sha256':sha(a.entry_fixtures/'manifest.json'),
              'scope':'Real FP8/FP4 weights; official-formula CPU activation/block-GEMM transcription. CUDA kernels not executed.',
              'cases':cases,'fixture_sha256':{name:sha(out/name) for name in files}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Exported three matrices and 10-token inputs (embedding, zero, tiny, large).')

if __name__=='__main__': main()
