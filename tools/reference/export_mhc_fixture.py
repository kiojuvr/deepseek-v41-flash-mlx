"""CPU formula transcription for mHC; no CUDA Sinkhorn oracle execution."""
import argparse,hashlib,json,struct
from pathlib import Path
import torch
import torch.nn.functional as F

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for key in ['checkpoint','summary','verification','entry','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();root=a.checkpoint.resolve();out=a.output.resolve()
 if out.is_relative_to(root):raise ValueError('checkpoint read-only')
 s=json.loads(a.summary.read_text());v=json.loads(a.verification.read_text())
 if s['status']!='verified' or v['status']!='verified' or s['revision']!=v['revision']:raise ValueError('M1 mismatch')
 sources=['inference/model.py','inference/kernel.py','inference/config.json']
 for name in sources:
  if sha(root/name)!=next(x['sha256'] for x in v['files'] if x['path']==name):raise ValueError('source changed')
 config=json.loads((root/'inference/config.json').read_text())
 if (config['hc_mult'],config['dim'],config['hc_sinkhorn_iters'],config['hc_eps'],config['norm_eps'])!=(4,5120,20,1e-6,1e-20):raise ValueError('config mismatch')
 entry=json.loads((a.entry/'manifest.json').read_text())
 if sha(a.entry/'embedding.bf16')!=entry['fixture_sha256']['embedding.bf16']:raise ValueError('embedding changed')
 embedding=torch.frombuffer(bytearray((a.entry/'embedding.bf16').read_bytes()),dtype=torch.bfloat16).reshape(7,5120)[:2]
 h=embedding[:,None,:].repeat(1,4,1)
 for i in range(4):h[:,i,:]=(h[:,i,:].float()*(i+1)/4).to(torch.bfloat16)
 index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
 def tensor(name):
  shard=index[name]
  if Path(shard).name!=shard:raise ValueError('unsafe shard')
  with (root/shard).open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0]
   if n>64*1024*1024:raise ValueError('header too large')
   raw=f.read(n)
   if hashlib.sha256(raw).hexdigest()!=s['shards'][shard]['header_sha256']:raise ValueError('header changed')
   t=json.loads(raw)[name];f.seek(8+n+t['data_offsets'][0]);data=f.read(t['data_offsets'][1]-t['data_offsets'][0])
  if t['dtype']!='F32':raise ValueError('dtype mismatch')
  return torch.frombuffer(bytearray(data),dtype=torch.float32).reshape(t['shape'])
 out.mkdir(parents=True,exist_ok=True);files=[]
 def save(name,x):
  (out/name).write_bytes(x.contiguous().view(torch.uint8).numpy().tobytes());files.append(name)
 save('hidden.bf16',h);save('sublayer.bf16',embedding)
 torch.set_num_threads(4)
 for kind in ['attn','ffn']:
  prefix='layers.0.hc_'+kind
  fn,scale,base=[tensor(prefix+suffix) for suffix in ['_fn','_scale','_base']]
  flat=h.flatten(1).float();mix=F.linear(flat,fn)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-20)
  pre=torch.sigmoid(mix[:,:4]*scale[0]+base[:4])+1e-6
  post=2*torch.sigmoid(mix[:,4:8]*scale[1]+base[4:8])
  comb=(mix[:,8:]*scale[2]+base[8:]).reshape(2,4,4)
  comb=comb.softmax(-1)+1e-6;comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
  for _ in range(19):
   comb=comb/(comb.sum(-1,keepdim=True)+1e-6);comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
  collapsed=(pre[:,:,None]*h.float()).sum(1).to(torch.bfloat16)
  expanded=(post[:,:,None]*embedding[:,None,:]+(comb[:,:,:,None]*h[:,:,None,:]).sum(1)).to(torch.bfloat16)
  for name,x in [('pre.f32',pre),('post.f32',post),('comb.f32',comb),('collapsed.bf16',collapsed),('expanded.bf16',expanded)]:save(kind+'-'+name,x)
 m={'revision':s['revision'],'tokens':2,'layer':0,'source_sha256':{n:sha(root/n) for n in sources},'script_sha256':sha(__file__),'torch_version':torch.__version__,'fixture_sha256':{n:sha(out/n) for n in files},'scope':'CPU transcription of official mHC equations with real layer-0 weights; no CUDA execution.'}
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print('Exported two-token mHC attention/FFN fixtures.')
if __name__=='__main__':main()
