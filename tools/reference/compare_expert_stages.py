"""Compare CPU FP4 expert stages with a native expert stage trace."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from cpu_reference import Weights,Linear,sha
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--expert',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('fresh output required')
 m=json.loads((a.input/'manifest.json').read_text());gm=json.loads((a.gate/'manifest.json').read_text());n=len(m['token_ids']);x=torch.from_numpy(np.load(a.input/'encoder.layer0.ffn_in.npy').view(np.uint16).copy()).view(torch.bfloat16);ids=np.array(gm['selected_experts']).reshape(n,6);rw=np.load(a.gate/'encoder.layer0.route_weights.npy').reshape(n,6)
 def read(name):return torch.from_numpy(np.load(a.native/f'encoder.layer0.expert_{name}.npy').view(np.uint16).copy()).view(torch.bfloat16)
 nat={s:read(s) for s in ('gate','up','activation','contribution')};w=Weights(a.checkpoint,'artifacts/checkpoint/summary.json','artifacts/checkpoint/verification.json');w1=Linear(w,f'layers.0.ffn.experts.{a.expert}.w1');w2=Linear(w,f'layers.0.ffn.experts.{a.expert}.w2');w3=Linear(w,f'layers.0.ffn.experts.{a.expert}.w3');cpu={s:[] for s in nat};occ=[]
 with torch.inference_mode():
  for t in range(n):
   vals={s:torch.zeros_like(nat[s][t:t+1]) for s in nat}
   for k in range(6):
    if ids[t,k]==a.expert:
     g=w1(x[t:t+1]);u=w3(x[t:t+1]);gf=g.float().clamp(max=10);uf=u.float().clamp(-10,10);act=(torch.sigmoid(gf)*gf*uf).to(torch.bfloat16);out=(w2(act).float()*float(rw[t,k])).to(torch.bfloat16);vals.update(gate=g,up=u,activation=act,contribution=out);occ.append(t)
   for s in cpu:cpu[s].append(vals[s])
 cpu={s:torch.cat(v) for s,v in cpu.items()}
 def stat(a,b):d=(a.float()-b.float()).abs();return {'bit_mismatch_count':int((a.view(torch.int16)!=b.view(torch.int16)).sum()),'max_abs_diff':float(d.max()),'mean_abs_diff':float(d.mean())}
 r={'scope':'CPU versus native routed expert internal stages; no qualification','expert':a.expert,'occurrences':occ,'stages':{s:stat(cpu[s],nat[s]) for s in nat},'full_model_qualified':False,'script_sha256':sha(Path(__file__))};a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stages'],indent=2))
if __name__=='__main__':main()
