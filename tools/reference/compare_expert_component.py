"""Compare one routed FP4 expert contribution against native trace."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from cpu_reference import Weights,Linear,sha
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--expert',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('fresh output required')
 m=json.loads((a.input/'manifest.json').read_text());nm=json.loads((a.native/'manifest.json').read_text());gm=json.loads((a.gate/'manifest.json').read_text());n=len(m['token_ids']); x=torch.from_numpy(np.load(a.input/'encoder.layer0.ffn_in.npy').view(np.uint16).copy()).view(torch.bfloat16);ids=np.array(gm['selected_experts']).reshape(n,6);rw=np.load(a.gate/'encoder.layer0.route_weights.npy').reshape(n,6); nat=torch.from_numpy(np.load(a.native/'encoder.layer0.expert_contribution.npy').view(np.uint16).copy()).view(torch.bfloat16)
 w=Weights(a.checkpoint,'artifacts/checkpoint/summary.json','artifacts/checkpoint/verification.json');
 class E:
  def __init__(s):s.w1=Linear(w,f'layers.0.ffn.experts.{a.expert}.w1');s.w2=Linear(w,f'layers.0.ffn.experts.{a.expert}.w2');s.w3=Linear(w,f'layers.0.ffn.experts.{a.expert}.w3')
  def __call__(s,z,q):
   g=s.w1(z).float().clamp(max=10);u=s.w3(z).float().clamp(-10,10);return s.w2((torch.sigmoid(g)*g*u).to(torch.bfloat16)).float()*q
 e=E(); rows=[];occ=[]
 with torch.inference_mode():
  for t in range(n):
   y=torch.zeros(1,5120,dtype=torch.float32)
   for k in range(6):
    if ids[t,k]==a.expert:y=y+e(x[t:t+1],float(rw[t,k]));occ.append(t)
   rows.append(y.to(torch.bfloat16))
 out=torch.cat(rows);u=out.view(torch.int16)!=nat.view(torch.int16);d=(out.float()-nat.float()).abs();r={'scope':'Single routed FP4 expert CPU contribution vs native; no full-model qualification','expert':a.expert,'occurrences':occ,'bit_mismatch_count':int(u.sum()),'max_abs_diff':float(d.max()),'mean_abs_diff':float(d.mean()),'nonfinite_cpu':int((~torch.isfinite(out)).sum()),'nonfinite_native':int((~torch.isfinite(nat)).sum()),'full_model_qualified':False,'native_manifest_sha256':sha(a.native/'manifest.json'),'gate_manifest_sha256':sha(a.gate/'manifest.json'),'source_sha256':w.sources,'tensor_sha256':w.tensor_hashes,'script_sha256':sha(Path(__file__))};a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
