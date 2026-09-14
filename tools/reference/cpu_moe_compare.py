"""Independent CPU FP4 layer-0 MoE comparison from frozen native routing."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from cpu_reference import Weights,Linear,sha

class Expert:
 def __init__(self,w,prefix): self.w1=Linear(w,prefix+'.w1'); self.w2=Linear(w,prefix+'.w2'); self.w3=Linear(w,prefix+'.w3')
 def __call__(self,x,weight=1.):
  g=self.w1(x).float().clamp(max=10); u=self.w3(x).float().clamp(-10,10)
  y=(torch.sigmoid(g)*g*u).to(torch.bfloat16)
  return self.w2(y).float()*weight

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--native',type=Path,required=True); p.add_argument('--native-output',type=Path); p.add_argument('--gate',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 if a.output.exists() or a.output.resolve().is_relative_to(a.checkpoint.resolve()): p.error('fresh output outside checkpoint required')
 m=json.loads((a.native/'manifest.json').read_text()); gm=json.loads((a.gate/'manifest.json').read_text()); n=len(m['token_ids'])
 if gm['token_ids']!=m['token_ids']: raise ValueError('token IDs mismatch')
 raw=np.load(a.native/'encoder.layer0.ffn_in.npy',allow_pickle=False); x=torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16)
 ids=torch.tensor(gm['selected_experts'],dtype=torch.int64).reshape(n,6); rw=torch.from_numpy(np.load(a.gate/'encoder.layer0.route_weights.npy',allow_pickle=False).copy()).reshape(n,6)
 native_root=a.native_output or a.native; native=np.load(native_root/'encoder.layer0.moe_out.npy',allow_pickle=False)
 weights=Weights(a.checkpoint,'artifacts/checkpoint/summary.json','artifacts/checkpoint/verification.json'); cache={}; shared=Expert(weights,'layers.0.ffn.shared_experts')
 outputs=[]; shared_rows=[]; routed_rows=[]; used=[]; slot_rows=[]
 with torch.inference_mode():
  for t in range(n):
   row=x[t:t+1]; shared_y=shared(row); y=shared_y; routed_y=torch.zeros_like(y); seen=[]; slots=[]
   for k in range(6):
    eid=int(ids[t,k]); seen.append(eid)
    if eid not in cache: cache[eid]=Expert(weights,f'layers.0.ffn.experts.{eid}')
    contribution=cache[eid](row,float(rw[t,k])); routed_y=routed_y+contribution; y=y+contribution; slots.append(contribution.to(torch.bfloat16))
   shared_rows.append(shared_y.to(torch.bfloat16)); routed_rows.append(routed_y.to(torch.bfloat16)); outputs.append(y.to(torch.bfloat16)); used.append(seen); slot_rows.append(torch.cat(slots,dim=0))
   if (t+1)%8==0: print(f'Completed token {t+1}/{n}',flush=True)
 result=torch.cat(outputs).contiguous(); shared_result=torch.cat(shared_rows); routed_result=torch.cat(routed_rows); slot_result=torch.stack(slot_rows); nr=torch.from_numpy(native.view(np.uint16).copy()).view(torch.bfloat16)
 diff=(result.float()-nr.float()).abs(); bits=result.view(torch.int16)!=nr.view(torch.int16)
 def raw_bf16(t): return t.contiguous().view(torch.uint16).reshape(t.shape).numpy().view('|V2')
 np.save(a.output.with_suffix('.shared.bf16.npy'),raw_bf16(shared_result),allow_pickle=False); np.save(a.output.with_suffix('.routed.bf16.npy'),raw_bf16(routed_result),allow_pickle=False)
 native_slots_path=native_root/'encoder.layer0.moe_slots.npy'
 report={'scope':'Independent CPU FP4 layer0 MoE using frozen native route IDs/weights; no routing oracle or full-model qualification','tokens':n,'expert_cache_unique':len(cache),'expert_ids_used':sorted(cache),'bit_mismatch_count':int(bits.sum()),'max_abs_diff':float(diff.max()),'mean_abs_diff':float(diff.mean()),'nonfinite_cpu':int((~torch.isfinite(result)).sum()),'nonfinite_native':int((~torch.isfinite(nr)).sum()),'token_ids':m['token_ids'],'native_manifest_sha256':sha(a.native/'manifest.json'),'gate_manifest_sha256':sha(a.gate/'manifest.json'),'source_sha256':weights.sources,'tensor_sha256':weights.tensor_hashes,'script_sha256':sha(Path(__file__)),'component_files':{'shared':str(a.output.with_suffix('.shared.bf16.npy')),'routed':str(a.output.with_suffix('.routed.bf16.npy'))},'full_model_qualified':False}
 if native_slots_path.exists():
  ns=torch.from_numpy(np.load(native_slots_path,allow_pickle=False).view(np.uint16).copy()).view(torch.bfloat16); sd=(slot_result.float()-ns.float()).abs(); sb=slot_result.view(torch.int16)!=ns.view(torch.int16); report['slot_comparison']={'bit_mismatch_count':int(sb.sum()),'max_abs_diff':float(sd.max()),'mean_abs_diff':float(sd.mean()),'per_rank':[{'rank':k,'bit_mismatch_count':int(sb[:,k].sum()),'max_abs_diff':float(sd[:,k].max()),'mean_abs_diff':float(sd[:,k].mean())} for k in range(6)]}
 report['per_token']=[{'token':t,'bit_mismatch_count':int(bits[t].sum()),'max_abs_diff':float((result[t].float()-nr[t].float()).abs().max()),'selected_experts':used[t]} for t in range(n)]
 a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:report[k] for k in ('expert_cache_unique','bit_mismatch_count','max_abs_diff','mean_abs_diff','nonfinite_cpu','nonfinite_native')},indent=2))
if __name__=='__main__': main()
