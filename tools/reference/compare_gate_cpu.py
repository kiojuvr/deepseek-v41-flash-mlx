"""Compare native layer-0 gate trace with the official formula on CPU."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cpu_reference import Weights, sha

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--native',type=Path,required=True); p.add_argument('--gate',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.checkpoint.resolve()): p.error('fresh output outside checkpoint required')
    m=json.loads((a.native/'manifest.json').read_text()); gm=json.loads((a.gate/'manifest.json').read_text())
    if m['token_ids'] != gm['token_ids']: raise ValueError('token IDs mismatch')
    n=len(m['token_ids']); raw=np.load(a.native/'encoder.layer0.ffn_in.npy',allow_pickle=False)
    if raw.shape != (n,5120) or raw.dtype.itemsize != 2: raise ValueError('invalid FFN input')
    x=torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16)
    weights=Weights(a.checkpoint,'artifacts/checkpoint/summary.json','artifacts/checkpoint/verification.json')
    wb,wd=weights.tensor('layers.0.ffn.gate.weight'); bb,bd=weights.tensor('layers.0.ffn.gate.bias')
    if wd!='BF16' or bd!='F32': raise ValueError('unexpected gate dtype')
    z=F.linear(x.float(),wb.float()); scores=torch.sqrt(F.softplus(z)); corrected=scores+bb.float()
    order=torch.argsort(corrected,dim=1,descending=True,stable=True); ids=order[:,:6]; route=scores.gather(1,ids); route=route/(route.sum(1,keepdim=True)+1e-20)*1.5
    native_raw=torch.from_numpy(np.load(a.gate/'encoder.layer0.gate_raw_scores.npy',allow_pickle=False).copy()).reshape(n,384)
    native_corr=torch.from_numpy(np.load(a.gate/'encoder.layer0.gate_corrected_scores.npy',allow_pickle=False).copy()).reshape(n,384)
    native_route=torch.from_numpy(np.load(a.gate/'encoder.layer0.route_weights.npy',allow_pickle=False).copy()).reshape(n,6)
    def stats(a,b):
        d=(a-b).abs(); return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'nonfinite':int((~torch.isfinite(a)).sum()+(~torch.isfinite(b)).sum())}
    selected=gm['selected_experts']; native_ids=torch.tensor(selected,dtype=torch.int64).reshape(n,6)
    top7=order[:,:7]; margins=corrected.gather(1,top7)[:,5]-corrected.gather(1,top7)[:,6]
    report={'scope':'Official gate CPU formula vs native layer0 gate trace; no expert output or full-model qualification','raw_scores':stats(native_raw,scores),'preactivation_cpu_probe':stats(native_raw,z),'corrected_scores':stats(native_corr,corrected),'route_weights':stats(native_route,route),'selected_id_mismatch':int((native_ids!=ids).sum()),'margin_min':float(margins.min()),'margin_max':float(margins.max()),'margin_below_1e-3':int((margins.abs()<1e-3).sum()),'token_ids':m['token_ids'],'native_manifest_sha256':sha(a.native/'manifest.json'),'gate_manifest_sha256':sha(a.gate/'manifest.json'),'source_sha256':weights.sources,'tensor_sha256':weights.tensor_hashes,'script_sha256':sha(Path(__file__)),'full_model_qualified':False}
    report['per_token']= [{'token':i,'id_mismatch':int((native_ids[i]!=ids[i]).sum()),'cpu_ids':ids[i].tolist(),'native_ids':native_ids[i].tolist(),'margin':float(margins[i])} for i in range(n)]
    a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:report[k] for k in ('raw_scores','corrected_scores','route_weights','selected_id_mismatch','margin_min','margin_max','margin_below_1e-3')},indent=2))
if __name__=='__main__': main()
