"""Offline Engram forward fixtures; CUDA projection is explicitly not executed."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--m1-summary', type=Path, required=True)
    p.add_argument('--row-fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.checkpoint.resolve(), args.output.resolve()
    require(out != root and root not in out.parents, 'output must be outside checkpoint')
    verification = json.loads(args.verification.read_text())
    summary = json.loads(args.m1_summary.read_text())
    pins = {r['path']:r for r in verification['files']}
    sources = ['inference/config.json','inference/model.py','inference/kernel.py']
    require(verification['status'] == summary['status'] == 'verified', 'M1 verification required')
    require(verification['revision'] == summary['revision'], 'M1 revision mismatch')
    for name in sources:
        require(sha(root/name) == pins[name]['sha256'], 'source mismatch: '+name)
    old = json.loads((args.row_fixtures/'fixture-provenance.json').read_text())
    require(old['revision'] == verification['revision'], 'row fixture revision mismatch')
    for name,h in old['fixture_sha256'].items():
        require(Path(name).name == name and sha(args.row_fixtures/name) == h, 'row fixture mismatch')
    config = json.loads((root/'inference/config.json').read_text())
    require(config['dim'] == 5120 and config['hc_mult'] == 4 and config['dtype'] == 'fp8', 'unsupported config')
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    def tensor(name, rows=None):
        shard = index[name]
        require(Path(shard).name == shard, 'unsafe shard')
        path = root/shard
        with path.open('rb') as f:
            n = struct.unpack('<Q',f.read(8))[0]
            require(n < 64*1024*1024, 'header too large')
            raw = f.read(n)
            require(hashlib.sha256(raw).hexdigest() == summary['shards'][shard]['header_sha256'], 'header mismatch')
            require(os.fstat(f.fileno()).st_size == summary['shards'][shard]['file_bytes'], 'file size mismatch')
            h = json.loads(raw)[name]
            dtype = {'BF16':torch.bfloat16,'F8_E4M3':torch.float8_e4m3fn,'F8_E8M0':torch.float8_e8m0fnu}[h['dtype']]
            shape = h['shape']
            base = 8+n+h['data_offsets'][0]
            if rows is None:
                f.seek(base); data = f.read(h['data_offsets'][1]-h['data_offsets'][0])
                require(len(data) == h['data_offsets'][1]-h['data_offsets'][0], 'short tensor read')
            else:
                width = shape[1]
                require(h['dtype'].startswith('F8_') and all(0 <= r < shape[0] for r in rows), 'bad row read')
                chunks=[]
                for r in rows:
                    f.seek(base+r*width); b=f.read(width)
                    require(len(b)==width,'short row read'); chunks.append(b)
                data=b''.join(chunks); shape=[len(rows),width]
            return torch.frombuffer(bytearray(data),dtype=dtype).reshape(shape)
    # Literal official Engram.forward executes with frozen lookup/projection boundaries.
    tree = ast.parse((root/'inference/model.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engram')
    # Capture diagnostics at the existing return; arithmetic statements are unchanged.
    forward = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    returned = forward.body[-1]
    require(isinstance(returned,ast.Return),'official forward return structure changed')
    returned.value = ast.Tuple(elts=[returned.value,ast.Name(id='dot',ctx=ast.Load()),
                                    ast.Name(id='gate',ctx=ast.Load())],ctx=ast.Load())
    ast.fix_missing_locations(cls)
    ns={'torch':torch,'nn':torch.nn,'ModelArgs':object,'EngramLayout':object}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(root/'inference/model.py'),'exec'),ns)
    torch.set_num_threads(4)
    torch.manual_seed(4110912)
    out.mkdir(parents=True,exist_ok=True)
    files=[]
    def save(name,t):
        data=t.contiguous().view(torch.uint8).numpy().tobytes()
        (out/name).write_bytes(data); files.append(name)
    # Cover every finite BF16 bit pattern; non-finite encodings and padding become zero.
    finite_bits=torch.arange(11*6144,dtype=torch.int32)
    finite_bits=torch.where((finite_bits<65536)&((finite_bits&0x7f80)!=0x7f80),finite_bits,0)
    finite_values=finite_bits.to(torch.uint16).view(torch.bfloat16).reshape(11,6144)
    finite_blocks=finite_values.float().reshape(11,192,32)
    maxima=finite_blocks.abs().amax(-1).clamp_min(1e-4)
    raw=(maxima*(1.0/448.0)).view(torch.int32)
    powers=((raw>>23)&255)-127+((raw&0x7fffff)!=0).int()
    finite_scales=torch.ldexp(torch.ones_like(maxima),powers)
    finite_quant=(finite_blocks/finite_scales.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
    save('finite-bf16-input.bf16',finite_values)
    save('finite-bf16-quant.u8',finite_quant.view(torch.uint8))
    save('finite-bf16-scales.u8',finite_scales.to(torch.float8_e8m0fnu).view(torch.uint8))
    rows_all = torch.frombuffer(bytearray((args.row_fixtures/'coding.rows.u64').read_bytes()),dtype=torch.int64).reshape(-1,2,24)
    cases=[]
    with torch.inference_mode():
        for li,layer in enumerate([1,14]):
            rows=rows_all[:3,li,:].reshape(-1).tolist()
            prefix=f'layers.{layer}.engram.'
            v=tensor(prefix+'embed.weight',rows).float().reshape(3,24,8,32)
            s=tensor(prefix+'embed.scale',rows).float().reshape(3,24,8,1)
            values=(v*s).reshape(3,6144).to(torch.bfloat16)
            # Source-derived act_quant: FP32 absmax, clamp, ceil power-of-two, E4M3 RNE.
            blocks=values.float().reshape(3,192,32)
            amax=blocks.abs().amax(-1).clamp_min(1e-4)
            bits=(amax*(1.0/448.0)).view(torch.int32)
            exponents=((bits>>23)&255)-127+((bits&0x7fffff)!=0).int()
            scales=torch.ldexp(torch.ones_like(amax),exponents)
            quant=(blocks/scales.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
            weight=tensor(prefix+'wkv.weight')
            weight_scale=tensor(prefix+'wkv.scale').float()
            require(tuple(weight.shape)==(25600,6144) and tuple(weight_scale.shape)==(800,192),'bad projection shape')
            # CPU transcription of kernel.py's 32-wide block math. No CUDA oracle claim.
            accumulator=torch.zeros(3,25600,dtype=torch.float32)
            for block in range(192):
                partial=F.linear(quant[:,block,:].float(),weight[:,block*32:(block+1)*32].float())
                scaled=partial*scales[:,block: block+1]
                scaled=scaled*weight_scale[:,block].repeat_interleave(32)
                accumulator=accumulator+scaled
            kv=accumulator.to(torch.bfloat16)
            q,k=tensor(prefix+'q_weight'),tensor(prefix+'k_weight')
            hidden=torch.randn(3,4,5120).to(torch.bfloat16)
            hidden[0]=0  # exact zero norm exercises eps and positive-zero signed sqrt.
            mask=torch.tensor([True,False,True])
            frozen=SimpleNamespace(dim=5120,hc_mult=4,eps=config['norm_eps'],clamp_value=1e-6,
                q_weight=q,k_weight=k,embed=lambda _:values.reshape(1,3,24,256),
                wkv=lambda _:kv.reshape(1,3,25600))
            expected,dot,gate=ns['Engram'].forward(frozen,hidden.unsqueeze(0),None,mask.unsqueeze(0))
            expected=expected.squeeze(0)
            require(torch.isfinite(expected).all().item(),'non-finite official gate output')
            require(torch.equal(expected[1].view(torch.int16),hidden[1].view(torch.int16)),'masked residual changed')
            tag=f'layer-{layer}'
            for suffix,t in [('hidden.bf16',hidden),('quant.u8',quant.reshape(3,6144).view(torch.uint8)),
                             ('scales.u8',scales.to(torch.float8_e8m0fnu).view(torch.uint8)),('kv.bf16',kv),
                             ('q.bf16',q),('k.bf16',k),('expected.bf16',expected),
                             ('dot.f32',dot),('gate.f32',gate)]: save(tag+'-'+suffix,t)
            cases.append({'layer_id':layer,'layer_index':li,'tokens':3,'rows':rows,'mask':mask.tolist(),'tag':tag})
            print(f'Exported layer {layer}: CPU-transcribed projection + official gate/residual.',flush=True)
    manifest={'schema_version':1,'revision':verification['revision'],'source_sha256':{n:sha(root/n) for n in sources},
              'exporter_sha256':sha(__file__),'row_fixture_provenance_sha256':sha(args.row_fixtures/'fixture-provenance.json'),
              'versions':{'python':sys.version.split()[0],'torch':torch.__version__},'norm_eps':config['norm_eps'],
              'cases':cases,'fixture_sha256':{name:sha(out/name) for name in files},
              'finite_bf16_patterns':65280,
              'oracle_scope':'Official Engram.forward with frozen lookup/projection; full CUDA fp8_gemm/act_quant NOT executed.',
              'projection_scope':'CPU transcription of kernel.py; PyTorch FP32 block GEMM, per-block scaling and BF16 cast. Reduction-order agreement with CUDA unqualified.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
