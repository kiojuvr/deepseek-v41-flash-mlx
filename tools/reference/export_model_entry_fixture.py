"""Offline real embedding rows + AST-extracted official RMSNorm CPU fixtures."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--m1-summary',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(a.checkpoint.resolve()):
        raise ValueError('checkpoint read-only')
    summary=json.loads(a.m1_summary.read_text()); verification=json.loads(a.verification.read_text())
    if summary['status']!='verified' or verification['status']!='verified' or summary['revision']!=verification['revision']:
        raise ValueError('M1 identity mismatch')
    source=a.checkpoint/'inference/model.py'
    pin=next(x for x in verification['files'] if x['path']=='inference/model.py')
    if sha(source)!=pin['sha256']:
        raise ValueError('official source changed')
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def tensor(name, rows=None):
        shard=index[name]
        if Path(shard).name!=shard:
            raise ValueError('invalid shard')
        path=a.checkpoint/shard
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            if not 0<length<64*1024*1024:
                raise ValueError('invalid header')
            header=f.read(length)
            if hashlib.sha256(header).hexdigest()!=summary['shards'][shard]['header_sha256'] or path.stat().st_size!=summary['shards'][shard]['file_bytes']:
                raise ValueError('shard identity mismatch')
            spec=json.loads(header)[name]
            if spec['dtype']!='BF16':
                raise ValueError('BF16 required')
            offset=8+length+spec['data_offsets'][0]
            if rows is None:
                f.seek(offset); data=f.read(spec['data_offsets'][1]-spec['data_offsets'][0]); shape=spec['shape']
            else:
                width=spec['shape'][1]*2
                chunks=[]
                for row in rows:
                    if not 0<=row<spec['shape'][0]:
                        raise ValueError('row outside tensor')
                    f.seek(offset+row*width); chunks.append(f.read(width))
                data=b''.join(chunks); shape=[len(rows),spec['shape'][1]]
        return torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(shape)
    ids=[0,1,127,128,129279,1,0]
    embed=tensor('embed.weight',ids)
    weight=tensor('layers.0.attn_norm.weight')
    # Exercise finite zero/tiny/large inputs as well as the real embedding rows.
    extra=torch.stack([torch.zeros(5120),torch.full((5120,),2**-60),torch.full((5120,),2**60)]).to(torch.bfloat16)
    norm_input=torch.cat([embed,extra])
    tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RMSNorm')
    ns={'torch':torch,'nn':torch.nn}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
    torch.set_num_threads(4)
    norm=ns['RMSNorm'].forward(SimpleNamespace(weight=weight,eps=1e-20),norm_input)
    a.output.mkdir(parents=True,exist_ok=True)
    tensors={'embedding.bf16':embed,'norm-weight.bf16':weight,'norm-input.bf16':norm_input,'norm-expected.bf16':norm}
    for name,t in tensors.items():
        (a.output/name).write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
    manifest={'schema_version':1,'revision':summary['revision'],'token_ids':ids,'norm_tokens':len(norm_input),'norm_eps':1e-20,
              'source_sha256':{'inference/model.py':sha(source)},'exporter_sha256':sha(Path(__file__)),
              'fixture_sha256':{name:sha(a.output/name) for name in tensors},'torch_version':torch.__version__,
              'scope':'Selected real embedding rows and official RMSNorm.forward on CPU; not full model or CUDA qualification.'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
