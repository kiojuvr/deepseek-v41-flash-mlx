"""Export small official DeepSeek-V4.1 encoding fixtures for offline comparison."""
import argparse, hashlib, importlib.util, json
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    source=a.checkpoint/'encoding'/'encoding.py'; spec=importlib.util.spec_from_file_location('dsv41_encoding',source); enc=importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
    fixtures=[]
    cases=[('chat',[{'role':'user','content':'Hello'}],{'thinking_mode':'chat'}),('thinking',[{'role':'user','content':'Explain the fix.'}],{'thinking_mode':'thinking','reasoning_effort':'low'}),('tool',[{'role':'user','content':'lookup value'},{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'lookup','arguments':'{"query":"value"}'}}]}],{'thinking_mode':'chat'}),('structured_image',[{'role':'user','content':[{'type':'text','text':'inspect'},{'type':'image_url','image_url':{'url':'/tmp/image.png'}}]}],{'thinking_mode':'chat','return_multi_modal_data':True})]
    for name,messages,kwargs in cases:
        value=enc.encode_messages(messages,**kwargs); prompt,media=value if isinstance(value,tuple) else (value,None); fixtures.append({'name':name,'messages':messages,'kwargs':kwargs,'prompt':prompt,'media':media})
    marker=f"literal {enc.IMAGE_PLACEHOLDER}"
    try: enc.encode_messages([{'role':'user','content':marker}],thinking_mode='chat')
    except Exception as exc: fixtures.append({'name':'literal_image_marker','accepted':False,'error_type':type(exc).__name__,'error':str(exc)})
    else: fixtures.append({'name':'literal_image_marker','accepted':True})
    a.output.parent.mkdir(parents=True,exist_ok=True); result={'schema_version':1,'source':str(source),'source_sha256':sha(source),'fixtures':fixtures,'scope':'Official checkpoint encoding.py offline fixture; not production API qualification'}; a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'fixtures':len(fixtures),'source_sha256':result['source_sha256']},indent=2))
if __name__=='__main__': main()
