"""Verify a recorded recipe fixture against the checkpoint encoding source."""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--fixture',type=Path,required=True); a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='dsv41-fixtures-') as tmp:
        regenerated=Path(tmp)/'fixtures.json'
        subprocess.run([sys.executable,'tools/reference/export_recipe_fixtures.py','--checkpoint',str(a.checkpoint),'--output',str(regenerated)],check=True)
        expected=json.loads(a.fixture.read_text()); actual=json.loads(regenerated.read_text())
    if expected != actual: raise SystemExit('fixture mismatch: regenerate and review source/tokenizer identity')
    print(json.dumps({'status':'exact','fixtures':len(actual['fixtures']),'source_sha256':actual['source_sha256'],'tokenizer_sha256':actual['tokenizer_sha256']},indent=2))
if __name__=='__main__': main()
