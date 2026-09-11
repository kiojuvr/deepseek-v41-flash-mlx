"""Orchestration tests with a tiny fake probe, without model I/O."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[2]/'tools/benchmark/run_engram_cache.py'


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixture = self.root/'fixture'
        self.fixture.mkdir()
        (self.fixture/'fixture-provenance.json').write_text(json.dumps({'fixture_sha256':{}}))
        self.checkpoint = self.root/'checkpoint'
        self.checkpoint.mkdir()
        self.summary = self.root/'summary.json'
        self.summary.write_text('{}')
        self.output = self.root/'results'
        self.binary = self.root/'probe'
        self.binary.write_text('''#!/usr/bin/env python3
import json,sys
from pathlib import Path
a=dict(zip(sys.argv[1::2],sys.argv[2::2]))
runs=[]
for name,checksum in [('coding',1),('random_ids',2),('coding',1)]:
 runs.append(dict(trace=name,checksum=checksum,mean_lookup_ms=1,p99_ms=2,unique_file_pages=3,
 telemetry_before=dict(diskio_bytesread=0),telemetry_after=dict(diskio_bytesread=1,phys_footprint=2)))
Path(a['--output']).write_text(json.dumps(dict(status='passed',mode=a['--mode'],runs=runs)))
''')
        self.binary.chmod(0o700)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *extra, ok=True):
        result = subprocess.run([sys.executable,str(RUNNER),'--checkpoint',str(self.checkpoint),
                                 '--binary',str(self.binary),'--fixtures',str(self.fixture),
                                 '--summary',str(self.summary),'--output',str(self.output),
                                 '--rounds','2',*extra],capture_output=True,text=True)
        if ok:
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0)
        return result

    def test_prepare_does_not_launch_then_resume(self):
        self.run_script('--prepare-only')
        plan=json.loads((self.output/'plan.json').read_text())
        self.assertEqual([r['mode'] for r in plan['runs']],['mmap','pread','pread','mmap'])
        self.assertFalse(list(self.output.glob('*.done.json')))
        self.run_script('--resume')
        self.assertEqual(len(list(self.output.glob('*.done.json'))),4)
        self.assertEqual(json.loads((self.output/'comparison.json').read_text())['status'],'passed')
        saved=(self.output/'00-mmap.json').stat().st_mtime_ns
        self.run_script('--resume')
        self.assertEqual(saved,(self.output/'00-mmap.json').stat().st_mtime_ns)

    def test_resume_rejects_changed_binary(self):
        self.run_script('--prepare-only')
        with self.binary.open('a') as f:
            f.write('\n# changed\n')
        self.run_script('--resume',ok=False)
        self.assertFalse((self.output/'runner.lock').exists())

    def test_result_corruption_detected(self):
        self.run_script()
        (self.output/'00-mmap.json').write_text('{}')
        self.run_script('--resume',ok=False)

    def test_failed_run_retry(self):
        flag=self.root/'fail-once'
        flag.touch()
        body=self.binary.read_text().replace('runs=[]',f'''flag=Path({str(flag)!r})
if flag.exists():
 flag.unlink()
 sys.exit(7)
runs=[]''')
        self.binary.write_text(body)
        self.run_script(ok=False)
        self.assertFalse(list(self.output.glob('*.done.json')))
        self.run_script('--resume')
        self.assertEqual(len(list(self.output.glob('*.done.json'))),4)

    def test_existing_output_and_lock(self):
        self.run_script('--prepare-only')
        self.run_script(ok=False)
        (self.output/'runner.lock').write_text(str(os.getpid()))
        self.run_script('--resume',ok=False)
        self.assertTrue((self.output/'runner.lock').exists())


if __name__=='__main__':
    unittest.main()
