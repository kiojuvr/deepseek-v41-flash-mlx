"""User-run, resumable natural-cache experiment. No model generation or cache purge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def checked_result(path, mode):
    value = json.loads(path.read_text())
    require(value['status'] == 'passed' and value['mode'] == mode, 'probe status/mode mismatch')
    require(len(value['runs']) == 3, 'probe must report three traces')
    runs = value['runs']
    require([r['trace'] for r in runs] == ['coding', 'random_ids', 'coding'], 'unexpected trace order')
    require(runs[0]['checksum'] == runs[2]['checksum'], 'anchor repeat changed values')
    return value


def main():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    p.add_argument('--binary', type=Path, default=repo/'build/dsv41-engram-probe')
    p.add_argument('--fixtures', type=Path, default=repo/'artifacts/engram')
    p.add_argument('--summary', type=Path, default=repo/'artifacts/checkpoint/summary.json')
    p.add_argument('--output', type=Path, default=repo/'artifacts/engram/cache-runs')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='validate inputs and write the run plan without launching probes')
    args = p.parse_args()
    if not 1 <= args.rounds <= 20:
        p.error('--rounds must be 1..20')
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output == checkpoint or checkpoint in output.parents:
        p.error('output must be outside checkpoint')
    binary = args.binary.resolve()
    fixture_root = args.fixtures.resolve()
    source_provenance = fixture_root/'fixture-provenance.json'
    fixture_manifest = json.loads(source_provenance.read_text())
    for name, expected in fixture_manifest['fixture_sha256'].items():
        require(Path(name).name == name and digest(fixture_root/name) == expected, 'fixture mismatch: '+name)
    require(binary.is_file() and os.access(binary, os.X_OK), 'build the probe first')
    inputs = {'binary_sha256': digest(binary), 'm1_summary_sha256': digest(args.summary),
              'fixture_provenance_sha256': digest(source_provenance),
              'checkpoint': str(checkpoint), 'rounds': args.rounds,
              'runner_sha256': digest(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'runner.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit(f'{lock} exists. Check its PID; remove this lock only after confirming no runner is active.')
    os.write(lock_fd, f'{os.getpid()}\n'.encode())
    os.close(lock_fd)
    try:
        plan_path = output/'plan.json'
        if plan_path.exists():
            require(args.resume, 'output already has a plan; use --resume or a new --output')
            plan = json.loads(plan_path.read_text())
            require(plan['inputs'] == inputs, 'inputs/build changed; use a new output directory')
        else:
            runs = []
            for round_index in range(args.rounds):
                order = ['mmap','pread'] if round_index%2 == 0 else ['pread','mmap']
                for mode in order:
                    runs.append({'id':f'{len(runs):02d}-{mode}', 'mode':mode})
            plan = {'inputs':inputs, 'runs':runs,
                    'scope':'Natural OS-cache Engram-only experiment; no backbone pressure, no global-cold claim.',
                    'measurement':'per-input-token hash + gather + CPU BF16; trace accounting is outside timed region'}
            atomic_json(plan_path, plan)
        print(f'Plan: {plan_path}', flush=True)
        if args.prepare_only:
            return
        for run in plan['runs']:
            result = output/(run['id']+'.json')
            done = output/(run['id']+'.done.json')
            if done.exists():
                require(args.resume, 'completed runs require --resume')
                marker = json.loads(done.read_text())
                require(marker['sha256'] == digest(result), 'completed result changed')
                checked_result(result, run['mode'])
                print(f'Skip verified {run["id"]}', flush=True)
                continue
            log = output/(run['id']+'.log')
            command = [str(binary),'--checkpoint',str(checkpoint),'--m1-summary',str(args.summary.resolve()),
                       '--fixtures',str(fixture_root),'--output',str(result),'--mode',run['mode']]
            print(f'Start {run["id"]}; log {log}', flush=True)
            started = time.time()
            with log.open('w') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if code != 0:
                raise RuntimeError(f'{run["id"]} failed with exit {code}; inspect {log}; resume retries this run')
            checked_result(result, run['mode'])
            atomic_json(done, {'sha256':digest(result),'wall_seconds':time.time()-started,'command':command})
        # Compare identical traces across every process/mode, not just repeated anchors.
        checksums = {}
        aggregate = []
        for run in plan['runs']:
            value = checked_result(output/(run['id']+'.json'), run['mode'])
            for position, r in enumerate(value['runs']):
                key = r['trace']
                require(checksums.setdefault(key,r['checksum']) == r['checksum'], f'cross-mode mismatch: {key}')
                aggregate.append({'run':run['id'],'position':position,'trace':key,
                                  'mean_lookup_ms':r['mean_lookup_ms'],'p99_ms':r['p99_ms'],
                                  'unique_file_pages':r['unique_file_pages'],
                                  'process_diskio_bytesread_delta':r['telemetry_after']['diskio_bytesread']-r['telemetry_before']['diskio_bytesread'],
                                  'footprint_after':r['telemetry_after']['phys_footprint']})
        atomic_json(output/'comparison.json', {'status':'passed','full_model_qualified':False,
                    'checksums':checksums,'runs':aggregate,
                    'limitations':'Order alternates but caches are inherited. Process diskio is not a block-device bandwidth or cache-hit-rate measurement. Re-runs after interruption further warm the cache.'})
        print(f'Complete: {output/"comparison.json"}', flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; completed runs preserved. Re-run with --resume.', file=sys.stderr)
        sys.exit(130)
