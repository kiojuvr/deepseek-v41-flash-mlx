"""Offline full-path gates; no production runtime dependency."""
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
reports = [json.loads((root / v / 'result.json').read_text()) for v in ('baseline', 'candidate')]
b, c = reports
cumulative = 'turns' in b
assert ('turns' in c) == cumulative
rows = lambda r: r['turns'] if cumulative else [r['phases']['decode']]
assert len(rows(b)) == len(rows(c))
for x, y in zip(rows(b), rows(c)):
    for key in ('generated_token_ids', 'state_position', 'next_position'):
        assert x[key] == y[key], key
for enabled, (variant, r) in enumerate(zip(('baseline', 'candidate'), reports)):
    if cumulative:
        assert r['scheduled_context_tokens'] == r['max_context_tokens'] == 65536
        assert r['turn_count'] == len(r['turns']) == 8
        assert [t['next_position'] for t in r['turns']] == list(range(8192, 65537, 8192))
        assert all(len(t['generated_token_ids']) == 64 for t in r['turns'])
    else:
        assert r['context_tokens'] == 2071 and r['decode_tokens'] == 8
    assert r['status'] == 'measurement_completed_requires_review'
    assert r['decode_stack_graph'] == bool(enabled)
    assert not r.get('grouped_expert_pipeline', False)
    assert r['packed_expert_bank_constructions'] == 40
    assert r['route_execution_stats']['diagnostic_readbacks'] == 0
    for key in ('index_host_readbacks', 'chunk_scalar_qk_calls', 'chunk_scalar_av_calls'):
        assert r['attention_telemetry'][key] == 0, key
    advancing = sum(len(t['generated_token_ids']) - 1 for t in rows(r))
    assert r['sweep_telemetry']['deferred_layer_evaluations'] == enabled * advancing * 40
    assert r['sweep_telemetry']['decode_stack_evaluations'] == enabled * advancing * 2
    resource = (root / variant / 'resource.log').read_text()
    assert re.search(r'^\s+0\s+swaps$', resource, re.M), 'process swap'
    footprint = re.search(r'^\s+(\d+)\s+peak memory footprint$', resource, re.M)
    assert footprint and int(footprint[1]) <= 340_000_000_000, 'footprint'
    def vm(name):
        text = (root / variant / name).read_text()
        return {k: int(v) for k, v in re.findall(r'(Swapins|Swapouts):\s*(\d+)', text)}
    before, after = vm('system-before.txt'), vm('system-after.txt')
    assert set(before) == set(after) == {'Swapins', 'Swapouts'}, 'missing VM counters'
    assert before == after, 'system swap delta'
def metrics(r):
    latency = [v for t in rows(r) for v in t['decode_latency_seconds' if cumulative else 'latency_seconds'][1:]]
    late = rows(r)[-1]['decode_latency_seconds' if cumulative else 'latency_seconds'][1:]
    return {'advancing_mean_seconds': sum(latency)/len(latency),
            'advancing_p95_seconds': sorted(latency)[int(.95*(len(latency)-1))],
            'late_turn_advancing_mean_seconds': sum(late)/len(late),
            'full_path_seconds': r['aggregates']['session_wall_seconds'] if cumulative else
                r['aggregates']['prefill_seconds'] + sum(r['phases']['decode']['latency_seconds'])}
bm, cm = metrics(b), metrics(c)
print(json.dumps({'status': 'gates_passed_requires_raw_log_review_not_promotion',
                  'baseline': bm, 'candidate': cm,
                  'ratio': {k: cm[k]/bm[k] for k in bm}}, indent=2))
