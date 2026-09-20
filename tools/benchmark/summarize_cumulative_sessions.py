#!/usr/bin/env python3
"""Summarize completed cumulative-session stages without declaring qualification."""
import argparse
import json
import re
from pathlib import Path


RESOURCE_LABELS = {
    'maximum resident set size': 'maximum_resident_set_bytes',
    'peak memory footprint': 'peak_footprint_bytes',
    'page faults': 'page_faults',
    'page reclaims': 'page_reclaims',
    'swaps': 'swaps',
}
VM_COUNTERS = ('Compressions', 'Decompressions', 'Swapins', 'Swapouts')
VM_GAUGES = ('Pages stored in compressor', 'Pages occupied by compressor')


def resource_metrics(path: Path):
    found = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*(\d+)\s+(.+?)\s*$', line)
        if match and match.group(2) in RESOURCE_LABELS:
            found[RESOURCE_LABELS[match.group(2)]] = int(match.group(1))
    return found


def vm_metrics(path: Path):
    text = path.read_text()
    page_match = re.search(r'page size of (\d+) bytes', text)
    page_size = int(page_match.group(1)) if page_match else None
    values = {}
    for label in VM_COUNTERS + VM_GAUGES:
        match = re.search(rf'^{re.escape(label)}:\s+(\d+)\.?$', text, re.MULTILINE)
        if match:
            values[label] = int(match.group(1))
    values['page_size_bytes'] = page_size
    return values


def vm_summary(before, after):
    result = {'page_size_bytes': after.get('page_size_bytes') or before.get('page_size_bytes')}
    for label in VM_COUNTERS:
        if label in before and label in after:
            result[label.lower() + '_delta'] = after[label] - before[label]
    for label in VM_GAUGES:
        key = label.lower().replace(' ', '_')
        if label in before:
            result[key + '_before'] = before[label]
        if label in after:
            result[key + '_after'] = after[label]
    return result


def summarize_stage(root: Path, name: str, expected_context: int):
    marker = root / name / 'completed-run.txt'
    if not marker.exists():
        return {'status': 'not_completed'}
    run = Path(marker.read_text().strip())
    required = ('result.json', 'resource.log', 'system-before.txt', 'system-after.txt')
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        return {'status': 'missing_artifact', 'run': str(run), 'missing': missing}
    result = json.loads((run / 'result.json').read_text())
    resources = resource_metrics(run / 'resource.log')
    vm = vm_summary(vm_metrics(run / 'system-before.txt'), vm_metrics(run / 'system-after.txt'))
    attention = result['attention_telemetry']
    final = result['final_state']
    checks = {
        'measurement_completed': result.get('status') == 'measurement_completed_requires_review',
        'exact_final_position': final['next_position'] == expected_context,
        'resident_bank_count': result.get('packed_expert_bank_constructions') == 40,
        'route_host_readbacks_zero': result['route_execution_stats']['diagnostic_readbacks'] == 0,
        'index_host_readbacks_zero': attention['index_host_readbacks'] == 0,
        'scalar_qk_zero': attention['chunk_scalar_qk_calls'] == 0,
        'scalar_av_zero': attention['chunk_scalar_av_calls'] == 0,
        'swap_zero': resources.get('swaps') == 0,
        'footprint_within_340gb': resources.get('peak_footprint_bytes', 340_000_000_001) <= 340_000_000_000,
    }
    turns = result['turns']
    recovery = result['recovery']
    return {
        'status': 'review_required' if all(checks.values()) else 'gate_failed',
        'run': str(run),
        'checks': checks,
        'turn_count': len(turns),
        'turn_end_positions': [turn['next_position'] for turn in turns],
        'append_seconds': [turn['append_seconds'] for turn in turns],
        'first_decode_seconds': [turn['first_decode_seconds'] for turn in turns],
        'decode_mean_seconds': [turn['decode_mean_seconds'] for turn in turns],
        'decode_p95_seconds': [turn['decode_p95_seconds'] for turn in turns],
        'turn_peak_bytes': [turn['memory_after']['peak_bytes'] for turn in turns],
        'aggregates': result['aggregates'],
        'final_memory': final['memory'],
        'recovery': recovery,
        'active_recovered_by_state_reset': final['memory']['active_bytes'] - recovery['after_state_reset']['active_bytes'],
        'cache_recovered_by_idle_clear': recovery['after_state_reset']['cache_bytes'] - recovery['after_idle_cache_clear']['cache_bytes'],
        'resources': resources,
        'vm': vm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary = {
        'schema_version': 1,
        'status': 'review_required',
        'qualification_claim': False,
        'stages': {
            '32k-cumulative': summarize_stage(args.root, '32k-cumulative', 32768),
            '64k-cumulative': summarize_stage(args.root, '64k-cumulative', 65536),
        },
    }
    if any(stage['status'] != 'review_required' for stage in summary['stages'].values()):
        summary['status'] = 'incomplete_or_gate_failed'
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
