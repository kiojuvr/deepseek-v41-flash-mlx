"""Offline file-backed residency gates; no production runtime dependency."""
import json, pathlib, re, sys
root=pathlib.Path(sys.argv[1]); limit=int(sys.argv[2])
pairs=[]
if (root/'baseline').is_dir(): pairs=[(root/'baseline',root/'candidate')]
else:
 for p in sorted(root.glob('pair-*')):
  if p.is_dir(): pairs.append((p/'baseline',p/'candidate'))
assert pairs, 'no complete pair directories'
def load(path, expected):
 r=json.loads((path/'result.json').read_text())
 assert r['status']=='measurement_completed_requires_review'
 assert r['context_tokens']==2071 and r['decode_tokens']==8
 assert r['wired_limit_bytes']==expected, (path,r.get('wired_limit_bytes'))
 assert r.get('expert_backing_file_backed',False)==bool(expected), (path,r.get('expert_backing_file_backed'))
 assert not r.get('component_profile',False) and not r.get('decode_stack_graph',False)
 assert r['packed_expert_bank_constructions']==40
 assert r['route_execution_stats']['diagnostic_readbacks']==0
 assert r['attention_telemetry']['index_host_readbacks']==0
 resource=(path/'resource.log').read_text()
 assert re.search(r'^\s+0\s+swaps$',resource,re.M), 'process swap'
 footprint=re.search(r'^\s+(\d+)\s+peak memory footprint$',resource,re.M)
 assert footprint and int(footprint[1])<=340_000_000_000, 'footprint'
 if expected:
  assert f'residency: requested={expected} applied={expected}' in resource
  assert 'residency: restored=0 bytes' in resource
 def vm(name):
  text=(path/name).read_text()
  return {k:int(v) for k,v in re.findall(r'(Swapins|Swapouts):\s*(\d+)',text)}
 before,after=vm('system-before.txt'),vm('system-after.txt')
 assert before==after, f'system swap delta: {before} -> {after}'
 return r
def metrics(r):
 d=r['phases']['decode']; lat=d['latency_seconds'][1:]
 return {'advancing_mean_seconds':sum(lat)/len(lat),
  'advancing_p95_seconds':sorted(lat)[int(.95*(len(lat)-1))],
  'full_path_seconds':r['aggregates']['prefill_seconds']+sum(d['latency_seconds']),
  'prefill_seconds':r['aggregates']['prefill_seconds']}
rows=[]
for baseline,candidate in pairs:
 b,c=load(baseline,0),load(candidate,limit)
 for key in ('generated_token_ids','state_position','next_position'):
  assert b['phases']['decode'][key]==c['phases']['decode'][key],key
 bm,cm=metrics(b),metrics(c)
 rows.append({'baseline':bm,'candidate':cm,'ratio':{k:cm[k]/bm[k] for k in bm}})
means={side:{k:sum(row[side][k] for row in rows)/len(rows) for k in rows[0][side]} for side in ('baseline','candidate')}
print(json.dumps({'status':'gates_passed_requires_raw_log_review_not_promotion','pairs':rows,
 'mean':{**means,'ratio':{k:means['candidate'][k]/means['baseline'][k] for k in means['baseline']}}},indent=2))
