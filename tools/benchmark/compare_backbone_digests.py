"""Compare process-separated exactness manifests."""
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);b=json.loads((root/'baseline/digest.json').read_text());c=json.loads((root/'candidate/digest.json').read_text())
assert b['status']==c['status']=='exactness_digest_requires_offline_comparison'
assert b['wired_limit_bytes']==0 and not b['expert_backing_file_backed'] and not b['resident_expert_atlas'] and b['compact_expert_bank']
assert c['wired_limit_bytes']==int(sys.argv[2]) and c['expert_backing_file_backed'] and c['resident_expert_atlas'] and not c['compact_expert_bank']
for key in ('steps','route_ties','invalid_request_atomicity','hash_continuation'):
 assert b[key]==c[key],key
print(json.dumps({'status':'exact_bitwise_match_requires_raw_log_review_not_promotion','steps':len(b['steps']),'route_ties':len(b['route_ties']),'baseline_wired':b['wired_limit_bytes'],'candidate_wired':c['wired_limit_bytes']},indent=2))
