"""Compare exactness payloads across independently reviewed process roots."""
import json,pathlib,sys
bp,cp=map(pathlib.Path,sys.argv[1:3]);b=json.loads(bp.read_text());c=json.loads(cp.read_text())
for x in (b,c):
 assert x['status']=='exactness_digest_requires_offline_comparison'
for key in ('steps','route_ties','invalid_request_atomicity','hash_continuation'):
 assert b[key]==c[key],key
print(json.dumps({'status':'exact_bitwise_payload_match_requires_raw_log_review_not_promotion','baseline_digest':str(bp),'candidate_digest':str(cp),'steps':len(b['steps']),'route_ties':len(b['route_ties'])},indent=2))
