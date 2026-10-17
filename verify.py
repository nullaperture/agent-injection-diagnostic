"""Offline artifact integrity only. No target imports or downloads."""
import hashlib,json
from pathlib import Path
r=Path(__file__).parent;m=json.loads((r/'manifest.json').read_text());actual={p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert actual==set(m['files'])|{'manifest.json'}
for name,row in m['files'].items():
 b=(r/name).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],name
s=json.loads((r/'results.json').read_text());assert len(s['models'])==3
assert {x['model']['model'] for x in s['models']}=={'meta-models/Muse-Glimmer-30B','Qwen/Qwen3.8-27B','Qwen/Qwen3.6-27B'}
assert all(len(x['rows'])==24 and len({z['id'] for z in x['rows']})==24 for x in s['models'])
for x in s['models']:
 for family in ('clean','authority','dependency','contamination'):
  rows=[z for z in x['rows'] if z['id'].startswith(family+'/')];assert len(rows)==6
  for rating in ('pass','fail','uncertain'):assert sum(z['assessment']['utility']==rating for z in rows)==x['families'][family].get(rating,0)
assert s['tools_executed'] is False and s['defense_applied'] is False
for p in r.rglob('*.py'): compile(p.read_bytes(),str(p),'exec')
print(json.dumps({'integrity':'pass','models':3,'responses':72,'target_executed':False}))
