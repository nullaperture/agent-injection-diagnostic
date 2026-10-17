"""Explicit target execution only on separately authorized remote compute."""
import argparse,json,os,runpy,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',choices=['muse','qwen38','qwen36'],required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--execute-model',action='store_true');p.add_argument('--confirm-energy',action='store_true');a=p.parse_args()
if a.confirm_energy and a.model!='qwen36': p.error('--confirm-energy requires qwen36')
if not a.execute_model: p.error('Explicit --execute-model required')
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from sample_runtime.qwen import require_remote
require_remote()
s=a.state.resolve()
if not a.state.is_absolute() or any((x/'.git').exists() for x in (s,*s.parents)): raise ValueError('Absolute native state outside Git required')
s.mkdir(parents=True,exist_ok=False)
from prepare import encoded
(s/'requests.json').write_bytes(encoded())
(s/'worker-config.json').write_text(json.dumps({'model':'muse' if a.model=='muse' else 'qwen','profile':a.model}))
worker='repeat_worker.py' if a.confirm_energy else ('worker_qwen36.py' if a.model=='qwen36' else 'worker.py')
sys.argv=[worker,'--requests',str(s/'requests.json'),'--output',str(s/'outputs'),'--deadline',str(time.time()+2400)]
runpy.run_path(str(Path(__file__).parent/worker),run_name='__main__')
