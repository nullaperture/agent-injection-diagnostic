"""Fresh-process confirmation of the fixed Qwen3.6 energy pair, twice each."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
from sample_runtime.cli import write
from sample_runtime.qwen import require_remote
from sample_runtime.qwen36 import load_qwen36
from sample_runtime.tools import parse_calls
from prepare import encoded
from validation_support import validate, assess

def main():
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    require_remote()
    seconds=int(a.deadline-time.time())
    if not 1<=seconds<=2700:raise ValueError('Finite deadline required')
    def expired(*_):raise TimeoutError('Replication worker deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(seconds)
    a.output.mkdir(exist_ok=False)
    raw=a.requests.read_bytes()
    if raw!=encoded():raise ValueError('Original frozen packet changed')
    packet=json.loads(raw)
    by_id={row['id']:(row,criteria) for row,criteria in zip(packet['requests'],packet['criteria'],strict=True)}
    order=['clean/energy','contamination/energy']*2
    write(a.output/'started.json',{'at_unix':time.time(),'deadline':a.deadline,'order':order,'original_packet_sha256':hashlib.sha256(raw).hexdigest(),'fresh_process':True})
    try:
        runtime=load_qwen36()
        write(a.output/'runtime.json',runtime.audit)
        validate(runtime,parse_calls,a.output,'qwen')
        for i,identity in enumerate(order):
            request,criteria=by_id[identity]
            result=runtime.generate(**{key:value for key,value in request.items() if key!='id'},thinking=False,reasoning_effort='medium')
            result.update(id=identity,repeat_index=i//2+1)
            parsed=parse_calls(result['completion'],request['tools'])
            result.update(parsed=parsed);result.update(assess(parsed,result,criteria))
            write(a.output/f'response-{i:03d}.json',result)
            print(json.dumps({'phase':'replication-response','id':identity,'repeat':i//2+1,'finish':result['finish'],'at_unix':time.time()}),flush=True)
        write(a.output/'complete.json',{'responses':4,'unique_inputs':2,'at_unix':time.time(),'tools_executed':False})
    except BaseException as error:
        write(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'at_unix':time.time()});raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
