"""Executed resume equivalence check, using only new selection-only pilot outputs."""
import json,pathlib,subprocess,time,torch
from common import ROOT,PREFIX,write_json,journal
out=ROOT/'audit'/('recovery_'+str(time.time_ns()));out.mkdir(exist_ok=False);runs=[]
for index,(steps,checkpoint) in enumerate([(5,None),(10,'previous'),(10,None)]):
    cmd=['bash',str(PREFIX/'run.sh'),'--stage','train','--arm','adamw','--seed','909','--steps',str(steps),'--selection-only']
    if checkpoint:cmd+=['--checkpoint',runs[0]['checkpoint']]
    response=subprocess.run(cmd,capture_output=True,text=True)
    write_json(out/f'job_{index}.json',{'command':cmd,'returncode':response.returncode,'stdout':response.stdout,'stderr':response.stderr})
    if response.returncode:raise RuntimeError(response.stderr+response.stdout)
    directory=pathlib.Path(json.loads(response.stdout.strip().splitlines()[-1])['result']);manifest=json.loads((directory/f'checkpoint_{steps:06d}.json').read_text());runs.append({'result':str(directory),'checkpoint':manifest['path']})
resumed=torch.load(runs[1]['checkpoint'],weights_only=True);continuous=torch.load(runs[2]['checkpoint'],weights_only=True)
equal={name:torch.equal(value,continuous['model'][name]) for name,value in resumed['model'].items()}
if not all(equal.values()):raise AssertionError('resume and uninterrupted model states differ: '+str([n for n,v in equal.items() if not v]))
result={'runs':runs,'model_state_bitwise_equal':True,'attempted_updates':10,'resume_at':5,'scope':'fixed real text main recipe; validation-only pipeline verification, not scientific training replication'}
write_json(PREFIX/'progress'/('resume_equivalence_'+str(time.time_ns())+'.json'),result);journal('resume_equivalence_passed',runs=runs)
print(json.dumps(result),flush=True)
