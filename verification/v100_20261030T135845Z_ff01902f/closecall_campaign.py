"""Predeclared conditional5-seed replication, with conservative sequential decisions."""
import json,pathlib,subprocess,time,numpy as np
from scipy.stats import t
from common import PREFIX,ROOT,write_json,journal

def interval(a):
    a=np.asarray(a,float);mean=float(a.mean());half=float(t.ppf(.9875,len(a)-1)*a.std(ddof=1)/len(a)**.5);return [mean-half,mean+half]

def main():
    directory=ROOT/'artifacts'/('closecall_'+str(time.time_ns()));directory.mkdir(exist_ok=False);start=time.monotonic()
    while True:
        events=[json.loads(s) for s in (PREFIX/'events.jsonl').read_text().splitlines()]
        if any(e.get('event')=='selection_campaign_complete' for e in events):break
        if any(e.get('event') in ['selection_job_exit','continuation_job_exit','campaign_job_exit'] and e.get('returncode') for e in events):raise RuntimeError('prerequisite campaign failed')
        if any(e.get('event')=='continuation_not_needed' for e in events):journal('closecall_campaign_complete',expanded=[],reason='no long continuation; do not promote a plateau claim');return
        if time.monotonic()-start>25200:raise TimeoutError('prerequisite wait exceeded7h')
        time.sleep(10)
    selected={}
    for path in sorted((PREFIX/'results').glob('*_train-select_*/selected_training_results.json')):
        d=json.loads(path.read_text())
        if not d['function_matched']:selected[(d['kind'],d['seed'])]=d
    expand=[];decisions=[]
    for kind in ['row','tail','row_tail','native']:
        delta=[selected[(kind,s)]['baseline']['test']['nll']-selected[('adamw',s)]['baseline']['test']['nll'] for s in [11,22,33]];ci=interval(delta);close=ci[0]<=.02<=ci[1] or abs(float(np.mean(delta))-.02)<.01
        decisions.append({'kind':kind,'n':3,'paired_deltas':delta,'decision_ci97_5':ci,'expand':close})
        if close:expand.append(kind)
    write_json(directory/'decision.json',{'decisions':decisions,'extra_seeds':[44,55],'expanded':expand,'decision_alpha_spending':'two-sided97.5% intervals at n3 and n5; union-bound5% sequential error per contrast; descriptive95% also reported; no cross-method multiplicity claim'})
    journal('closecall_replication_decision',decisions=decisions,expanded=expand)
    for seed in [44,55]:
        for kind in (['adamw']+expand if expand else []):
            cmd=['bash',str(PREFIX/'run.sh'),'--stage','train','--arm',kind,'--seed',str(seed),'--steps','6000'];r=subprocess.run(cmd,capture_output=True,text=True);job=f'{seed}_{kind}';write_json(directory/(job+'_train.json'),{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
            if r.returncode:raise RuntimeError('extra replication failed')
            result=pathlib.Path(json.loads(r.stdout.strip().splitlines()[-1])['result'])/'training_results.json';cmd=['bash',str(PREFIX/'run.sh'),'--stage','train-select','--checkpoint',str(result)];r=subprocess.run(cmd,capture_output=True,text=True);write_json(directory/(job+'_select.json'),{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
            journal('closecall_job_exit',kind=kind,seed=seed,returncode=r.returncode);print(seed,kind,r.returncode,flush=True)
            if r.returncode:raise RuntimeError('extra replication selection failed')
    journal('closecall_campaign_complete',expanded=expand)
if __name__=='__main__':main()
