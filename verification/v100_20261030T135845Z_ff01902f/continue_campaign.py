"""Dependency-ordered plateau continuation; preserves existing checkpoint/artifact bytes."""
import json,pathlib,subprocess,time
from common import PREFIX,ROOT,write_json,journal

def available():
    runs=[]
    for path in sorted((PREFIX/'results').glob('*_train_*/training_results.json')):
        parent=path.parent;exitfile=parent/'exit.json'
        if not exitfile.exists() or json.loads(exitfile.read_text())['exit_code']!=0:continue
        d=json.loads(path.read_text());cfg=json.loads((parent/'config.json').read_text())['arguments']
        if d['steps']==1500 and d.get('tier')=='main' and not cfg.get('selection_only'):
            runs.append((parent,d,cfg))
    return runs

def main():
    root=ROOT/'artifacts'/('continuation_'+str(time.time_ns()));root.mkdir(exist_ok=False)
    write_json(PREFIX/'configs'/('continuation_'+root.name+'.json'),{'endpoint_steps':6000,'criterion':'if any common-recipe structured arm improves selection validation NLL by >.02 from step1200 to1500, continue all paired common and function-matched arms to the equal-token 6000 endpoint','parameters':'all hyperparameters and batch table unchanged; no test-based tuning','expected_runs':27,'cost_estimate':'600-step pilots take about22 training seconds; 27 x4500 remaining updates about75 GPU minutes plus validation/checkpoint/export/reload; below24h ceiling','wait_limit_seconds':14400})
    start=time.monotonic()
    while True:
        runs=available()
        if len(runs)==27:break
        events=[json.loads(s) for s in (PREFIX/'events.jsonl').read_text().splitlines()]
        failures=[e for e in events if e.get('event')=='campaign_job_exit' and e.get('returncode')]
        if failures:raise RuntimeError('prior campaign failed; continuation requires fixing that run before proceeding')
        if time.monotonic()-start>14400:raise TimeoutError('main campaign did not complete within declared wait bound')
        time.sleep(10)
    needs=[]
    for parent,d,cfg in runs:
        curves={r['step']:r for r in d['curve']}
        if cfg.get('function_matched') or d['kind'] not in ['row_tail','native']:continue
        delta=curves[1200]['validation_selection_64_blocks']['nll']-curves[1500]['validation_selection_64_blocks']['nll']
        if delta>.02:needs.append({'kind':d['kind'],'seed':d['seed'],'improvement':delta})
    write_json(root/'decision.json',{'triggered_by_validation_only':needs,'continue':bool(needs)})
    if not needs:journal('continuation_not_needed',reason='predeclared validation improvement criterion not met');return
    journal('continuation_started',trigger=needs,endpoint_steps=6000)
    for index,(parent,d,cfg) in enumerate(runs):
        checkpoint=json.loads((parent/'checkpoint_001500.json').read_text())['path']
        cmd=['bash',str(PREFIX/'run.sh'),'--stage','train','--arm',d['kind'],'--seed',str(d['seed']),'--lr',str(cfg['lr']),'--steps','6000','--checkpoint',checkpoint]
        if cfg.get('function_matched'):cmd+=['--function-matched']
        with (root/f'job_{index:03d}.stdout').open('x') as out,(root/f'job_{index:03d}.stderr').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
        write_json(root/f'job_{index:03d}.json',{'command':cmd,'returncode':r.returncode})
        journal('continuation_job_exit',index=index,kind=d['kind'],seed=d['seed'],returncode=r.returncode)
        print(index,d['kind'],d['seed'],r.returncode,flush=True)
        if r.returncode:raise RuntimeError('continuation failed; inspect fresh result error, preserve source identity')
    journal('continuation_complete',endpoint_steps=6000)

if __name__=='__main__':main()
