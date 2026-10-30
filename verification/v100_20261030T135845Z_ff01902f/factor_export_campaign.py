import json,pathlib,subprocess,time
from common import PREFIX,ROOT,write_json,journal
def main():
    directory=ROOT/'artifacts'/('factor_export_campaign_'+str(time.time_ns()));directory.mkdir(exist_ok=False);start=time.monotonic()
    while True:
        events=[json.loads(x) for x in (PREFIX/'events.jsonl').read_text().splitlines()]
        if any(e.get('event')=='closecall_campaign_complete' for e in events):break
        if time.monotonic()-start>28800:raise TimeoutError('replication wait exceeded8h')
        time.sleep(10)
    selected={}
    for path in sorted((PREFIX/'results').glob('*_train-select_*/selected_training_results.json')):
        d=json.loads(path.read_text())
        if d['kind'] in ['native','lowrank','adamw'] and not d['function_matched']:selected[(d['kind'],d['seed'])]=path
    for index,((kind,seed),path) in enumerate(sorted(selected.items())):
        for variant in (['all_fp16'] if kind=='adamw' else ['expert_fp16_shared_fp32','all_fp16']):
            cmd=['bash',str(PREFIX/'run.sh'),'--stage','direct-factor-export','--checkpoint',str(path),'--codec',variant]
            with (directory/f'job_{index:03d}_{variant}.stdout').open('x') as out,(directory/f'job_{index:03d}_{variant}.stderr').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
            write_json(directory/f'job_{index:03d}_{variant}.json',{'command':cmd,'returncode':r.returncode});print(kind,seed,variant,r.returncode,flush=True)
            if r.returncode:raise RuntimeError('direct-factor matched-container export failed')
    journal('factor_export_campaign_complete',runs=len(selected))
if __name__=='__main__':main()
