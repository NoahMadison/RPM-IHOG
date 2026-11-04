"""Complete FP16 dense/native/ordinary-factor storage baselines, with fidelity reloads."""
import json,pathlib,subprocess,time
from common import PREFIX,ROOT,write_json,journal
def main():
    directory=ROOT/'artifacts'/('precision_export_'+str(time.time_ns()));directory.mkdir(exist_ok=False);start=time.monotonic()
    while True:
        events=[json.loads(s) for s in (PREFIX/'events.jsonl').read_text().splitlines()]
        if any(e.get('event')=='factor_export_campaign_complete' for e in events):break
        if time.monotonic()-start>28800:raise TimeoutError('factor-export wait exceeded8h')
        time.sleep(10)
    selected={}
    for path in sorted((PREFIX/'results').glob('*_train-select_*/selected_training_results.json')):
        d=json.loads(path.read_text())
        if d['kind'] in ['adamw','native','lowrank'] and not d['function_matched']:selected[(d['kind'],d['seed'])]=path
    for index,((kind,seed),path) in enumerate(sorted(selected.items())):
        cmd=['bash',str(PREFIX/'run.sh'),'--stage','direct-factor-export','--checkpoint',str(path),'--codec','all_fp16']
        with (directory/f'job_{index:03d}.stdout').open('x') as out,(directory/f'job_{index:03d}.stderr').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
        write_json(directory/f'job_{index:03d}.json',{'command':cmd,'returncode':r.returncode});print(kind,seed,r.returncode,flush=True)
        if r.returncode:raise RuntimeError('complete FP16 precision baseline failed')
    journal('precision_export_campaign_complete',runs=len(selected))
if __name__=='__main__':main()
