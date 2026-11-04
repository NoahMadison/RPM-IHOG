"""Wait for complete continuation, then select checkpoints using validation only."""
import json,pathlib,subprocess,time
from common import ROOT,PREFIX,write_json,journal

def main():
    directory=ROOT/'artifacts'/('selection_campaign_'+str(time.time_ns()));directory.mkdir(exist_ok=False);start=time.monotonic()
    while True:
        events=[json.loads(line) for line in (PREFIX/'events.jsonl').read_text().splitlines()]
        if any(e.get('event')=='continuation_complete' for e in events):break
        if any(e.get('event')=='continuation_job_exit' and e.get('returncode') for e in events):raise RuntimeError('continuation failed; cannot silently select incomplete coverage')
        if any(e.get('event')=='continuation_not_needed' for e in events):journal('selection_campaign_not_needed',reason='continuation criterion not met; preserve1500 results');return
        if time.monotonic()-start>21600:raise TimeoutError('continuation wait exceeded6h')
        time.sleep(10)
    paths=[]
    for path in sorted((PREFIX/'results').glob('*_train_*/training_results.json')):
        d=json.loads(path.read_text())
        if d['steps']==6000 and (path.parent/'exit.json').exists():paths.append(path)
    assert len(paths)==27,'expected complete paired common and function-matched6000 endpoints'
    for index,path in enumerate(paths):
        cmd=['bash',str(PREFIX/'run.sh'),'--stage','train-select','--checkpoint',str(path)]
        with (directory/f'job_{index:03d}.stdout').open('x') as out,(directory/f'job_{index:03d}.stderr').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
        write_json(directory/f'job_{index:03d}.json',{'command':cmd,'returncode':r.returncode});journal('selection_job_exit',index=index,returncode=r.returncode);print(index,r.returncode,flush=True)
        if r.returncode:raise RuntimeError('checkpoint selection job failed')
    journal('selection_campaign_complete')
if __name__=='__main__':main()
