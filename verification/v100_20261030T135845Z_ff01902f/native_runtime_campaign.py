"""Wait for selected factor exports, then collect the native full-model runtime."""
import json,subprocess,time
from common import ROOT,PREFIX,write_json,journal

def main():
    directory=ROOT/'artifacts'/('native_runtime_campaign_'+str(time.time_ns()));directory.mkdir(exist_ok=False);start=time.monotonic()
    while True:
        events=[json.loads(s) for s in (PREFIX/'events.jsonl').read_text().splitlines()]
        if any(e.get('event')=='precision_export_campaign_complete' for e in events):break
        if time.monotonic()-start>28800:raise TimeoutError('factor-export wait exceeded8h')
        time.sleep(10)
    cmd=['bash',str(PREFIX/'run.sh'),'--stage','native-runtime']
    with (directory/'job.stdout').open('x') as out,(directory/'job.stderr').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
    write_json(directory/'job.json',{'command':cmd,'returncode':r.returncode});journal('native_runtime_campaign_complete',returncode=r.returncode)
    if r.returncode:raise RuntimeError('native runtime failed; inspect new stage evidence')
if __name__=='__main__':main()
