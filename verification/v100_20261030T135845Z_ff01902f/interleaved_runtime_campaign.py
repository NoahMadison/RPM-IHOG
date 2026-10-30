"""Interleaved timing after all source archives and their full fidelity checks exist."""
import json,subprocess,time
from common import ROOT,PREFIX,write_json,journal

def main():
    directory=ROOT/'artifacts'/('interleaved_campaign_'+str(time.time_ns()));directory.mkdir(exist_ok=False);start=time.monotonic()
    while True:
        events=[json.loads(s) for s in (PREFIX/'events.jsonl').read_text().splitlines()];done=[e for e in events if e.get('event')=='native_runtime_campaign_complete']
        if done:
            assert done[-1]['returncode']==0,'native runtime failed';break
        if time.monotonic()-start>28800:raise TimeoutError('native fidelity/export wait exceeded8h')
        time.sleep(10)
    cmd=['bash',str(PREFIX/'run.sh'),'--stage','interleaved-runtime']
    with (directory/'job.stdout').open('x') as out,(directory/'job.stderr').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
    write_json(directory/'job.json',{'command':cmd,'returncode':r.returncode});journal('interleaved_runtime_campaign_complete',returncode=r.returncode)
    if r.returncode:raise RuntimeError('interleaved fullruntime failed; inspect evidence')
if __name__=='__main__':main()
