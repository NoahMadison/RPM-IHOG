"""Read-only progress monitor. Does not infer completion from background processes."""
import argparse,collections,json,time
from common import PREFIX

def snapshot():
    counts=collections.Counter();pending=[]
    for path in sorted((PREFIX/'results').iterdir()):
        config=path/'config.json'
        if not config.exists():continue
        stage=json.loads(config.read_text())['arguments']['stage'];exitfile=path/'exit.json'
        if exitfile.exists():counts[(stage,json.loads(exitfile.read_text())['exit_code'])]+=1
        else:
            log=path/'stdout.log';tail=log.read_text().splitlines()[-1:] if log.exists() else []
            pending.append({'run':path.name,'last_output':tail,'worker_error':(path/'error.json').exists(),'note':'a missing exit file may represent an active/waiting worker or interrupted parent; inspect host processes'})
    events=[json.loads(s) for s in (PREFIX/'events.jsonl').read_text().splitlines()]
    milestones=[e for e in events if e.get('event') in ['campaign_complete','continuation_started','continuation_complete','selection_campaign_complete','closecall_replication_decision','closecall_campaign_complete','factor_export_campaign_complete','precision_export_campaign_complete']]
    return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'completed':{stage+':exit'+str(code):n for (stage,code),n in sorted(counts.items())},'pending':pending,'milestones':milestones[-4:]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=1);ap.add_argument('--interval',type=float,default=45);a=ap.parse_args()
    for n in range(a.samples):
        d=snapshot()
        if a.samples>1:d={'utc':d['utc'],'training_exits':{k:v for k,v in d['completed'].items() if k.startswith(('train:','train-select:','direct-factor-export:'))},'progress':[{'run':r['run'],'last_output':r['last_output']} for r in d['pending'] if not r['worker_error']],'milestones':d['milestones'][-1:]}
        print(json.dumps(d),flush=True)
        if n+1<a.samples:time.sleep(min(55,max(1,a.interval)))
if __name__=='__main__':main()
