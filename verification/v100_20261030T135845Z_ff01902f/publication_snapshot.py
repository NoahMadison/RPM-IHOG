"""Stage one completed result plus its progress/source; export exact reviewed Git objects."""
import argparse,base64,json,pathlib,subprocess,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'src'))
from common import PREFIX,ROOT,write_json,journal,digest

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT/'repo')
def eligible(path):
    if path.is_symlink():raise ValueError('unexpected symlink in publication source')
    if path.suffix in ['.pt','.pdf','.safetensors','.arrow','.parquet','.bin'] or path.stat().st_size>5_000_000:return False
    if path.name.startswith('profile_') and path.suffix=='.json':return False
    if path.suffix=='.json':
        try:json.loads(path.read_text())
        except (ValueError,UnicodeDecodeError):return False
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);a=ap.parse_args();repo=ROOT/'repo';prefix=str(PREFIX.relative_to(repo));assert git('rev-parse','HEAD').decode().strip()==a.parent,'local head differs from expected remote parent';subprocess.run(['git','diff','--cached','--quiet',a.parent],cwd=repo,check=True)
    published={pathlib.Path(row.decode()).parent.name for row in git('ls-tree','-r','--name-only','-z',a.parent,'--',prefix+'/results').split(b'\0') if row.endswith(b'/exit.json')}
    lines=(PREFIX/'events.jsonl').read_text().splitlines();events=[]
    for line in lines:
        try:events.append(json.loads(line))
        except json.JSONDecodeError:continue
    collected={e['result'].split('/')[-1] for e in events if e.get('event')=='stage_exit'}
    ready=[p for p in sorted((PREFIX/'results').iterdir()) if p.name not in published and p.name in collected and (p/'exit.json').exists() and (p/'artifact_hashes.json').exists()]
    if not ready:print(json.dumps({'noop':True,'reason':'no unpublished completed stage'}));return
    result=ready[0];exitdata=json.loads((result/'exit.json').read_text());cfg=json.loads((result/'config.json').read_text());stage=cfg['arguments']['stage'];summary={'stage':stage,'result':str(result.relative_to(PREFIX)),'exit':exitdata,'prior_publication_commit':a.parent,'PR':'https://github.com/NoahMadison/RPM-IHOG/pull/1','artifact_manifest_sha256':digest(result/'artifact_hashes.json'),'pending_stages':'see append-only events and latest recovery handoff; blocked multiGPU is not a scientific failure'}
    for name in ['training_results.json','selected_training_results.json','direct_factor_results.json']:
        path=result/name
        if path.exists():
            d=json.loads(path.read_text());summary['metrics']={key:d[key] for key in ['kind','seed','steps','selected_steps','function_matched','successful_updates','skipped_updates','tokens_consumed','training_seconds_including_periodic_validation_checkpointing','expert_bpw','whole_model_bpw','serialized_bytes','delta_nll'] if key in d}
            if 'baseline' in d:summary['metrics']['original_test_nll']=d['baseline'].get('test',{}).get('nll')
            summary['metrics']['raw_evidence']=str(path.relative_to(PREFIX))
    label=stage
    if summary.get('metrics'):
        m=summary['metrics'];label+=' '+str(m.get('kind',''))+' seed'+str(m.get('seed',''))
    message='Record '+label+' completed stage '+result.name.split('_')[0]
    progress=PREFIX/'progress'/('stage_'+result.name+'_'+str(time.time_ns())+'.json');write_json(progress,summary);journal('stage_publication_prepared',result=str(result.relative_to(PREFIX)),parent=a.parent,progress=str(progress.relative_to(PREFIX)))
    paths=[p for p in PREFIX.rglob('*') if p.is_file() and p.relative_to(PREFIX).parts[0]!='results' and eligible(p)];paths += [p for p in result.rglob('*') if p.is_file() and eligible(p)]
    pathspec=ROOT/'audit'/('incremental_stage_paths_'+str(time.time_ns())+'.nul')
    with pathspec.open('xb') as f:f.write(b'\0'.join(str(p.relative_to(repo)).encode() for p in paths)+b'\0')
    subprocess.run(['git','add','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'],cwd=repo,check=True)
    check=subprocess.run([sys.executable,'-B',str(PREFIX/'verify_integrity.py'),'--staged','--check-only','--parent',a.parent],capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
    changed={r.decode() for r in git('diff','--cached','--name-only','-z',a.parent,'--',prefix).split(b'\0') if r};rows=[]
    for raw in git('ls-files','--stage','-z','--',prefix).split(b'\0'):
        if not raw:continue
        head,path=raw.split(b'\t',1);path=path.decode()
        if path not in changed:continue
        mode,sha,index=head.decode().split();assert index=='0';rows.append({'path':path,'mode':mode,'sha':sha,'encoding':'base64' if path.endswith(('.png','.nul','.gz')) else 'utf-8'})
    directory=ROOT/'audit'/('stage_spool_'+str(time.time_ns()));directory.mkdir(exist_ok=False);bodies={};proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    for row in rows:
        sha=row['sha'];proc.stdin.write((sha+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().decode().split();assert header[0]==sha and header[1]=='blob';body=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';row['size']=len(body);bodies[sha]=body
    proc.stdin.close();assert proc.wait()==0
    batches=[];current=[];size=0;blobs=[]
    for row in rows:
        if row['encoding']!='utf-8' or row['size']>400000:blobs.append(row);continue
        if current and size+row['size']>400000:batches.append(current);current=[];size=0
        current.append(row);size+=row['size']
    if current:batches.append(current)
    for i,batch in enumerate(batches):write_json(directory/f'batch_{i:04d}.json',[{'path':r['path'],'mode':r['mode'],'type':'blob','content':bodies[r['sha']].decode()} for r in batch])
    for i,row in enumerate(blobs):
        content=base64.b64encode(bodies[row['sha']]).decode() if row['encoding']=='base64' else bodies[row['sha']].decode();row['chunks']=(len(content)+399999)//400000
        for j in range(row['chunks']):write_json(directory/f'blob_{i:04d}_{j:04d}.json',{'content':content[j*400000:(j+1)*400000]})
    meta={'parent':a.parent,'rows':rows,'batches':len(batches),'blobs':blobs,'expected_tree':git('write-tree').decode().strip(),'directory':str(directory),'stage':str(result.relative_to(PREFIX)),'message':message,'integrity':json.loads(check.stdout),'progress':str(progress.relative_to(PREFIX))};write_json(directory/'metadata.json',meta);print(json.dumps(meta))
if __name__=='__main__':main()
