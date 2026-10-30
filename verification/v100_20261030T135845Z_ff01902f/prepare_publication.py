"""Choose/stage only additions in this prefix; preserve local-only artifacts untouched."""
import json,pathlib,subprocess,time
from common import PREFIX,ROOT,digest,write_json,journal

def main():
    base=json.loads((PREFIX/'source_inventory.json').read_text())['base_sha'];repo=ROOT/'repo';prefix=str(PREFIX.relative_to(repo));result=subprocess.run(['git','cat-file','-e',base+':'+prefix],cwd=repo,capture_output=True);assert result.returncode!=0,'prefix existed at base'
    completed_at_start={d.name for d in (PREFIX/'results').iterdir() if (d/'exit.json').exists() and (d/'artifact_hashes.json').exists()}
    selected=[];excluded=[]
    for path in sorted(PREFIX.rglob('*')):
        if not path.is_file():continue
        reason=None
        relative=path.relative_to(PREFIX)
        if len(relative.parts)>=3 and relative.parts[0]=='results' and relative.parts[1] not in completed_at_start:reason='active/waiting or interrupted stage; retain locally and publish after collected exit'
        elif path.suffix in ['.pt','.pdf','.safetensors','.arrow','.parquet','.bin']:reason='local-only tensor/data/history fixture'
        elif path.name.startswith('profile_') and path.suffix=='.json':reason='full profiler trace retained locally; hash/summary published'
        elif path.stat().st_size>5_000_000:reason='large local-only artifact'
        elif path.suffix=='.json':
            try:json.loads(path.read_text())
            except (json.JSONDecodeError,UnicodeDecodeError):reason='invalid partial JSON from failed run; error/exit evidence retained'
        if reason:excluded.append({'path':str(path.relative_to(PREFIX)),'sha256':digest(path),'bytes':path.stat().st_size,'reason':reason})
        else:selected.append(str(path.relative_to(repo)))
    manifest=PREFIX/'reports'/('publication_exclusions_'+str(time.time_ns())+'.json');write_json(manifest,{'excluded':excluded,'policy':'source/concise raw metrics/reports only; exclusions never deleted/moved/modified'});selected.append(str(manifest.relative_to(repo)));journal('publication_stage_prepared',files=len(selected),excluded=len(excluded))
    # Include the final append-only journal version after recording preparation.
    pathspec=ROOT/'audit'/('stage_paths_'+str(time.time_ns())+'.nul')
    with pathspec.open('xb') as f:f.write(b'\0'.join(s.encode() for s in selected)+b'\0')
    subprocess.run(['git','add','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'],cwd=repo,check=True)
    files=subprocess.check_output(['git','diff','--cached',base,'--name-status'],cwd=repo,text=True);assert all(line.startswith('A\t'+prefix+'/') for line in files.splitlines());write_json(ROOT/'audit'/('staged_review_'+str(time.time_ns())+'.json'),{'files':files,'count':len(selected),'excluded_count':len(excluded),'expected_tree':subprocess.check_output(['git','write-tree'],cwd=repo,text=True).strip()});print(json.dumps({'staged':len(selected),'excluded':len(excluded),'manifest':str(manifest)}))
if __name__=='__main__':main()
