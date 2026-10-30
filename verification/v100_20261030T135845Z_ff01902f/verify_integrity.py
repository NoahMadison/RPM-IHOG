"""Base-byte preservation, original-input hashes, additions-only and staged secret checks."""
import argparse,hashlib,json,pathlib,re,subprocess,time
from common import PREFIX,ROOT,write_json,digest,journal

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT/'repo')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--staged',action='store_true');ap.add_argument('--check-only',action='store_true');a=ap.parse_args();base=json.loads((PREFIX/'source_inventory.json').read_text())['base_sha'];prefix=str(PREFIX.relative_to(ROOT/'repo'))+'/'
    original={}
    for name in ['original_inputs.json','additional_inputs.json']:
        original.update(json.loads((ROOT/'audit'/name).read_text()))
    inputs={pathlib.Path(path).name:digest(path)==expected for path,expected in original.items()}
    expected=json.loads((PREFIX/'source_hashes.json').read_text());tracked={name:digest(ROOT/'repo'/name)==sha for name,sha in expected.items()}
    base_paths=[p.decode() for p in git('ls-tree','-r','--name-only','-z',base).split(b'\0') if p];assert set(base_paths)==set(expected)
    assert not (ROOT/'repo/.git/objects/info/alternates').exists(),'shared-clone alternates prohibited'
    assert all(inputs.values()),'an original local input changed'
    assert all(tracked.values()),'a base-commit tracked file changed'
    diff=git('diff','--name-status',base).decode();assert not diff or all(line.startswith('A\t'+prefix) for line in diff.splitlines()),diff
    staged=[];findings=[]
    if a.staged:
        records=git('diff','--cached',base,'--name-status','-z').split(b'\0');records=[r for r in records if r]
        assert len(records)%2==0
        for kind,name in zip(records[::2],records[1::2]):
            name=name.decode();assert kind==b'A' and name.startswith(prefix),(kind,name);staged.append(name)
            body=git('show',':'+name)
            if name.endswith(('.pt','.pdf','.safetensors','.arrow','.parquet','.bin')):findings.append({'file':name,'reason':'forbidden artifact extension'})
            for pattern in [rb'AKIA[0-9A-Z]{16}',rb'gh[pousr]_[A-Za-z0-9]{20,}',rb'hf_[A-Za-z0-9]{20,}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY',rb'https://gemini\.google\.com/app/']:
                if re.search(pattern,body):findings.append({'file':name,'reason':'credential/private-history pattern'})
            if len(body)>5_000_000:findings.append({'file':name,'reason':'large staged artifact requires exclusion'})
        assert staged,'nothing staged'
        assert not findings,findings
    result={'base_sha':base,'all_original_inputs_unchanged':all(inputs.values()),'original_input_checks':inputs,'all_base_commit_paths_unchanged_including_ignored':all(tracked.values()),'base_tracked_file_count':len(base_paths),'base_tracked_sha256':expected,'independent_clone_no_alternates':True,'additions_only_prefix':prefix,'git_diff_name_status':diff,'staged_files':staged,'credential_and_private_history_findings':findings,'checked_staged':a.staged}
    if a.check_only:
        print(json.dumps({'base_paths':len(tracked),'original_inputs':len(inputs),'staged':len(staged),'findings':findings,'tree':git('write-tree').decode().strip()}));return
    path=PREFIX/'reports'/('integrity_'+str(time.time_ns())+'.json');write_json(path,result);journal('integrity_verified',path=str(path.relative_to(PREFIX)),staged=a.staged);print(json.dumps({'report':str(path),'base_paths':len(tracked),'original_inputs':len(inputs),'staged':len(staged)}))

if __name__=='__main__':main()
