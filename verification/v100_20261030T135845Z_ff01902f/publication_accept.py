"""Verify published objects and fast-forward clone bookkeeping without a worktree checkout."""
import argparse,json,os,re,subprocess,time
from common import ROOT,write_json,journal

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT/'repo',text=True).strip()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--new',required=True);ap.add_argument('--tree',required=True);ap.add_argument('--result',required=True);ap.add_argument('--no-journal',action='store_true');a=ap.parse_args()
    assert all(re.fullmatch('[0-9a-f]{40}',value) for value in [a.old,a.new,a.tree]);assert git('rev-parse','HEAD')==a.old
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0');r=subprocess.run(['git','fetch','origin','verification/v100-20261030T135845Z-ff01902f'],cwd=ROOT/'repo',env=env,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert git('rev-parse','FETCH_HEAD')==a.new and git('rev-parse',a.new+'^')==a.old
    assert git('rev-parse',a.new+'^{tree}')==a.tree==git('write-tree')
    subprocess.run(['git','merge-base','--is-ancestor',a.old,a.new],cwd=ROOT/'repo',check=True)
    subprocess.run(['git','update-ref','refs/heads/verification/v100-20261030T135845Z-ff01902f',a.new,a.old],cwd=ROOT/'repo',check=True)
    record={'parent':a.old,'commit':a.new,'tree':a.tree,'stage':a.result,'PR':'https://github.com/NoahMadison/RPM-IHOG/pull/1','remote_matches_staged_tree':True,'local_fast_forward':'expectedoldref update afterfetch,tree,parent,ancestry checks; no worktree file checkout','fetch_stdout':r.stdout,'fetch_stderr':r.stderr}
    write_json(ROOT/'audit'/('publication_receipt_'+str(time.time_ns())+'.json'),record)
    if not a.no_journal:journal('published_stage',commit=a.new,result=a.result)
    print(json.dumps({'published':a.new,'stage':a.result}))
if __name__=='__main__':main()
