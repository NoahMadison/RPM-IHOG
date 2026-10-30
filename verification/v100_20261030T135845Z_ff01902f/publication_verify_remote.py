"""Fresh read-only proof of remote parent, direct ancestry and reviewed tree before ref update."""
import argparse,json,os,re,subprocess,time,urllib.request
from common import ROOT,write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);ap.add_argument('--new',required=True);ap.add_argument('--tree',required=True);a=ap.parse_args();assert all(re.fullmatch('[0-9a-f]{40}',v) for v in [a.parent,a.new,a.tree]);repo=ROOT/'repo'
    remote=subprocess.check_output(['git','ls-remote','--heads','origin','refs/heads/verification/v100-20261030T135845Z-ff01902f'],cwd=repo,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),text=True).split()[0];assert remote==a.parent,'remote advanced; do not rewind or force'
    url='https://api.github.com/repos/NoahMadison/RPM-IHOG/git/commits/'+a.new
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept':'application/vnd.github+json'}),timeout=30) as r:d=json.load(r)
    parents=[p['sha'] for p in d['parents']];assert parents==[remote] and d['sha']==a.new and d['tree']['sha']==a.tree
    index=subprocess.check_output(['git','write-tree'],cwd=repo,text=True).strip();assert index==a.tree
    record={'observed_remote_head':remote,'new_commit':a.new,'actual_new_commit_parents':parents,'actual_new_tree':d['tree']['sha'],'reviewed_index_tree':index,'safe_direct_fast_forward':True,'force':False};write_json(ROOT/'audit'/('remote_parent_proof_'+str(time.time_ns())+'.json'),record);print(json.dumps(record))
if __name__=='__main__':main()
