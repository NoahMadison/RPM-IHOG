"""Export already-reviewed staged Git objects to the authorized GitHub connector.

No credentials are handled here. It reads the index, never a changing working file.
Large blobs use bounded text/base64 chunks; tree batches remain below400KB source.
"""
import argparse,base64,json,pathlib,subprocess
from common import ROOT,PREFIX
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT/'repo')
def entries():
    prefix=str(PREFIX.relative_to(ROOT/'repo'))+'/'
    rows=[]
    for entry in git('ls-files','--stage','-z','--',prefix).split(b'\0'):
        if not entry:continue
        head,path=entry.split(b'\t',1);mode,sha,stage=head.decode().split();path=path.decode();assert stage=='0';size=int(git('cat-file','-s',sha));rows.append({'path':path,'mode':mode,'sha':sha,'size':size,'encoding':'base64' if path.endswith(('.png','.nul','.gz')) else 'utf-8'})
    return rows
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--meta',action='store_true');ap.add_argument('--batch',type=int);ap.add_argument('--blob');ap.add_argument('--encoding',default='utf-8');ap.add_argument('--offset',type=int,default=0);ap.add_argument('--length',type=int,default=400000);a=ap.parse_args()
    rows=entries();batches=[];current=[];size=0;blobs=[]
    for row in rows:
        if row['encoding']!='utf-8' or row['size']>400000:blobs.append(row);continue
        if current and size+row['size']>400000:batches.append(current);current=[];size=0
        current.append(row);size+=row['size']
    if current:batches.append(current)
    if a.meta:print(json.dumps({'rows':rows,'batches':len(batches),'blobs':blobs,'expected_tree':git('write-tree').decode().strip()}));return
    if a.batch is not None:
        result=[{'path':r['path'],'mode':r['mode'],'type':'blob','content':git('cat-file','blob',r['sha']).decode('utf-8')} for r in batches[a.batch]];print(json.dumps(result));return
    if a.blob:
        assert any(r['sha']==a.blob for r in rows),'only staged objects may be exported';body=git('cat-file','blob',a.blob);text=base64.b64encode(body).decode() if a.encoding=='base64' else body.decode('utf-8');part=text[a.offset:a.offset+a.length];print(json.dumps({'content':part,'eof':a.offset+len(part)>=len(text),'total_chars':len(text)}));return
    ap.error('select --meta, --batch or --blob')
if __name__=='__main__':main()
