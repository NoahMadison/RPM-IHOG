"""Resolve only verified immutable small-model snapshots; no large-model weight download."""
import json,pathlib,time
from huggingface_hub import snapshot_download
from common import PREFIX,ROOT,write_json,journal
identities=json.loads(sorted((PREFIX/'results').glob('*_data_*/model_identities.json'))[-1].read_text())
out=ROOT/'audit'/('downloads_'+str(time.time_ns())+'.json');rows=[]
for identity in identities[:2]:
    start=time.perf_counter();path=snapshot_download(identity['id'],revision=identity['revision'],allow_patterns=['*.safetensors','*.json','*.txt','*.model'],max_workers=3)
    row={'id':identity['id'],'revision':identity['revision'],'path':path,'seconds':time.perf_counter()-start};rows.append(row);print(json.dumps(row),flush=True);journal('model_download',**row)
write_json(out,rows)
