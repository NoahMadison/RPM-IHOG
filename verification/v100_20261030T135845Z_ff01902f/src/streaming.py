"""Strict immutable-revision safetensors range reader and bounded archive checks."""
import hashlib,http.server,json,math,pathlib,re,resource,struct,threading,time,urllib.error,urllib.request
import torch
from common import ROOT,PREFIX,write_json,digest
from math_codec import encode,decode,to_device

class RangeReader:
    def __init__(self,url):self.url=url;self.downloaded=0;self.requests=[];self.peak_buffer=0;self.total=None;self.etag=None
    def read(self,start,end):
        if start<0 or end<start:raise ValueError('invalid requested range')
        for attempt in range(3):
            req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'RPM-IHOG-verification'})
            try:
                with urllib.request.urlopen(req,timeout=90) as response:
                    if response.status!=206:raise ValueError('server ignored byte range: HTTP '+str(response.status))
                    cr=response.headers.get('Content-Range','');match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                    if not match or (int(match[1]),int(match[2]))!=(start,end) or int(match[3])<=end:raise ValueError('mismatched Content-Range '+cr)
                    total=int(match[3]);etag=response.headers.get('ETag')
                    if self.total is not None and total!=self.total:raise ValueError('source size changed between ranges')
                    if self.etag is not None and etag!=self.etag:raise ValueError('source ETag changed or disappeared between ranges')
                    self.total=total
                    if etag is not None:self.etag=etag
                    body=response.read(end-start+2);self.downloaded+=len(body);self.peak_buffer=max(self.peak_buffer,len(body))
                    if len(body)!=end-start+1:raise ValueError('truncated or oversized range')
                    self.requests.append({'start':start,'end':end,'bytes':len(body),'attempt':attempt,'content_range':cr,'etag':etag,'sha256':hashlib.sha256(body).hexdigest()});return body
            except (urllib.error.URLError,TimeoutError):
                if attempt==2:raise
        raise RuntimeError('unreachable')
    def header(self):
        n=struct.unpack('<Q',self.read(0,7))[0]
        if n>16*1024*1024:raise ValueError('unbounded safetensors header')
        header=json.loads(self.read(8,7+n));return header,8+n

def remote_samples(artifact):
    from huggingface_hub import hf_hub_download
    ids=json.loads(sorted((PREFIX/'results').glob('*_data_*/model_identities.json'))[-1].read_text());identity=ids[0];mid=identity['id'];rev=identity['revision']
    index=json.loads(pathlib.Path(hf_hub_download(mid,'model.safetensors.index.json',revision=rev)).read_text());mapping=index['weight_map'];rows=[];headers={}
    for layer in [0,12,23]:
        keys=[k for k in mapping if f'layers.{layer}.' in k and any(x in k for x in ['input_linear.weight','output_linear.weight','gate_up_proj','down_proj'])]
        for key in keys:
            shard=mapping[key]
            if shard not in headers:
                reader=RangeReader(f'https://huggingface.co/{mid}/resolve/{rev}/{shard}');h,base=reader.header();headers[shard]=(reader,h,base)
            reader,h,base=headers[shard];entry=h[key];shape=entry['shape'];dtype=entry['dtype']
            if len(shape)!=3 or dtype not in ['BF16','F16']:raise ValueError('unexpected expert bank '+str(entry))
            expert=(layer*13+7)%shape[0];size=math.prod(shape[1:])*2;start=base+entry['data_offsets'][0]+expert*size
            body=reader.read(start,start+size-1);w=torch.frombuffer(bytearray(body),dtype=torch.bfloat16 if dtype=='BF16' else torch.float16).reshape(shape[1:]).float().to('cuda:0')
            p=encode(w,'ternary');path=artifact/f'remote_L{layer}_{len(rows):03d}.pt'
            with path.open('xb') as f:torch.save(to_device(p,'cpu'),f)
            rec=decode(p);mse=float((rec-w).square().mean());signal=float(w.square().mean())
            rows.append({'model':mid,'revision':rev,'tensor':key,'bank_shape':shape,'complete_expert':expert,'expert_shape':shape[1:],'source_range':[start,start+size-1],'source_bytes':size,'source_sha256':hashlib.sha256(body).hexdigest(),'archive_bytes':path.stat().st_size,'archive_sha256':digest(path),'sqnr_db':10*math.log10(signal/max(mse,1e-30)),'scope':'complete expert matrix sample, not full archive/model'})
    return {'samples':rows,'downloaded_bytes':sum(r.downloaded for r,_,_ in headers.values()),'requests':[req for r,_,_ in headers.values() for req in r.requests],'max_source_bytes_buffered':max(r.peak_buffer for r,_,_ in headers.values())}

def main(out,args):
    from safetensors.torch import save_file
    artifact=ROOT/'artifacts'/('stream_'+out.name);artifact.mkdir(exist_ok=False);results=[]
    # Manageable complete archive fixture: includes every tensor and raw fallback for vectors.
    for count in [1,16,64]:
        tensors={f'expert_{n:03d}':torch.randn(256,512,dtype=torch.float16,generator=torch.Generator().manual_seed(n+821)) for n in range(count)}
        tensors['norm']=torch.arange(128,dtype=torch.float16);source=artifact/f'fixture_{count:03d}.safetensors';save_file(tensors,str(source));del tensors
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        identity={'source_sha256':digest(source),'codec':'ternary','count':count};archive=artifact/f'archive_{count:03d}';archive.mkdir(exist_ok=False)
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_GET(self):
                match=re.fullmatch(r'bytes=(\d+)-(\d+)',self.headers.get('Range',''))
                if not match:self.send_error(416);return
                start,end=map(int,match.groups());self.send_response(206);self.send_header('Content-Range',f'bytes {start}-{end}/{source.stat().st_size}');self.end_headers()
                with source.open('rb') as f:f.seek(start);self.wfile.write(f.read(end-start+1))
        server=http.server.HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        reader=RangeReader(f'http://127.0.0.1:{server.server_port}/fixture');h,base=reader.header();manifest={'identity':identity,'tensors':[]};write_json(archive/'identity.json',identity)
        names=sorted(k for k in h if k!='__metadata__');reused=0
        try:
            # Simulate interruption after one item; restart validates identity and existing artifact hash.
            for pass_number,subset in enumerate([names[:1],names]):
                if json.loads((archive/'identity.json').read_text())!=identity:raise ValueError('changed source on resume')
                for key in subset:
                    path=archive/(key+'.pt');entry=h[key];a,b=entry['data_offsets'];shape=entry['shape']
                    if path.exists():
                        old=next(r for r in manifest['tensors'] if r['name']==key)
                        assert digest(path)==old['sha256'];reused+=1;continue
                    body=reader.read(base+a,base+b-1);w=torch.frombuffer(bytearray(body),dtype=torch.float16).clone().reshape(shape)
                    package=to_device(encode(w.to('cuda:0'),'ternary'),'cpu') if len(shape)==2 else {'raw':w}
                    with path.open('xb') as f:torch.save(package,f)
                    restored=decode(package) if 'raw' not in package else package['raw'];err=float((restored-w).float().square().mean())
                    manifest['tensors'].append({'name':key,'shape':shape,'source_offsets':[a,b],'source_sha256':hashlib.sha256(body).hexdigest(),'path':path.name,'sha256':digest(path),'bytes':path.stat().st_size,'mse':err,'decoded_sha256':hashlib.sha256(restored.contiguous().numpy().tobytes()).hexdigest(),'raw_fallback':'raw' in package})
                    del w,restored,package
                write_json(archive/f'manifest_{pass_number:03d}.json',manifest)
            assert sorted(r['name'] for r in manifest['tensors'])==names
            # Independent process restores every archived tensor without source access.
            import subprocess,sys
            child=archive/'reload_check.json'
            subprocess.run([sys.executable,'-B','-c','import sys;from streaming import reload_archive;reload_archive(*sys.argv[1:])',str(archive),str(child)],check=True)
            results.append({'fixture_tensor_count':count+1,'source_bytes':source.stat().st_size,'downloaded_bytes':reader.downloaded,'peak_live_source_range_bytes':reader.peak_buffer,'archive_bytes_including_manifests':sum(p.stat().st_size for p in archive.iterdir()),'resumed_reused_artifacts':reused,'coverage_complete':True,'fresh_process_reload':json.loads(child.read_text()),'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gpu_peak_allocated':torch.cuda.max_memory_allocated()})
        finally:server.shutdown();server.server_close();thread.join()
    try:remote=remote_samples(artifact)
    except Exception as e:remote={'status':'blocked','error':type(e).__name__+': '+str(e)}
    write_json(out/'streaming.json',{'complete_fixture_archives':results,'real_tensor_streaming':remote,'scope':'bounded complete fixtures and sampled real complete expert matrices; no full large model archive'})

def reload_archive(directory,outpath):
    a=pathlib.Path(directory);m=json.loads((a/'manifest_001.json').read_text());names=[]
    for row in m['tensors']:
        p=a/row['path'];assert digest(p)==row['sha256'];obj=torch.load(p,weights_only=True);w=obj['raw'] if 'raw' in obj else decode(obj);assert list(w.shape)==row['shape'] and bool(w.isfinite().all())
        if 'decoded_sha256' in row:assert hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()==row['decoded_sha256']
        names.append(row['name'])
    write_json(outpath,{'verified':names,'source_consulted':False})
