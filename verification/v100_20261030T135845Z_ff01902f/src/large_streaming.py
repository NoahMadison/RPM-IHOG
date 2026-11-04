"""Bounded real complete matrices from verified large sources; never a full-model claim."""
import hashlib,json,math,pathlib,resource,torch
from common import PREFIX,ROOT,write_json,digest
from streaming import RangeReader
from math_codec import encode,decode,to_device,tensor_bytes

def main(out,args):
    from huggingface_hub import hf_hub_download
    identities=json.loads(sorted((PREFIX/'results').glob('*_data_*/model_identities.json'))[-1].read_text());artifact=ROOT/'artifacts'/('large_stream_'+out.name);artifact.mkdir(exist_ok=False);rows=[];requests=[]
    for identity in identities[2:]:
        if identity['status']!='available':rows.append({'model':identity['id'],'status':'blocked','reason':'source unavailable at audited identity'});continue
        name,revision=identity['id'],identity['revision'];headers={}
        try:
            index=json.loads(pathlib.Path(hf_hub_download(name,'model.safetensors.index.json',revision=revision)).read_text());mapping=index['weight_map'];moe='Flash' in name
            selectors=[(0,'gate'),(24,'up'),(47,'down')] if moe else [(4,'gate'),(28,'up'),(52,'down')]
            for layer,projection in selectors:
                keys=sorted(k for k in mapping if f'layers.{layer}.' in k and projection in k and ('experts' in k if moe else '.mlp.' in k) and k.endswith('weight'))
                if not keys:keys=sorted(k for k in mapping if f'layers.{layer}.' in k and projection in k and ('experts' in k if moe else '.mlp.' in k))
                if not keys:raise ValueError('no verified complete projection key: '+str((layer,projection)))
                desired=(layer*13+7)%512;key=next((k for k in keys if f'.experts.{desired}.' in k),keys[0]);shard=mapping[key]
                if shard not in headers:
                    reader=RangeReader(f'https://huggingface.co/{name}/resolve/{revision}/{shard}');header,base=reader.header();headers[shard]=(reader,header,base)
                reader,header,base=headers[shard];entry=header[key];shape=entry['shape'];dtype=entry['dtype'];types={'BF16':(torch.bfloat16,2),'F16':(torch.float16,2),'F32':(torch.float32,4)}
                if dtype not in types:raise ValueError('unsupported source dtype '+dtype)
                dtype_,width=types[dtype];start,end=entry['data_offsets'];assert end-start==math.prod(shape)*width
                if len(shape)==3:
                    expert=desired%shape[0];matrix_shape=shape[1:];size=math.prod(matrix_shape)*width;start+=expert*size;end=start+size
                elif len(shape)==2:expert=desired if f'.experts.{desired}.' in key else None;matrix_shape=shape
                else:raise ValueError('not a complete matrix/expert bank '+str(shape))
                torch.cuda.reset_peak_memory_stats();body=reader.read(base+start,base+end-1);source_sha=hashlib.sha256(body).hexdigest()
                # Read-only buffer view; conversion allocates only this complete matrix, never a shard/model.
                matrix=torch.frombuffer(body,dtype=dtype_).reshape(matrix_shape).to(device='cuda:0',dtype=torch.float32)
                payload=encode(matrix,'int2_sign');reconstructed=decode(payload);mse=float((reconstructed-matrix).square().mean());power=float(matrix.square().mean());path=artifact/f'sample_{len(rows):03d}.pt'
                with path.open('xb') as f:torch.save(to_device(payload,'cpu'),f)
                row={'model':name,'revision':revision,'status':'completed','source_tensor':key,'bank_or_tensor_shape':shape,'complete_matrix_shape':matrix_shape,'expert_id':expert,'projection_selector':projection,'source_dtype':dtype,'source_range_inclusive':[base+start,base+end-1],'source_bytes':len(body),'source_sha256':source_sha,'codec':'int2_sign','tensor_payload_bytes':tensor_bytes(payload),'tensor_payload_bpw':8*tensor_bytes(payload)/matrix.numel(),'serialized_bytes':path.stat().st_size,'archive_sha256':digest(path),'weight_sqnr_db':10*math.log10(power/max(mse,1e-30)),'gpu_peak_allocated':torch.cuda.max_memory_allocated(),'gpu_peak_reserved':torch.cuda.max_memory_reserved(),'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'actual complete dense projection or expert matrix; no full language-model quality/archive/residency','max_live_HTTP_response_bytes':reader.peak_buffer,'conversion_memory_note':'source buffer plus per-matrix dtype/device conversions and SVD/codec workspace included in RSS/GPU peaks; range bytes are not total process memory'}
                rows.append(row);write_json(out/f'sample_{len(rows):03d}.json',row);print(name,key,matrix_shape,row['weight_sqnr_db'],flush=True)
                del body,matrix,payload,reconstructed;torch.cuda.empty_cache()
        except Exception as e:
            rows.append({'model':name,'revision':revision,'status':'blocked','reason':type(e).__name__+': '+str(e),'interpretation':'source/interface/streaming blocker, not compression-hypothesis falsification'})
        requests.extend({'model':name,'revision':revision,**request} for reader,_,_ in headers.values() for request in reader.requests)
    write_json(out/'large_streaming.json',{'samples':rows,'HTTP_range_requests':requests,'actual_downloaded_bytes':sum(r['bytes'] for r in requests),'full_large_model_archive_status':'not_run','large_model_full_quality_status':'not_run','large_model_residency_status':'not_run'})
