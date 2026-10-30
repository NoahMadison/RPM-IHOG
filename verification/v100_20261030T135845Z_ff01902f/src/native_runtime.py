"""Full learned native-factor model: exact-payload FP32 reference versus direct core."""
import gc,hashlib,json,pathlib,resource,subprocess,sys,time,math
import torch
import torch.nn.functional as F
from torch import nn
from common import ROOT,PREFIX,write_json,digest,gpu_guard,synchronize,paired_ci,block_bootstrap
from models import Model
from math_codec import materialize,kron_forward,tensor_bytes
from data_audit import latest_data

class FactorMoE(nn.Module):
    def __init__(self,router,state,layer):
        super().__init__();self.router=router
        for direction in ['up','down']:
            for letter in ['a','b','row']:
                self.register_buffer(letter+'_'+direction,state[f'blocks.{layer}.moe.{letter}_{direction}'].float())
    def project(self,direction,expert,x):
        return kron_forward(x,getattr(self,'a_'+direction)[expert],getattr(self,'b_'+direction)[expert],getattr(self,'row_'+direction)[expert])
    def forward(self,x):
        shape=x.shape;flat=x.reshape(-1,256);prob=F.softmax(self.router(flat).float(),-1);weights,indices=prob.topk(2,-1);weights/=weights.sum(-1,keepdim=True);out=torch.zeros_like(flat)
        for expert in range(16):
            idx,slot=(indices==expert).nonzero(as_tuple=True);inputs=flat[idx]
            hidden=F.silu(self.project('up',expert,inputs))
            result=self.project('down',expert,hidden);out.index_add_(0,idx,(result*weights[idx,slot,None]).to(out.dtype))
        return out.reshape(shape)

class DenseMoE(FactorMoE):
    def __init__(self,router,state,layer):
        nn.Module.__init__(self);self.router=router
        for direction in ['up','down']:
            a,b,row=[state[f'blocks.{layer}.moe.{name}_{direction}'].float() for name in ['a','b','row']]
            self.register_buffer('w_'+direction,materialize(a,b)*row[:,:,None])
    def project(self,direction,expert,x):return F.linear(x,getattr(self,'w_'+direction)[expert])

def load_model(mode,archive,checkpoint):
    if mode=='original_fp32':
        cp=torch.load(checkpoint,weights_only=True,map_location='cpu');assert cp['kind']=='native';state=cp['model'];shared={n:v for n,v in state.items() if '.moe.' not in n or '.router.' in n}
    else:
        package=torch.load(archive,weights_only=True,map_location='cpu');assert package['kind']=='native';shared=package['shared'];state=package['factors'];assert all(v.dtype==torch.float16 for v in list(shared.values())+list(state.values()) if v.is_floating_point())
    source_model_tensor_bytes=tensor_bytes(cp['model']) if mode=='original_fp32' else tensor_bytes(package)
    model=Model('adamw');model.load_state_dict(shared,strict=False)
    for layer,block in enumerate(model.blocks):
        if mode.startswith('direct_factors'):block.moe=FactorMoE(block.moe.router,state,layer)
        else:block.moe=DenseMoE(block.moe.router,state,layer)
    if mode.endswith('_fp16'):model.half()
    model.to('cuda:0').eval()
    if mode.startswith('direct_factors'):assert not any('.moe.w_' in n for n,p in list(model.named_parameters())+list(model.named_buffers()))
    return model,source_model_tensor_bytes

@torch.no_grad()
def evaluate_fp32(model,tokens):
    sums=[];counts=[];seq=128;n=(len(tokens)-1)//seq
    for start in range(0,n,16):
        number=min(16,n-start);idx=torch.arange(start,start+number,device=tokens.device)[:,None]*seq+torch.arange(seq+1,device=tokens.device);chunk=tokens[idx];logits=model(chunk[:,:-1]).float();loss=F.cross_entropy(logits.flatten(0,1),chunk[:,1:].flatten(),reduction='none').reshape(number,seq).sum(-1);sums.extend(loss.cpu().tolist());counts.extend([seq]*number)
    return {'nll':sum(sums)/sum(counts),'ppl':math.exp(sum(sums)/sum(counts)),'block_nll':[s/c for s,c in zip(sums,counts)],'counts':counts,'tokens_scored':sum(counts),'dropped_final_tail':len(tokens)-1-n*seq,'precision':str(next(model.parameters()).dtype)+' parameters/contractions; FP32 NLL accumulation, no autocast'}

@torch.no_grad()
def generation(model,prompt,continuation,forced):
    ids=prompt;generated=[];synchronize();start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    first=None
    for step in range(16):
        prediction=model(ids)[:,-1].argmax(-1,keepdim=True);generated.append(prediction)
        if step==0:synchronize();first=time.perf_counter()-start
        incoming=continuation[:,step:step+1] if forced else prediction;ids=torch.cat((ids,incoming),-1)
    b.record();b.synchronize();elapsed=time.perf_counter()-start
    return {'prompt_length':prompt.shape[1],'teacher_forced':forced,'generated_tokens':16,'prefill_first_token_gpu_ready_host_seconds':first,'host_prefill_and_generation_seconds':elapsed,'device_prefill_and_generation_ms':a.elapsed_time(b),'decode_tokens_per_second':15/(elapsed-first),'generated_sha256':hashlib.sha256(torch.cat(generated,-1).cpu().numpy().tobytes()).hexdigest()}

def child(mode,archive,checkpoint,tokenroot,outpath,artifact):
    gpu_guard();torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();model,source_bytes=load_model(mode,archive,checkpoint);gc.collect();synchronize();load_seconds=time.perf_counter()-begin;load_peak=torch.cuda.max_memory_allocated();steady=torch.cuda.memory_allocated();resident=tensor_bytes([dict(model.named_parameters()),dict(model.named_buffers())]);val=torch.load(pathlib.Path(tokenroot)/'validation.pt',weights_only=True).to('cuda:0');test=torch.load(pathlib.Path(tokenroot)/'test.pt',weights_only=True).to('cuda:0');samples=[]
    for length in [32,96]:
        prompt=val[:length].reshape(1,-1);continuation=val[length:length+16].reshape(1,-1)
        for forced in [False,True]:
            generation(model,prompt,continuation,forced)
            for group in range(20):row=generation(model,prompt,continuation,forced);row['group']=group;samples.append(row)
    quality=evaluate_fp32(model,test)
    profile=None
    if mode.startswith('direct_factors'):
        trace=pathlib.Path(artifact)/(mode+'_trace.json')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True,record_shapes=True) as prof:model(val[:32].reshape(1,-1));synchronize()
        prof.export_chrome_trace(str(trace));profile={'trace_sha256':digest(trace),'local_trace':str(trace),'bytes':trace.stat().st_size,'summary':prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30)}
    write_json(outpath,{'mode':mode,'source_checkpoint_loaded':mode=='original_fp32','original_weights_used_by_decoder':False if mode!='original_fp32' else None,'dense_expert_banks_remaining':0 if mode.startswith('direct_factors') else 8,'dense_expert_matrices_remaining':0 if mode.startswith('direct_factors') else 128,'source_model_tensor_bytes_excluding_optimizer':source_bytes,'resident_unique_storage_bytes':resident,'steady_allocated_before_inputs':steady,'load_peak_allocated':load_peak,'inference_peak_allocated':torch.cuda.max_memory_allocated(),'load_seconds':load_seconds,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'quality':quality,'samples':samples,'profile':profile,'precision':str(next(model.parameters()).dtype)+' execution; '+('unrounded originalFP32 checkpoint' if mode=='original_fp32' else 'archive factors/rows/sharedFP16; actual resident precision/storage counted'),'KV_cache_bytes':0,'limits':'small training model has no KV cache; every generated step recomputes its prefix; fixed fresh-process method order; finalD2H/hash excluded; includes router/top2 weights,SiLUup/down,attention,embedding/head,positionalweights and launches'})

@torch.no_grad()
def check_payload(archive,outpath):
    gpu_guard();package=torch.load(archive,weights_only=True,map_location='cpu');state=package['factors'];generator=torch.Generator(device='cuda:0').manual_seed(811);rows=[]
    for layer in range(4):
        for direction in ['up','down']:
            for expert in [0,7,15]:
                a,b,row=[state[f'blocks.{layer}.moe.{name}_{direction}'][expert].to('cuda:0') for name in ['a','b','row']]
                for dtype in [torch.float32,torch.float16]:
                    aa,bb,rr=[v.to(dtype) for v in [a,b,row]];weight=materialize(aa.float(),bb.float())*rr.float()[:,None]
                    for batch in [1,4,16]:
                        x=torch.randn(batch,512,device='cuda:0',dtype=dtype,generator=generator)[:,::2];actual=kron_forward(x,aa,bb,rr);reference=F.linear(x.float(),weight);error=float((actual.float()-reference).norm()/reference.norm().clamp_min(1e-20));limit=5e-5 if dtype==torch.float32 else .01
                        assert bool(actual.isfinite().all()) and error<limit,(layer,direction,expert,str(dtype),batch,error)
                        rows.append({'layer':layer,'projection':direction,'expert':expert,'dtype':str(dtype),'batch':batch,'relative_l2':error,'max_abs':float((actual.float()-reference).abs().max()),'relative_l2_limit':limit,'input_stride':list(x.stride())})
    write_json(outpath,{'passed':True,'cases':len(rows),'archive_only':True,'original_source_checkpoint_loaded':False,'rows':rows,'reference':'exact savedfactor/row payload materialized inFP32; sameinputs, direct FP32/FP16 contractions; no finaltest calibration'})

def main(out,args):
    candidates=[]
    for path in sorted((PREFIX/'results').glob('*_direct-factor-export_*/direct_factor_results.json')):
        d=json.loads(path.read_text())
        if d['kind']=='native' and d['seed']==11 and d['variant']=='all_fp16' and (path.parent/'exit.json').exists() and json.loads((path.parent/'exit.json').read_text())['exit_code']==0:candidates.append((path,d))
    source,export=candidates[-1];archive=pathlib.Path(export['archive']);assert digest(archive)==export['archive_sha256'];selected=json.loads(pathlib.Path(export['selected_source']).read_text());checkpoint=next(r['checkpoint'] for r in selected['choices'] if r['steps']==selected['selected_steps']);assert digest(checkpoint)==export['source_checkpoint_sha256'],'original checkpoint bytes changed';data=latest_data();artifact=ROOT/'artifacts'/('native_runtime_'+out.name);artifact.mkdir(exist_ok=False);runs={}
    subprocess.run([sys.executable,'-B','-c','import sys;from native_runtime import check_payload;check_payload(*sys.argv[1:])',str(archive),str(out/'payload_correctness.json')],check=True)
    for mode in ['original_fp32','reference_decompressed','direct_factors','reference_dense_fp16','direct_factors_fp16']:
        path=out/(mode+'.json');cmd=[sys.executable,'-B','-c','import sys;from native_runtime import child;child(*sys.argv[1:])',mode,str(archive),checkpoint,data['artifact'],str(path),str(artifact)];subprocess.run(cmd,check=True);runs[mode]=json.loads(path.read_text())
    original=runs['original_fp32']['quality'];reference=runs['reference_decompressed']['quality'];direct=runs['direct_factors']['quality'];comparisons=[]
    for precision,base_mode,direct_mode in [('FP32','reference_decompressed','direct_factors'),('FP16','reference_dense_fp16','direct_factors_fp16')]:
        for length in [32,96]:
            for forced in [False,True]:
                baseline=[r for r in runs[base_mode]['samples'] if r['prompt_length']==length and r['teacher_forced']==forced];actual=[r for r in runs[direct_mode]['samples'] if r['prompt_length']==length and r['teacher_forced']==forced];comparisons.append({'precision':precision,'prompt_length':length,'teacher_forced':forced,'direct_minus_dense_generation_seconds':paired_ci([a['host_prefill_and_generation_seconds']-b['host_prefill_and_generation_seconds'] for a,b in zip(actual,baseline)]),'same_generated_hash_groups':sum(a['generated_sha256']==b['generated_sha256'] for a,b in zip(actual,baseline))})
    write_json(out/'native_runtime_results.json',{'scope':'complete trained native MoE seed11,validation-selected checkpoint; noKVcache; separate FP32/FP16 execution policies, distinct from primary mixed-AMP training evaluation','selected_steps':selected['selected_steps'],'training_original_AMP_test_nll':export['original_test_nll'],'archive_sha256':export['archive_sha256'],'serialized_bytes':export['serialized_bytes'],'expert_bpw':export['expert_bpw'],'whole_model_tensor_bpw':export['whole_model_bpw'],'runs':runs,'rounded_export_minus_original_fp32_nll':block_bootstrap([a-b for a,b in zip(reference['block_nll'],original['block_nll'])]),'direct_minus_same_payload_reference_nll':block_bootstrap([a-b for a,b in zip(direct['block_nll'],reference['block_nll'])]),'direct_fp16_minus_dense_fp16_nll':block_bootstrap([a-b for a,b in zip(runs['direct_factors_fp16']['quality']['block_nll'],runs['reference_dense_fp16']['quality']['block_nll'])]),'direct_fp16_minus_original_fp32_nll':block_bootstrap([a-b for a,b in zip(runs['direct_factors_fp16']['quality']['block_nll'],original['block_nll'])]),'comparisons':comparisons,'training_equal_quality_claim':'not assessed here; use paired training seeds and the sameAMP protocol','acceptance':{'same_payload_nll_tolerance':.001,'same_model_storage_rounding_nll_tolerance':.01}})
