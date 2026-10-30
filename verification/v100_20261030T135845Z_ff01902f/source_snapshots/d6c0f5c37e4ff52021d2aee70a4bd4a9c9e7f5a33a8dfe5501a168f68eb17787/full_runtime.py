"""Complete Granite direct packed inference, memory and generation; fresh processes."""
import hashlib,json,pathlib,random,resource,subprocess,sys,time,torch
from torch import nn
from common import ROOT,PREFIX,write_json,digest,gpu_guard,synchronize,paired_ci
from math_codec import tensor_bytes
from packed_kernel import DirectOperator,load_kernel
from payload_validation import validate
from analysis_stats import weighted_block_delta

class DirectExperts(nn.Module):
    def __init__(self,up,down,kernel):
        super().__init__();self.num_experts=len(up['parts']);self.operators=[]
        for u,d in zip(up['parts'],down['parts']):
            for p in u+d:validate(p)
            self.operators.append(([DirectOperator(p,kernel) for p in u],DirectOperator(d[0],kernel)))
    def forward(self,hidden_states,top_k_index,top_k_weights):
        output=torch.zeros_like(hidden_states)
        for e,(pair,down) in enumerate(self.operators):
            idx,slot=(top_k_index==e).nonzero(as_tuple=True);x=hidden_states[idx]
            y=down(torch.nn.functional.silu(pair[0](x))*pair[1](x));output.index_add_(0,idx,y*top_k_weights[idx,slot,None])
        return output

def load_direct(directory):
    from transformers import AutoConfig,AutoModelForCausalLM
    directory=pathlib.Path(directory);manifest=json.loads((directory/'manifest.json').read_text());cfg=AutoConfig.for_model(manifest['config']['model_type'],**{k:v for k,v in manifest['config'].items() if k!='model_type'});cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    kernel=load_kernel();model=AutoModelForCausalLM.from_config(cfg,dtype=torch.float32).eval();all_operators=[]
    for l,layer in enumerate(model.model.layers):
        up=next(r for r in manifest['parameters'] if f'.layers.{l}.' in r['name'] and r['name'].endswith('gate_up_proj'));down=next(r for r in manifest['parameters'] if f'.layers.{l}.' in r['name'] and r['name'].endswith('down_proj'))
        for row in [up,down]:assert digest(directory/row['file'])==row['sha256']
        u=torch.load(directory/up['file'],weights_only=True);d=torch.load(directory/down['file'],weights_only=True);module=DirectExperts(u,d,kernel)
        layer.block_sparse_moe.experts=module
        for pair,downop in module.operators:all_operators.extend(pair+[downop])
    # Expert dense parameter banks have been removed before moving the model to CUDA.
    params=dict(model.named_parameters());assert not any('.experts.' in name for name in params)
    assert digest(directory/'shared.pt')==manifest['shared_sha256'];shared=torch.load(directory/'shared.pt',weights_only=True);assert set(params)==set(shared),'shared parameter coverage mismatch'
    with torch.no_grad():
        for name,p in shared.items():params[name].copy_(p)
    model.to('cuda:0');resident=tensor_bytes([dict(model.named_parameters()),dict(model.named_buffers()),[op.p for op in all_operators]])
    return model,{'direct_expert_banks':len(model.model.layers),'direct_matrix_operators':len(all_operators),'dense_expert_parameters_remaining':0,'model_and_payload_unique_storage_bytes':resident,'resident_operator_payload_bytes':sum(op.bytes for op in all_operators),'source_weights_loaded':False,'weights_transferred_from_CPU_per_forward':False}

@torch.no_grad()
def generate(model,prompt,continuation,teacher_forced):
    synchronize();host_start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();output=model(prompt,use_cache=True);next_token=output.logits[:,-1].argmax(-1,keepdim=True);b.record();b.synchronize();ttft=time.perf_counter()-host_start;prefill_ms=a.elapsed_time(b);past=output.past_key_values
    generated=[next_token];dstart=torch.cuda.Event(enable_timing=True);dend=torch.cuda.Event(enable_timing=True);host_decode=time.perf_counter();dstart.record()
    for n in range(15):
        incoming=continuation[:,n:n+1] if teacher_forced else next_token
        output=model(incoming,past_key_values=past,use_cache=True);past=output.past_key_values;next_token=output.logits[:,-1].argmax(-1,keepdim=True);generated.append(next_token)
    dend.record();dend.synchronize();seconds=time.perf_counter()-host_decode;device_ms=dstart.elapsed_time(dend);tokens=torch.cat(generated,-1).cpu().contiguous()
    return {'prefill_device_ms':prefill_ms,'host_time_to_first_token_seconds':ttft,'decode_device_ms_for_15_steps':device_ms,'host_decode_seconds_for_15_steps':seconds,'decode_tokens_per_second':15/seconds,'end_to_end_seconds':ttft+seconds,'generated_token_sha256':hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),'kv_cache_bytes':tensor_bytes([getattr(layer,'keys',None) for layer in getattr(past,'layers',[])]+[getattr(layer,'values',None) for layer in getattr(past,'layers',[])])}

def child(directory,mode,tokens_root,outpath,artifact):
    gpu_guard();torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    if mode=='direct':model,details=load_direct(directory)
    else:
        from downstream import archive_model
        model,_=archive_model(directory);details={'dense_expert_parameters_remaining':48,'model_and_payload_unique_storage_bytes':tensor_bytes([dict(model.named_parameters()),dict(model.named_buffers())]),'source_weights_loaded':False}
    synchronize();load_seconds=time.perf_counter()-begin;load_peak=torch.cuda.max_memory_allocated();steady=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();val=torch.load(pathlib.Path(tokens_root)/'validation.pt',weights_only=True).to('cuda:0');samples=[]
    for length in [32,128]:
        prompt=val[:length].unsqueeze(0);continuation=val[length:length+16].unsqueeze(0)
        for forced in [False,True]:
            generate(model,prompt,continuation,forced)
            for group in range(20):
                r=generate(model,prompt,continuation,forced);r.update(prompt_length=length,generated_tokens=16,teacher_forced=forced,group=group);samples.append(r)
    quality=None
    if mode=='direct':
        from pretrained import evaluate
        tokens=torch.load(pathlib.Path(tokens_root)/'test.pt',weights_only=True);quality=evaluate(model,tokens)
        # One actual decode call, with all weights resident, proves the path's operator mix.
        with torch.no_grad():warm=model(val[:32].unsqueeze(0),use_cache=True)
        trace=pathlib.Path(artifact)/'direct_one_decode_trace.json'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True,record_shapes=True) as prof:
            model(val[32:33].reshape(1,1),past_key_values=warm.past_key_values,use_cache=True);synchronize()
        prof.export_chrome_trace(str(trace));summary=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40)
        write_json(pathlib.Path(outpath).with_name('direct_profiler.json'),{'local_trace':str(trace),'trace_bytes':trace.stat().st_size,'trace_sha256':digest(trace),'summary':summary,'scope':'one complete actual decode forward, CPU+CUDA; full trace local-only'})
    write_json(outpath,{'mode':mode,'details':details,'load_seconds':load_seconds,'load_peak_allocated':load_peak,'steady_allocated_before_inputs':steady,'steady_reserved_before_inputs':reserved,'inference_peak_allocated':torch.cuda.max_memory_allocated(),'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'samples':samples,'full_test_quality':quality,'precision':'FP32 shared weights, direct FP32 contractions/accumulation with saved FP16 factors/scales and packed codes','host_sync_note':'one synchronization at first-token boundary; none added for per-token logging/return; eager expert nonzero routing synchronizations remain part of both measured paths'})

def main(out,args):
    candidates=sorted((PREFIX/'results').glob('*_distillation_*/distillation_results.json'));result=next(json.loads(p.read_text()) for p in reversed(candidates) if json.loads(p.read_text()).get('tier')=='main');directory=pathlib.Path(result['archive']);assert digest(directory/'manifest.json')==result['archive_manifest_sha256'];tokenroot=next(pathlib.Path(json.loads(p.read_text())['results']['int2_sign']['artifact']).parent for p in sorted((PREFIX/'results').glob('*_pretrained_*/pretrained_results.json')) if 'granite' in json.loads(p.read_text())['model']);artifact=ROOT/'artifacts'/('full_runtime_'+out.name);artifact.mkdir(exist_ok=False);runs={}
    # Fresh process order fixed in advance; every variant has20 timing groups and equal warmup.
    for mode in ['reference_decompressed','direct']:
        path=out/(mode+'.json');subprocess.run([sys.executable,'-B','-c','import sys;from full_runtime import child;child(*sys.argv[1:])',str(directory),mode,str(tokenroot),str(path),str(artifact)],check=True);runs[mode]=json.loads(path.read_text())
    original_reference=next(json.loads(p.read_text()) for p in sorted((PREFIX/'results').glob('*_distillation_*/fresh_reload_distilled.json')))
    direct=runs['direct']['full_test_quality'];quality_delta=weighted_block_delta(original_reference['rows'],direct['rows']);comparisons=[]
    for length in [32,128]:
        for forced in [False,True]:
            ref=[r for r in runs['reference_decompressed']['samples'] if r['prompt_length']==length and r['teacher_forced']==forced];act=[r for r in runs['direct']['samples'] if r['prompt_length']==length and r['teacher_forced']==forced]
            comparisons.append({'prompt_length':length,'teacher_forced':forced,'decode_seconds_direct_minus_reference_ci':paired_ci([a['host_decode_seconds_for_15_steps']-b['host_decode_seconds_for_15_steps'] for a,b in zip(act,ref)]),'same_generated_token_hash_groups':sum(a['generated_token_sha256']==b['generated_token_sha256'] for a,b in zip(act,ref)),'groups':20})
    write_json(out/'full_runtime_results.json',{'model':result['model'],'revision':result['revision'],'codec':'int2_sign_closed_loop_functional_fit','archive_manifest_sha256':result['archive_manifest_sha256'],'runs':runs,'quality_direct_minus_same_payload_decompressed':quality_delta,'comparisons':comparisons,'scope':'complete actual24-layer Granite forward, prefill, KV-cached greedy and identical-token teacher-forced decode; direct packed experts, unmodified shared weights; source checkpoint absent fromdecoder','limits':'one fixed validation prompt at context32/128,16 generated tokens; mode order not interleaved across fresh processes; no vendor-optimized packedINT4 whole-model comparator; does not establish universal speed','acceptance_reference_nll_tolerance':.001})
