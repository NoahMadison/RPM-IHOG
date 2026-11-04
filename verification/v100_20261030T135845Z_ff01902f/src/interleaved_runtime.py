"""Randomized complete-model comparisons of the exact fidelity-tested archives."""
import gc,json,pathlib,random,subprocess,sys,time,torch
from common import ROOT,PREFIX,write_json,digest,gpu_guard,synchronize,paired_ci

def share_unmodified(baseline,direct):
    # Assign every alias, including tied embedding/head aliases, rather than only unique names.
    params=dict(baseline.named_parameters(remove_duplicate=False));count=0
    for name,p in list(direct.named_parameters(remove_duplicate=False)):
        assert name in params and p.dtype==params[name].dtype and torch.equal(p,params[name]),name
        parent,_,leaf=name.rpartition('.');setattr(direct.get_submodule(parent),leaf,params[name]);count+=1
    buffers=dict(baseline.named_buffers(remove_duplicate=False))
    for name,b in list(direct.named_buffers(remove_duplicate=False)):
        if name in buffers and b.dtype==buffers[name].dtype and torch.equal(b,buffers[name]):
            parent,_,leaf=name.rpartition('.');setattr(direct.get_submodule(parent),leaf,buffers[name])
    gc.collect()
    if torch.cuda.is_initialized():torch.cuda.empty_cache()
    return count

@torch.no_grad()
def child(kind,precision,archive,tokenroot,outpath):
    gpu_guard();torch.set_num_threads(4);begin=time.perf_counter()
    if kind=='native':
        from native_runtime import load_model,generation
        ref_mode='reference_dense_fp16' if precision=='FP16' else 'reference_decompressed';direct_mode='direct_factors_fp16' if precision=='FP16' else 'direct_factors'
        baseline,_=load_model(ref_mode,archive,'');direct,_=load_model(direct_mode,archive,'');generate=generation;lengths=[32,96]
    else:
        from downstream import archive_model
        from full_runtime import load_direct,generate
        baseline,_=archive_model(archive);direct,_=load_direct(archive,fused=True);lengths=[32,128]
    aliases=share_unmodified(baseline,direct);synchronize();setup=time.perf_counter()-begin;tokens=torch.load(pathlib.Path(tokenroot)/'validation.pt',weights_only=True).to('cuda:0');prompt=tokens[:32].reshape(1,-1);r=baseline(prompt);a=direct(prompt);r=r.logits if hasattr(r,'logits') else r;a=a.logits if hasattr(a,'logits') else a;relative=float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-20));max_abs=float((a.float()-r.float()).abs().max());limit=.01 if precision=='FP16' else .001
    assert bool(a.isfinite().all()) and bool(r.isfinite().all()) and relative<limit,(kind,precision,relative)
    del r,a;gc.collect();torch.cuda.empty_cache();models={'dense':baseline,'direct':direct};rng=random.Random(4711);samples=[]
    for length in lengths:
        prompt=tokens[:length].reshape(1,-1);continuation=tokens[length:length+16].reshape(1,-1)
        for forced in [False,True]:
            for mode in ['dense','direct']:
                for _ in range(3):generate(models[mode],prompt,continuation,forced)
            for group in range(20):
                order=['dense','direct'];rng.shuffle(order)
                for position,mode in enumerate(order):
                    begin_ns=time.time_ns();row=generate(models[mode],prompt,continuation,forced)
                    total=row.get('host_prefill_and_generation_seconds',row.get('end_to_end_seconds'));first=row.get('prefill_first_token_gpu_ready_host_seconds',row.get('host_time_to_first_token_seconds'));row.update(mode=mode,group=group,order=order,order_position=position,prompt_length=length,teacher_forced=forced,host_start_unix_ns=begin_ns,total_model_seconds=total,decode_seconds=total-first);samples.append(row)
            print(kind,precision,length,forced,'completed20interleavedgroups',flush=True)
    comparisons=[]
    for length in lengths:
        for forced in [False,True]:
            rows=[r for r in samples if r['prompt_length']==length and r['teacher_forced']==forced];dense=sorted([r for r in rows if r['mode']=='dense'],key=lambda r:r['group']);direct_rows=sorted([r for r in rows if r['mode']=='direct'],key=lambda r:r['group']);hashes=lambda r:r.get('generated_sha256',r.get('generated_token_sha256'))
            comparisons.append({'prompt_length':length,'teacher_forced':forced,'total_model_seconds_direct_minus_dense_ci':paired_ci([a['total_model_seconds']-b['total_model_seconds'] for a,b in zip(direct_rows,dense)]),'decode_seconds_direct_minus_dense_ci':paired_ci([a['decode_seconds']-b['decode_seconds'] for a,b in zip(direct_rows,dense)]),'same_generated_hash_groups':sum(hashes(a)==hashes(b) for a,b in zip(direct_rows,dense)),'groups':20})
    write_json(outpath,{'kind':kind,'precision':precision,'archive':str(archive),'setup_seconds':setup,'shared_parameter_alias_count':aliases,'sanity_forward_relative_l2':relative,'sanity_forward_max_abs':max_abs,'sanity_forward_relative_limit':limit,'both_models_resident_allocated':torch.cuda.memory_allocated(),'both_models_resident_reserved':torch.cuda.memory_reserved(),'samples':samples,'comparisons':comparisons,'protocol':'seed4711 method-order shuffle per group;20pairedgroups,3equalwarmups/condition; shared nonexpert parameter aliases including tiedweights; expertpayloads distinct and resident; no per-call CPU weight transfer','limits':'both expert representations simultaneously resident only for interleaved timing; per-representation residency comes from isolated prior runs. Granite has KV caching; native recomputes prefixes. FinalD2H/hash excluded. One fixed validation prompt per context/forcing condition; not independent training seeds.'})

def main(out,args):
    full=[]
    for path in sorted((PREFIX/'results').glob('*_full-runtime_*/full_runtime_results.json')):
        if (path.parent/'exit.json').exists() and json.loads((path.parent/'exit.json').read_text())['exit_code']==0:
            d=json.loads(path.read_text())
            if d['runs']['direct']['details'].get('implementation')=='fused_FWHT_v2':full.append((path,d))
    source,core=full[-1];fit_path=next(p for p in sorted((PREFIX/'results').glob('*_distillation_*/distillation_results.json')) if json.loads(p.read_text()).get('archive_manifest_sha256')==core['archive_manifest_sha256']);fit=json.loads(fit_path.read_text());pretrained=next((p,json.loads(p.read_text())) for p in sorted((PREFIX/'results').glob('*_pretrained_*/pretrained_results.json')) if 'granite' in json.loads(p.read_text())['model']);p,pretrained=pretrained;int4=pretrained['results']['int4'];int4_quality=next(q for q in sorted((PREFIX/'results').glob('*_full-int4-runtime_*/full_runtime_results.json')) if json.loads(q.read_text())['archive_manifest_sha256']==int4['archive_sha256'] and (q.parent/'exit.json').exists() and json.loads((q.parent/'exit.json').read_text())['exit_code']==0);granite_tokens=str(pathlib.Path(int4['artifact']).parent)
    native_path=next(p for p in sorted((PREFIX/'results').glob('*_native-runtime_*/native_runtime_results.json')) if (p.parent/'exit.json').exists() and json.loads((p.parent/'exit.json').read_text())['exit_code']==0);native=json.loads(native_path.read_text());export=next(json.loads(p.read_text()) for p in sorted((PREFIX/'results').glob('*_direct-factor-export_*/direct_factor_results.json')) if json.loads(p.read_text())['archive_sha256']==native['archive_sha256'])
    from data_audit import latest_data
    native_tokens=latest_data()['artifact'];cases=[('granite_core','FP32',fit['archive'],granite_tokens,core['archive_manifest_sha256'],str(source)),('granite_int4','FP32',int4['artifact'],granite_tokens,int4['archive_sha256'],str(int4_quality)),('native','FP32',export['archive'],native_tokens,native['archive_sha256'],str(native_path)),('native','FP16',export['archive'],native_tokens,native['archive_sha256'],str(native_path))];results=[]
    write_json(out/'cases_before_timing.json',{'cases':cases,'quality_reused':'exact saved artifact and same operator/precision; previous complete fresh-process NLL retained; new forward sanity check before timing','protocol_config':'configs/027_interleaved_complete_runtime.json'})
    for kind,precision,archive,tokens,sha,quality_source in cases:
        archive_path=pathlib.Path(archive);assert digest(archive_path/'manifest.json' if archive_path.is_dir() else archive_path)==sha,'payload changed';path=out/(kind+'_'+precision+'.json');subprocess.run([sys.executable,'-B','-c','import sys;from interleaved_runtime import child;child(*sys.argv[1:])',kind,precision,archive,tokens,str(path)],check=True);d=json.loads(path.read_text());prior=json.loads(pathlib.Path(quality_source).read_text());quality=prior['quality_direct_minus_same_payload_decompressed'] if kind!='native' else prior['direct_fp16_minus_dense_fp16_nll'] if precision=='FP16' else prior['direct_minus_same_payload_reference_nll'];bounds=quality['ci95'];d.update(archive_sha256=sha,prior_complete_fidelity_source=quality_source,same_payload_full_NLL_delta=quality,same_payload_NLL_equivalent_within_001=bounds[0]>=-.001 and bounds[1]<=.001);results.append(d)
    write_json(out/'interleaved_runtime_results.json',{'results':results,'random_seed':4711,'scope':'same actual complete artifacts as fidelity/memory tests; interleaved method groups supersede fixed-order latency confidence, not their separately measured memory or quality','quality_is_not_task_accuracy':True})
