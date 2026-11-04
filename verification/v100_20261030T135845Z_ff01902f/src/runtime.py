import json,pathlib,random,time,torch
import numpy as np
from common import ROOT,write_json,synchronize,paired_ci
from math_codec import decode,to_device,reference_forward,kron_forward,fwht,tensor_bytes
from packed_kernel import DirectOperator,load_kernel

def actual_payloads():
    # Exact artifacts whose full-model quality is measured; never mix another rank/codec's timings.
    paths=sorted((ROOT/'artifacts').glob('pretrained_*/packed_*/manifest.json'));rows=[]
    for path in paths:
        manifest=json.loads(path.read_text())
        for parameter in manifest['parameters'][:1]:
            package=torch.load(path.parent/parameter['file'],weights_only=True)
            parts=package['parts'][0] if package['mode']=='bank' else package['parts']
            for idx,payload in enumerate(parts[:1]):rows.append(({'model':manifest['model'],'revision':manifest['revision'],'codec':manifest['codec'],'parameter':parameter['name'],'complete_expert':0 if package['mode']=='bank' else None,'part':idx,'manifest':str(path),'shape':payload['meta']['shape']},payload))
    return rows

@torch.no_grad()
def main(out,args):
    torch.set_num_threads(4);compile_start=time.perf_counter()
    if args.codec=='fused':
        from packed_kernel_fused import load_kernel as loader,FusedOperator as operator_class
    else:loader=load_kernel;operator_class=DirectOperator
    try:kernel=loader()
    except Exception as e:
        write_json(out/'runtime.json',{'status':'blocked','stage':'optimized_sm70_kernel_compilation','error':type(e).__name__+': '+str(e),'fallback':'reference correctness and quality stages remain independently valid'});return
    compile_seconds=time.perf_counter()-compile_start;rows=[];rng=random.Random(171)
    payloads=actual_payloads()
    if not payloads:raise RuntimeError('run full pretrained export/reload first; no faithful payload available')
    for identity,payload in payloads:
        p=to_device(payload,'cuda:0');w=decode(p);wh=w.half();op=operator_class(payload,kernel);o,i=w.shape
        for batch in [1,4,16,128]:
            xh=torch.randn(batch,i,device='cuda:0',dtype=torch.float16);x=xh.float();baseline=x@w.T;direct=op(xh);err=float((direct-baseline).abs().max());relative=float((direct-baseline).norm()/baseline.norm().clamp_min(1e-20))
            assert torch.allclose(direct,baseline,atol=2e-3,rtol=2e-3),(identity,err,relative)
            fns={'dense_fp16':lambda:xh@wh.T,'dense_fp32':lambda:xh.float()@w.T,'direct_complete':lambda:op(xh),'reference_decompressed':lambda:reference_forward(xh,p)}
            for fn in fns.values():
                for _ in range(10):fn()
            synchronize();samples={key:[] for key in fns};host={key:[] for key in fns}
            for group in range(20):
                order=list(fns);rng.shuffle(order)
                for name in order:
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);synchronize();t=time.perf_counter();a.record()
                    for _ in range(20):fns[name]()
                    b.record();b.synchronize();host[name].append((time.perf_counter()-t)*1000/20);samples[name].append(a.elapsed_time(b)/20)
            torch.cuda.reset_peak_memory_stats();base_alloc=torch.cuda.memory_allocated();op(x);synchronize();temporary=torch.cuda.max_memory_allocated()-base_alloc
            rows.append({'identity':identity,'batch':batch,'input_dtype':'FP16; already resident; direct path includes FP32 promotion','max_abs_error':err,'relative_l2_error':relative,'device_ms':samples,'host_ms':host,'direct_minus_dense_fp16_ms_paired_ci':paired_ci([a-b for a,b in zip(samples['direct_complete'],samples['dense_fp16'])]),'median_speedup_vs_fp16':float(np.median(samples['dense_fp16'])/np.median(samples['direct_complete'])),'serialized_payload_tensor_bytes':tensor_bytes(payload),'resident_direct_bytes':op.bytes,'dense_fp16_bytes':wh.numel()*2,'direct_peak_temporary_bytes':temporary,'scope':'complete matrix operator; no model decode speed claim','cache':'same-weight cache-resident reuse; streaming working-set not measured'})
            if batch==1:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True,record_shapes=True) as prof:op(x);synchronize()
                trace=out/f'profile_{len(rows):03d}.json';prof.export_chrome_trace(str(trace));(out/f'profile_{len(rows):03d}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
    from bank_runtime import benchmark_banks
    bank_rows=[]
    for path in sorted((ROOT/'artifacts').glob('pretrained_*/packed_*/manifest.json')):
        manifest=json.loads(path.read_text())
        if 'granite' in manifest['model']:bank_rows.extend(benchmark_banks(path,kernel,operator_class))
    # CPU pure-core component timing against exactly materialized core, fixed one thread.
    cpu_rows=[];torch.set_num_threads(1)
    for identity,payload in payloads:
        if 'a' not in payload:continue
        from math_codec import materialize
        a=payload['a'].float().cpu();b=payload['b'].float().cpu();wcore=materialize(a,b);xcpu=torch.randn(1,wcore.shape[1]);ref=xcpu@wcore.T
        assert torch.allclose(kron_forward(xcpu,a,b),ref,atol=1e-3,rtol=1e-3)
        samples={'core':[],'dense_core':[]}
        functions={'core':lambda:kron_forward(xcpu,a,b),'dense_core':lambda:xcpu@wcore.T}
        for fn in functions.values():
            for _ in range(10):fn()
        for group in range(20):
            order=list(functions);rng.shuffle(order)
            for name in order:
                t=time.perf_counter()
                for _ in range(20):functions[name]()
                samples[name].append((time.perf_counter()-t)*1000/20)
        cpu_rows.append({'identity':identity,'threads':1,'samples_ms':samples,'median_core_speedup':float(np.median(samples['dense_core'])/np.median(samples['core'])),'scope':'CPU pure core component only; dense comparator materializes exactly this core, no residual or full-model quality attached'})
    # Sustained device-to-device copy bandwidth: measured diagnostic, separate from kernel arithmetic roofline.
    a=torch.randn(32*1024*1024,device='cuda:0');b=torch.empty_like(a)
    for _ in range(10):b.copy_(a)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
    for _ in range(50):b.copy_(a)
    end.record();end.synchronize();seconds=start.elapsed_time(end)/1000;bandwidth=(a.numel()*a.element_size()*2*50)/seconds
    write_json(out/'runtime.json',{'status':'completed','compilation_seconds':compile_seconds,'architecture':'sm_70','rows':rows,'routed_gated_bank_rows':bank_rows,'cpu_core_components':cpu_rows,'device_copy_bandwidth_bytes_per_second_counting_read_write':bandwidth,'roofline_notes':'Direct kernel uses FP32 FMA for residual, integer bit extraction, FP16 loads/conversion, PyTorch FP32 core contractions and activation/output block-16 FWHT. It cannot use the tensor-core peak as its compute ceiling. Intermediate vectors and contractions, repeated launches, and cache-resident reuse are included in complete-operator timing. Copy bandwidth is diagnostic, not proof of decompression being free. Bank timings include gating/routing; no complete model generation or streaming-working-set speed claim is established.'})
