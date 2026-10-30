"""Complete gated/routed Granite MoE block from an exact packed fidelity archive."""
import json,random,time,torch
import numpy as np
import torch.nn.functional as F
from common import synchronize,paired_ci
from math_codec import decode,to_device
from packed_kernel import DirectOperator

@torch.no_grad()
def benchmark_banks(manifest_path,kernel,operator_class=DirectOperator):
    m=json.loads(manifest_path.read_text());directory=manifest_path.parent
    up=next(r for r in m['parameters'] if '.layers.0.' in r['name'] and r['name'].endswith('gate_up_proj'));down=next(r for r in m['parameters'] if '.layers.0.' in r['name'] and r['name'].endswith('down_proj'))
    u=torch.load(directory/up['file'],weights_only=True);d=torch.load(directory/down['file'],weights_only=True)
    prefix=up['name'].rsplit('.experts.',1)[0];shared=torch.load(directory/'shared.pt',weights_only=True);router=shared[prefix+'.router.weight'].to('cuda:0').float();topk=m['config']['num_experts_per_tok']
    ops=[([operator_class(p,kernel) for p in parts],operator_class(dp[0],kernel)) for parts,dp in zip(u['parts'],d['parts'])]
    dense=[([decode(to_device(p,'cuda:0')) for p in parts],decode(to_device(dp[0],'cuda:0'))) for parts,dp in zip(u['parts'],d['parts'])]
    densehalf=[([p.half() for p in parts],pdown.half()) for parts,pdown in dense]
    def forward(x,kind):
        score=F.linear(x.float(),router);value,index=score.topk(topk,-1);prob=value.softmax(-1);out=torch.zeros(x.shape[0],router.shape[1],device=x.device)
        for e in range(len(ops)):
            idx,slot=(index==e).nonzero(as_tuple=True)
            xe=x[idx]
            if kind=='direct':
                gate=ops[e][0][0](xe);up_=ops[e][0][1](xe);y=ops[e][1](F.silu(gate)*up_)
            else:
                bank=densehalf if kind=='dense_fp16' else dense;xe=xe if kind=='dense_fp16' else xe.float()
                gate=F.linear(xe,bank[e][0][0]);up_=F.linear(xe,bank[e][0][1]);y=F.linear(F.silu(gate)*up_,bank[e][1]).float()
            out.index_add_(0,idx,y*prob[idx,slot,None])
        return out,index
    rows=[];rng=random.Random(551)
    for batch in [1,4,16,128]:
        x=torch.randn(batch,router.shape[1],device='cuda:0',dtype=torch.float16)*.2;direct,index=forward(x,'direct');ref,_=forward(x,'dense_fp32');rel=float((direct-ref).norm()/ref.norm().clamp_min(1e-20));assert rel<.005,(batch,rel)
        # Avoid launching empty CUDA grids for inactive experts; DirectOperator supports empty input.
        samples={k:[] for k in ['direct','dense_fp16','dense_fp32']};host={k:[] for k in samples}
        for name in samples:
            for _ in range(3):forward(x,name)
        for group in range(20):
            order=list(samples);rng.shuffle(order)
            for name in order:
                synchronize();t=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(5):forward(x,name)
                b.record();b.synchronize();host[name].append((time.perf_counter()-t)*1000/5);samples[name].append(a.elapsed_time(b)/5)
        rows.append({'model':m['model'],'revision':m['revision'],'codec':m['codec'],'layer':0,'batch':batch,'topk':topk,'active_expert_union':int(index.unique().numel()),'relative_l2_error_to_exact_payload_block':rel,'device_ms':samples,'host_ms':host,'median_speedup_vs_dense_fp16':float(np.median(samples['dense_fp16'])/np.median(samples['direct'])),'direct_minus_dense_fp16_ci':paired_ci([a-b for a,b in zip(samples['direct'],samples['dense_fp16'])]),'resident_direct_operator_bytes':sum(op.bytes for pair,downop in ops for op in pair+[downop])+router.numel()*4,'scope':'full gated expert bank plus top-k routing/weights/scatter; excludes attention/residual/norm and complete-model generation','implementation':'readable expert loops; nonzero routing entails synchronization in all methods; no host logging inside calls','timing_groups':20,'repeats_per_group':5})
    return rows
