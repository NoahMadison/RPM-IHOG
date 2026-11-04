"""One-GPU algebra/payload/compression diagnostics, explicitly not multi-GPU throughput."""
import json,pathlib,time,torch
import torch.nn.functional as F
from common import ROOT,PREFIX,write_json,synchronize
from math_codec import materialize,unfold,fold,pack_codes,unpack_codes,tensor_bytes,spectra
from models import make_paired
from data_audit import latest_data

def parameter_payload(model):
    parameters={n:p.detach() for n,p in model.named_parameters()};expert={n:p for n,p in parameters.items() if '.moe.' in n and '.router.' not in n}
    return {'whole_parameter_count':sum(p.numel() for p in parameters.values()),'whole_gradient_bytes_fp32':tensor_bytes(parameters),'expert_gradient_bytes_fp32':tensor_bytes(expert),'adam_moments_bytes_fp32':2*tensor_bytes(parameters),'ddp_optimizer_state_synchronized':False}

def shared_project(g,basis,kron=False):
    k=unfold(g,16,16) if kron else g
    coefficient=basis.T@k;reconstructed=basis@coefficient
    if kron:reconstructed=fold(reconstructed,16,g.shape[0]//16,16,g.shape[1]//16)
    return coefficient,reconstructed

def main(out,args):
    torch.set_num_threads(4);algebra=[];models={}
    data=latest_data();tokens=torch.load(pathlib.Path(data['artifact'])/'train.pt',weights_only=True).to('cuda:0');batch=torch.stack([tokens[n*32:n*32+17] for n in range(3)])
    for kind in ['adamw','native','lowrank']:
        model=make_paired(kind,11).to('cuda:0').eval();models[kind]=parameter_payload(model)
        def gradients(chunks):
            model.zero_grad(set_to_none=True)
            for c in chunks:
                loss=F.cross_entropy(model(c[:,:-1]).reshape(-1,50257),c[:,1:].reshape(-1),reduction='sum')/(3*16);loss.backward()
            return {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        full=gradients([batch]);shard=gradients([batch[:2],batch[2:]])
        maxerr=max(float((full[n]-shard[n]).abs().max()) for n in full)
        rel=(sum(float((full[n]-shard[n]).square().sum()) for n in full)/max(1e-30,sum(float(g.square().sum()) for g in full.values())))**.5
        assert rel<2e-5,(kind,rel)
        # Explicit decoupled-SGD update from one shared parameter state, same weighted mean gradients.
        update_error=max(float(((p.detach()*.999-.001*full[n])-(p.detach()*.999-.001*shard[n])).abs().max()) for n,p in model.named_parameters() if n in full)
        assert update_error<2e-6,(kind,update_error)
        algebra.append({'kind':kind,'microbatch_shards':[2,1],'global_sequences':3,'tokens_per_sequence':16,'gradient_max_abs_error':maxerr,'gradient_relative_l2_error':rel,'optimizer':'SGD lr=.001 decoupled decay .001','optimizer_update_max_abs_error':update_error,'stochastic_effects':'dropout disabled, fixed inputs, FP32'})
        del model,full,shard;torch.cuda.empty_cache()
    # Real dense expert gradients at a trained checkpoint, rather than weight compressibility as proxy.
    checkpoint_paths=sorted((ROOT/'artifacts').glob('training_*/checkpoint_*.pt'));chosen=None
    for path in reversed(checkpoint_paths):
        cp=torch.load(path,weights_only=True,map_location='cpu')
        if cp['kind']=='adamw' and not cp.get('hyperparameters',{}).get('function_matched',False):chosen=(path,cp);break
    if chosen is None:raise RuntimeError('need trained dense checkpoint for real gradient audit')
    path,cp=chosen;model=make_paired('adamw',cp['seed']).to('cuda:0');model.load_state_dict(cp['model']);model.eval();local=[]
    for c in [batch[:2],batch[2:]]:
        model.zero_grad(set_to_none=True);F.cross_entropy(model(c[:,:-1]).reshape(-1,50257),c[:,1:].reshape(-1)).backward();local.append(model.blocks[2].moe.w_up.grad.detach().clone())
    weights=[2/3,1/3];bank_gradient=sum(w*g for w,g in zip(weights,local));bank_norms=bank_gradient.flatten(1).norm(dim=-1);chosen_expert=int(bank_norms.argmax());assert float(bank_norms[chosen_expert])>1e-12,'all expert gradients zero; no useful real-gradient audit'
    local=[g[chosen_expert] for g in local];target=sum(w*g for w,g in zip(weights,local));rank=8;g0=target
    synchronize();basis_start=time.perf_counter();ordinary_basis=torch.linalg.svd(g0,full_matrices=False).U[:,:rank];synchronize();ordinary_refresh_seconds=time.perf_counter()-basis_start
    synchronize();basis_start=time.perf_counter();kron_basis=torch.linalg.svd(unfold(g0,16,16),full_matrices=False).U[:,:rank];synchronize();kron_refresh_seconds=time.perf_counter()-basis_start
    observed_spectra={'real_gradient':spectra(target),'trained_weight':spectra(model.blocks[2].moe.w_up.detach()[chosen_expert])}
    previous=[p for p in sorted(path.parent.glob('checkpoint_*.pt')) if p.name<path.name]
    if previous:
        old=torch.load(previous[-1],weights_only=True,map_location='cpu');delta=cp['model']['blocks.2.moe.w_up'][chosen_expert].to('cuda:0')-old['model']['blocks.2.moe.w_up'][chosen_expert].to('cuda:0');observed_spectra['weight_delta_between_checkpoints']={'from':str(previous[-1]),'to':str(path),**spectra(delta)}
    compressors=[]
    for name,basis,kron in [('ordinary_shared_PowerSGD_style',ordinary_basis,False),('shared_Kronecker_unfolded_projection',kron_basis,True)]:
        residual=[torch.zeros_like(g) for g in local];start=time.perf_counter();synchronize();coefficient_bytes=0
        for step in range(20):
            coefficients=[];recon=[]
            for worker,g in enumerate(local):
                corrected=g+residual[worker];c,r=shared_project(corrected,basis,kron);residual[worker]=corrected-r;coefficients.append(c);recon.append(r)
            mean_c=sum(w*c for w,c in zip(weights,coefficients));rmean=basis@mean_c
            if kron:rmean=fold(rmean,16,16,16,16)
            weighted=sum(w*r for w,r in zip(weights,recon));assert torch.allclose(rmean,weighted,atol=1e-5)
            coefficient_bytes=tensor_bytes(coefficients[0])
        synchronize();seconds=(time.perf_counter()-start)/20
        zero=torch.zeros_like(target);_,zr=shared_project(zero,basis,kron);assert not bool(zr.any())
        compressors.append({'name':name,'rank':rank,'coefficient_bytes_per_worker':coefficient_bytes,'shared_basis_bytes':tensor_bytes(basis),'basis_refresh_protocol':'centralized exact SVD of initial weighted pilot gradient; full gradient aggregate plus one basis broadcast, then frozen for20 repeated-gradient diagnostic steps; no claim of deployable distributed basis training','basis_SVD_seconds':kron_refresh_seconds if kron else ordinary_refresh_seconds,'basis_refresh_full_gradient_bytes_per_worker':target.numel()*target.element_size(),'basis_refresh_collective_rounds':2,'coefficient_collectives_per_step':1,'payload_dtype':'FP32 numeric coefficients, never uint8 code summation','gpu_compression_aggregation_seconds':seconds,'relative_error_final':float((rmean-target).norm()/target.norm().clamp_min(1e-20)),'local_error_feedback_bytes_per_worker':target.numel()*target.element_size(),'projection_changes_with_error_feedback':True})
    # Quantized error feedback, per-worker scales decoded before weighted aggregation (all-gather semantics).
    residual=[torch.zeros_like(g) for g in local];start=time.perf_counter()
    for _ in range(20):
        decoded=[]
        for worker,g in enumerate(local):
            corrected=g+residual[worker];scale=corrected.abs().max().clamp_min(1e-12)/127;codes=(corrected/scale).round().clamp(-127,127).to(torch.int8);r=codes.float()*scale;residual[worker]=corrected-r;decoded.append(r)
        result=sum(w*r for w,r in zip(weights,decoded))
    synchronize();compressors.append({'name':'int8_error_feedback_allgather_decode','payload_bytes_per_worker':target.numel()+4,'collective_rounds_per_step':1,'worker_scaling':'all-gather P*(P-1)*payload total bytes; each worker decodes P local scales','gpu_seconds':(time.perf_counter()-start)/20,'relative_error_final':float((result-target).norm()/target.norm().clamp_min(1e-20)),'zero_scale_floor':1e-12})
    # Counterexample to averaging independently fitted Kronecker factors.
    a1=torch.randn(2,2,device='cuda:0');a2=torch.randn_like(a1);b1=torch.randn_like(a1);b2=torch.randn_like(a1)
    correct=.5*(torch.kron(a1,b1)+torch.kron(a2,b2));incorrect=torch.kron((a1+a2)/2,(b1+b2)/2);counterexample_error=float((correct-incorrect).norm());assert counterexample_error>1e-4
    projections=[]
    for workers in [2,4,8]:
        for gbps in [1,10,25,100]:
            for latency_us in [10,100]:
                for kind,payload in models.items():
                    # Ring allreduce bytes per worker, bandwidth/latency sensitivity only.
                    traffic=2*(workers-1)/workers*payload['whole_gradient_bytes_fp32'];seconds=traffic/(gbps*1e9/8)+2*(workers-1)*latency_us/1e6
                    projections.append({'workers':workers,'link_gbps':gbps,'hop_latency_us':latency_us,'kind':kind,'ring_bytes_per_worker':traffic,'projected_collective_seconds':seconds,'measured_multi_GPU':False})
    write_json(out/'distributed.json',{'aggregation_checks':algebra,'payloads':models,'gradient_source_checkpoint':str(path),'real_gradient_expert_selection':{'rule':'largest weighted-gradient Frobenius norm in middle-layer up bank; avoids treating a cold zero gradient as compression evidence','expert':chosen_expert,'all_expert_norms':bank_norms.cpu().tolist(),'gradient_frobenius':float(target.norm())},'real_weight_gradient_delta_spectra':observed_spectra,'compressor_diagnostics':compressors,'independent_factor_averaging_counterexample_l2':counterexample_error,'sensitivity_projections':projections,'actual_multi_gpu_status':'blocked','blocker':'only one authorized V100; no duplicate-device ranks or RTX 4060 use permitted','scope':'sequential sharded gradients and measured one-GPU codec cost; no measured distributed speed or time-to-quality'})
