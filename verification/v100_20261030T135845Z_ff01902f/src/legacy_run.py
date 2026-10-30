"""Reproduce the supplied SHA-verified legacy task; never invoke original-directory I/O."""
import json,pathlib,time,torch
import torch.nn.functional as F
import legacy_reference as legacy
from common import ROOT,write_json,digest
from data_audit import latest_data

def safe_forward(self,x):
    shape=x.shape;xf=x.reshape(-1,self.d_model)
    probs=F.softmax(self.router(xf),-1);tp,ti=torch.topk(probs,self.top_k,-1);tp=tp/(tp.sum(-1,keepdim=True)+1e-8)
    out=torch.zeros_like(xf);wu=self.get_up_weights();wd=self.get_down_weights()
    for k in range(self.top_k):
        for e in range(self.num_experts):
            idx=(ti[:,k]==e).nonzero(as_tuple=True)[0]
            if idx.numel():
                y=F.linear(F.silu(F.linear(xf.index_select(0,idx),wu[e])),wd[e])
                out.index_add_(0,idx,(tp[idx,k:k+1]*y).to(out.dtype))
    return out.reshape(shape)

def main(out,args):
    torch.set_num_threads(4)
    data=latest_data();tokens=torch.load(pathlib.Path(data['artifact'])/'legacy_tokens.pt',weights_only=True).to('cuda:0')
    legacy.load_vram_token_dataset=lambda device,vocab_size=512,num_tokens=1000000:(tokens[:900000],tokens[900000:1000000],'TinyShakespeare (Bigram-512 Tokenized); pinned source')
    # Explicit cast fixes current PyTorch index_add_ source/destination autocast dtype mismatch.
    legacy.DenseMoELayer.forward=safe_forward;legacy.NativeRPMIHOGMoELayer.forward=safe_forward
    # Prevent extra shrinkage when AMP skips updates; instrument exact successful steps.
    skipped=[];original_step=torch.amp.GradScaler.step
    def scaler_step(self,optimizer,*a,**kw):
        ret=original_step(self,optimizer,*a,**kw)
        found=sum(float(v) for v in self._per_optimizer_states[id(optimizer)]['found_inf_per_device'].values())
        skipped.append(bool(found));return ret
    torch.amp.GradScaler.step=scaler_step
    original_pool=legacy.pool_adamw_second_moments_rowwise;original_shrink=legacy.apply_qr_subspace_kronecker_shrinkage
    legacy.pool_adamw_second_moments_rowwise=lambda *a,**kw:None if skipped[-1] else original_pool(*a,**kw)
    legacy.apply_qr_subspace_kronecker_shrinkage=lambda w,q,**kw:q if skipped[-1] else original_shrink(w,q,**kw)
    steps=args.steps or (3 if args.tier=='smoke' else 300)
    result=legacy.run_three_way_step1_ablation(steps=steps,batch_size=32,seq_len=64)
    result['deviations']=['pinned corpus; no Markov fallback','index_add_ contribution cast to accumulator dtype for current PyTorch AMP','interventions only after successful updates; skipped counts recorded','legacy labels preserved only as original field names: evaluation is FP32 and weights reconstructed dense']
    result['amp']={'attempted_updates':len(skipped),'skipped_updates':sum(skipped),'per_update_skipped':skipped}
    result['seed']=42;result['batch_rng_seed']=1337;result['parameters_dense']=sum(p.numel() for p in legacy.NanoMoELM().parameters());result['parameters_native']=sum(p.numel() for p in legacy.NanoMoELM(native_generator=True).parameters())
    write_json(out/'legacy_results.json',result)
