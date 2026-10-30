"""Real routed inputs, actual gated outputs, and controlled codec component toggles."""
import json,pathlib,time,torch
import torch.nn.functional as F
from common import ROOT,PREFIX,write_json
from pretrained import load_model,reconstruct
from distillation import capture_inputs,metrics
from math_codec import encode,to_device,tensor_bytes
from payload_validation import validate

@torch.no_grad()
def main(out,args):
    source=next(p for p in sorted((ROOT/'artifacts').glob('pretrained_*/packed_int2_sign/manifest.json')) if 'granite' in json.loads(p.read_text())['model']);manifest=json.loads(source.read_text());tokenroot=source.parent.parent;train=torch.load(tokenroot/'train.pt',weights_only=True);val=torch.load(tokenroot/'validation.pt',weights_only=True);model=load_model(manifest['model'],manifest['revision']);rows=[]
    variants=[('activation_balance_rotation',True,True,True),('no_activation_weighting',False,True,True),('no_balancing',True,False,True),('no_rotation',True,True,False)]
    for layer in [0,12,23]:
        module=model.model.layers[layer].block_sparse_moe;x=capture_inputs(model,module,train,128);xv=capture_inputs(model,module,val,16);target=module(xv.unsqueeze(0)).reshape_as(xv);original_up=module.experts.gate_up_proj.detach().clone();original_down=module.experts.down_proj.detach().clone();values,indices=F.linear(x,module.router.weight).topk(model.config.num_experts_per_tok,-1);rms_up=[];rms_down=[];coverage=[]
        for e in range(module.experts.num_experts):
            idx=(indices==e).any(-1);xe=x[idx];coverage.append(int(idx.sum()))
            if xe.numel():
                g,u=F.linear(xe,original_up[e]).chunk(2,-1);h=F.silu(g)*u;rms_up.append(xe.square().mean(0).sqrt().clamp_min(1e-6));rms_down.append(h.square().mean(0).sqrt().clamp_min(1e-6))
            else:rms_up.append(torch.ones(x.shape[-1],device=x.device));rms_down.append(torch.ones(original_down.shape[-1],device=x.device))
        for name,activation,balance,rotation in variants:
            start=time.perf_counter();up={'mode':'bank','shape':list(original_up.shape),'parts':[]};down={'mode':'bank','shape':list(original_down.shape),'parts':[]}
            for e,(wu,wd) in enumerate(zip(original_up,original_down)):
                up['parts'].append([to_device(encode(part,'int2_sign',balance=balance,rotation=rotation,activation=rms_up[e] if activation else None),'cpu') for part in wu.chunk(2,0)])
                down['parts'].append([to_device(encode(wd,'int2_sign',balance=balance,rotation=rotation,activation=rms_down[e] if activation else None),'cpu')])
            try:
                for package in [up,down]:
                    for parts in package['parts']:
                        for payload in parts:validate(payload)
                module.experts.gate_up_proj.copy_(reconstruct(up));module.experts.down_proj.copy_(reconstruct(down));pred=module(xv.unsqueeze(0)).reshape_as(xv)
                if not bool(pred.isfinite().all()):raise ValueError('nonfinite decoded bank output')
                measurement=metrics(pred,target);status='completed'
            except ValueError as error:measurement={'r_squared':None,'error':str(error)};status='unsupported_numerical_configuration';pred=None
            row={'layer':layer,'variant':name,'status':status,'codec':'rank8_int2_sign','activation_weighting':activation,'balancing':balance,'rotation':rotation,'real_expert_token_coverage':coverage,'heldout_real_gated_bank':measurement,'payload_bytes':tensor_bytes([up,down]),'seconds':time.perf_counter()-start,'calibration':'128 train-only length128 sequences; true routed gate/up inputs and true nonlinear down inputs','validation':'16 untouched-for-fitting validation sequences length128','prefix':'original independent prefix; all other weights unchanged','scope':'complete gated/routed bank at selected layer; no full-model test or downstream claim'}
            rows.append(row);write_json(out/f'layer_{layer:03d}_{name}.json',row);print(layer,name,row['heldout_real_gated_bank']['r_squared'],flush=True)
            module.experts.gate_up_proj.copy_(original_up);module.experts.down_proj.copy_(original_down)
            del up,down,pred
        del x,xv,target,original_up,original_down;torch.cuda.empty_cache()
    write_json(out/'calibration_ablation.json',{'model':manifest['model'],'revision':manifest['revision'],'rows':rows,'selected_depths':[0,12,23],'task_accuracy_claim':False,'full_model_fidelity_claim':False})
