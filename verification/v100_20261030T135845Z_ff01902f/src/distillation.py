"""Sequential closed-loop real-activation fitting of complete gated Granite banks.

Only factors, row/column multipliers and a scalar residual gain per matrix train.
Codes stay fixed; gain folds into existing residual scales before FP16 export.
Quality execution remains reference_decompressed, with a fresh full-archive load.
"""
import copy,json,math,pathlib,shutil,subprocess,sys,time
import torch
from torch import nn
import torch.nn.functional as F
from common import PREFIX,ROOT,write_json,digest,synchronize
from math_codec import materialize,residual_dense,rotate,to_device,tensor_bytes
from pretrained import load_model,reconstruct
from analysis_stats import weighted_block_delta

class Matrix(nn.Module):
    def __init__(self,payload):
        super().__init__();p=to_device(payload,'cuda:0');self.original=payload
        self.a=nn.Parameter(p['a'].float());self.b=nn.Parameter(p['b'].float())
        self.register_buffer('row',p['row'].float());self.register_buffer('col',p['col'].float());self.register_buffer('residual',rotate(residual_dense(p),p['meta']['block']))
        self.row_mult=nn.Parameter(torch.ones_like(self.row));self.col_mult=nn.Parameter(torch.ones_like(self.col));self.gain=nn.Parameter(torch.ones((),device='cuda:0'))
    def weight(self):return (materialize(self.a,self.b)+self.residual*self.gain)*(self.row*self.row_mult)[:,None]*(self.col*self.col_mult)[None,:]
    @torch.no_grad()
    def export(self):
        p=copy.deepcopy(self.original);p['a']=self.a.half().cpu();p['b']=self.b.half().cpu();p['row']=(self.row*self.row_mult).half().cpu();p['col']=(self.col*self.col_mult).half().cpu();p['scales']=(p['scales'].float()*self.gain.cpu()).half()
        if 'gain' in p:p['gain']=(p['gain'].float()*self.gain.cpu()).half()
        return p

class Bank(nn.Module):
    def __init__(self,up,down,router,topk):
        super().__init__();self.gate=nn.ModuleList([Matrix(p[0]) for p in up['parts']]);self.up=nn.ModuleList([Matrix(p[1]) for p in up['parts']]);self.down=nn.ModuleList([Matrix(p[0]) for p in down['parts']]);self.register_buffer('router',router.detach().float().clone());self.topk=topk
    def forward(self,x):
        values,indices=F.linear(x,self.router).topk(self.topk,-1);prob=values.softmax(-1);out=torch.zeros_like(x)
        for e,(g,u,d) in enumerate(zip(self.gate,self.up,self.down)):
            idx,slot=(indices==e).nonzero(as_tuple=True);xe=x[idx];h=F.silu(F.linear(xe,g.weight()))*F.linear(xe,u.weight());y=F.linear(h,d.weight());out.index_add_(0,idx,y*prob[idx,slot,None])
        return out
    @torch.no_grad()
    def packages(self,up,down):
        u={'mode':up['mode'],'shape':up['shape'],'parts':[[g.export(),a.export()] for g,a in zip(self.gate,self.up)]}
        d={'mode':down['mode'],'shape':down['shape'],'parts':[[a.export()] for a in self.down]}
        return u,d

@torch.no_grad()
def capture_inputs(model,module,tokens,sequences):
    captured=[]
    class Captured(Exception):pass
    def hook(mod,args):captured.append(args[0].reshape(-1,args[0].shape[-1]).detach().cpu().clone());raise Captured()
    handle=module.register_forward_pre_hook(hook)
    try:
        for n in range(sequences):
            try:model(tokens[n*128:(n+1)*128].unsqueeze(0).to('cuda:0'),use_cache=False)
            except Captured:pass
    finally:handle.remove()
    assert len(captured)==sequences
    return torch.cat(captured).to('cuda:0')

@torch.no_grad()
def metrics(pred,target):
    mse=float((pred-target).square().mean());variance=float(target.var(unbiased=False))
    return {'mse':mse,'target_variance':variance,'r_squared':1-mse/max(variance,1e-30),'cosine_mean':float(F.cosine_similarity(pred,target,dim=-1).mean()),'relative_l2':float((pred-target).norm()/target.norm().clamp_min(1e-20))}

def main(out,args):
    torch.set_num_threads(4)
    candidates=sorted((ROOT/'artifacts').glob('pretrained_*/packed_int2_sign/manifest.json'));source=next(p for p in candidates if 'granite' in json.loads(p.read_text())['model']);manifest=json.loads(source.read_text());original_manifest=copy.deepcopy(manifest)
    artifact=ROOT/'artifacts'/('distillation_'+out.name);artifact.mkdir(exist_ok=False);directory=artifact/'packed_int2_sign';directory.mkdir(exist_ok=False);shutil.copyfile(source.parent/'shared.pt',directory/'shared.pt')
    pretrained_result=next((p for p in sorted((PREFIX/'results').glob('*_pretrained_*/pretrained_results.json')) if json.loads(p.read_text())['model']==manifest['model']))
    original_result=json.loads(pretrained_result.read_text());baseline=original_result['baseline']['test'];tokenroot=pathlib.Path(original_result['results']['int2_sign']['artifact']).parent
    train=torch.load(tokenroot/'train.pt',weights_only=True);validation=torch.load(tokenroot/'validation.pt',weights_only=True)
    model=load_model(manifest['model'],manifest['revision']);layers=[];total_start=time.perf_counter();steps=args.steps or (2 if args.tier=='smoke' else 80);layer_count=1 if args.tier=='smoke' else model.config.num_hidden_layers
    for layer in range(layer_count):
        start=time.perf_counter();module=model.model.layers[layer].block_sparse_moe
        # All previous layers already contain rounded exported payload reconstructions.
        x=capture_inputs(model,module,train,128);xv=capture_inputs(model,module,validation,16)
        with torch.no_grad():target=module(x.unsqueeze(0)).reshape_as(x).detach();targetv=module(xv.unsqueeze(0)).reshape_as(xv).detach()
        up_row=next(r for r in manifest['parameters'] if f'.layers.{layer}.' in r['name'] and r['name'].endswith('gate_up_proj'));down_row=next(r for r in manifest['parameters'] if f'.layers.{layer}.' in r['name'] and r['name'].endswith('down_proj'))
        up=torch.load(source.parent/up_row['file'],weights_only=True);down=torch.load(source.parent/down_row['file'],weights_only=True);student=Bank(up,down,module.router.weight,model.config.num_experts_per_tok)
        with torch.no_grad():before=metrics(student(xv),targetv)
        opt=torch.optim.AdamW(student.parameters(),lr=.003,betas=(.9,.95),weight_decay=0);gen=torch.Generator(device='cuda:0').manual_seed(17+layer*1009);curve=[]
        for step in range(steps):
            idx=torch.randint(0,len(x),(256,),device='cuda:0',generator=gen);opt.zero_grad(set_to_none=True);pred=student(x[idx]);loss=(pred-target[idx]).square().mean()/target[idx].square().mean().clamp_min(1e-12);loss.backward();gn=torch.nn.utils.clip_grad_norm_(student.parameters(),1.);opt.step()
            with torch.no_grad():
                for matrix in list(student.gate)+list(student.up)+list(student.down):matrix.row_mult.clamp_(.2,5);matrix.col_mult.clamp_(.2,5);matrix.gain.clamp_(.2,5)
            if (step+1)%20==0 or step+1==steps:curve.append({'step':step+1,'normalized_training_mse':float(loss.detach()),'gradient_norm':float(gn)})
        with torch.no_grad():after_float=metrics(student(xv),targetv);new_up,new_down=student.packages(up,down)
        with torch.no_grad():
            module.experts.gate_up_proj.copy_(reconstruct(new_up));module.experts.down_proj.copy_(reconstruct(new_down));after_export=metrics(module(xv.unsqueeze(0)).reshape_as(xv),targetv)
        for row,package in [(up_row,new_up),(down_row,new_down)]:
            path=directory/row['file']
            with path.open('xb') as f:torch.save(package,f)
            row.update(sha256=digest(path),bytes=path.stat().st_size,tensor_bytes=tensor_bytes(package))
        result={'layer':layer,'calibration_input_scope':'actual compressed prefix; train128x128, held-out validation16x128','before':before,'after_float':after_float,'after_FP16_storage_reload_into_layer':after_export,'curve':curve,'seconds':time.perf_counter()-start,'parameters_trained':sum(p.numel() for p in student.parameters()),'seed':17+layer*1009}
        layers.append(result);write_json(out/f'layer_{layer:03d}.json',result);write_json(directory/f'partial_manifest_{layer:03d}.json',manifest);print(layer,before['r_squared'],after_export['r_squared'],flush=True)
        del student,opt,x,xv,target,targetv,new_up,new_down;torch.cuda.empty_cache()
    if args.tier=='smoke':
        write_json(out/'distillation_results.json',{'tier':'smoke','layers':layers,'full_model_evaluation':False});return
    manifest['distillation']={'steps_per_layer':steps,'lr':.003,'seed':17,'input_protocol':'sequential closed-loop train-only prefix inputs','trained_fields':'A/B, multiplicative row/column scales, scalar residual gain folded into existing residual scale and sign-gain fields; codes fixed','scope':'actual full gated/router-weighted expert bank, FP32 fitting; FP16 packed factor/scale export'}
    write_json(directory/'manifest.json',manifest);del model;torch.cuda.empty_cache();evpath=out/'fresh_reload_distilled.json'
    subprocess.run([sys.executable,'-B','-c','import sys;from pretrained import reload_evaluate;reload_evaluate(*sys.argv[1:])',str(directory),str(tokenroot/'test.pt'),str(evpath),'0'],check=True)
    ev=json.loads(evpath.read_text());params=sum(math.prod(r['shape']) for r in manifest['parameters']);total=sum(r['tensor_bytes'] for r in manifest['parameters'])+manifest['shared_tensor_bytes'];whole=json.loads((pretrained_result.parent/'architecture.json').read_text())['unique_parameter_count']
    write_json(out/'distillation_results.json',{'tier':'main','model':manifest['model'],'revision':manifest['revision'],'codec':'int2_sign_closed_loop_functional_fit','baseline_nll':baseline['nll'],'one_shot_nll':original_result['results']['int2_sign']['evaluation']['nll'],'distilled_nll':ev['nll'],'delta_to_original_nll':ev['nll']-baseline['nll'],'ppl_ratio_to_original':math.exp(ev['nll']-baseline['nll']),'block_bootstrap_delta':weighted_block_delta(baseline['rows'],ev['rows']),'layers':layers,'serialized_bytes_including_all_partial_manifests':sum(p.stat().st_size for p in directory.iterdir()),'tensor_bytes':total,'expert_bpw':8*sum(r['tensor_bytes'] for r in manifest['parameters'])/params,'whole_model_bpw':8*total/whole,'archive':str(directory),'archive_manifest_sha256':digest(directory/'manifest.json'),'elapsed_seconds':time.perf_counter()-total_start,'execution':'reference_decompressed','fresh_process_evaluation':True,'causal_ablation_limit':'independent one-shot versus closed-loop plus fitting changes two interventions; cannot isolate closed-loop effect without matched independently fitted control','not_run':['distilled direct-kernel runtime','downstream task accuracy','matched independent-distillation full-model control']})
