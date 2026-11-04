"""Bounded real-text teacher agreement/KL/routing/block errors, separate from accuracy."""
import json,pathlib,torch
import torch.nn.functional as F
from common import ROOT,PREFIX,write_json
from pretrained import identity_for,load_model
from downstream import archive_model

def capture(model):
    outputs={};routes={};handles=[]
    for name,module in model.named_modules():
        if name.endswith(('.block_sparse_moe','.mlp')):
            def hook(mod,args,result,name=name):outputs[name]=result.detach().float().cpu()
            handles.append(module.register_forward_hook(hook))
        elif name.endswith('.router'):
            def hook(mod,args,result,name=name):
                if isinstance(result,tuple) and len(result)==3:routes[name]=result[2].detach().cpu()
            handles.append(module.register_forward_hook(hook))
    return outputs,routes,handles

@torch.no_grad()
def main(out,args):
    torch.set_num_threads(4);name=args.model or 'ibm-granite/granite-3.1-1b-a400m-base';identity=identity_for(name);teacher=load_model(name,identity['revision']);original=next(json.loads(p.read_text()) for p in sorted((PREFIX/'results').glob('*_pretrained_*/pretrained_results.json')) if json.loads(p.read_text())['model']==name);tokenroot=pathlib.Path(original['results']['int4']['artifact']).parent;tokens=torch.load(tokenroot/'validation.pt',weights_only=True).to('cuda:0');variants=[('int4',pathlib.Path(original['results']['int4']['artifact'])),('int2_sign',pathlib.Path(original['results']['int2_sign']['artifact']))]
    if 'granite' in name:
        paths=sorted((ROOT/'artifacts').glob('distillation_*/packed_int2_sign/manifest.json'))
        if paths:variants.append(('distilled',paths[-1].parent))
    results=[]
    for codec,path in variants:
        student,manifest=archive_model(path);targets={r['name'] for r in manifest['parameters']};tparams=dict(teacher.named_parameters());sparams=dict(student.named_parameters());assert all(torch.equal(tparams[n],p) for n,p in sparams.items() if n not in targets),'untargeted tensors changed';tact,troute,th=capture(teacher);sact,sroute,sh=capture(student);rows=[]
        for block in range(8):
            ids=tokens[block*512:(block+1)*512].unsqueeze(0);t=teacher(ids,use_cache=False).logits[:,:-1].float();s=student(ids,use_cache=False).logits[:,:-1].float();labels=ids[:,1:];tp=t.argmax(-1);sp=s.argmax(-1);ttop=t.topk(5,-1).indices;stop=s.topk(5,-1).indices;tl=F.log_softmax(t,-1);sl=F.log_softmax(s,-1);count=labels.numel();blocks=[];routing=[]
            for layer,reference in tact.items():
                actual=sact[layer];mse=float((actual-reference).square().mean());variance=float(reference.var(unbiased=False));blocks.append({'layer':layer,'mse':mse,'r_squared':1-mse/max(variance,1e-30),'relative_l2':float((actual-reference).norm()/reference.norm().clamp_min(1e-20))})
            for layer,reference in troute.items():
                actual=sroute[layer];overlap=(reference[...,None]==actual[...,None,:]).any(-1).float();routing.append({'layer':layer,'topk_member_overlap':float(overlap.mean()),'exact_ordered_topk_fraction':float((reference==actual).all(-1).float().mean()),'teacher_expert_union':int(reference.unique().numel()),'student_expert_union':int(actual.unique().numel())})
            row={'block':block,'tokens_scored':count,'teacher_nll':float(F.cross_entropy(t.reshape(-1,t.shape[-1]),labels.reshape(-1))),'student_nll':float(F.cross_entropy(s.reshape(-1,s.shape[-1]),labels.reshape(-1))),'teacher_KL_to_student_per_token':float((tl.exp()*(tl-sl)).sum()/count),'top1_teacher_agreement':float((tp==sp).float().mean()),'teacher_top1_in_student_top5':float((stop==tp[...,None]).any(-1).float().mean()),'top5_set_member_overlap':float((ttop[...,None]==stop[...,None,:]).any(-1).float().mean()),'block_activations':blocks,'routing':routing};rows.append(row);tact.clear();sact.clear();troute.clear();sroute.clear()
        for handle in th+sh:handle.remove()
        results.append({'codec':codec,'rows':rows,'same_shared_tensor_values':True,'teacher_weights_loaded_separately':True,'archive':str(path),'precision':str(next(student.parameters()).dtype),'scope':'8 validation context512 blocks; teacher agreement/KL and actual-prefix block errors, not task accuracy or full-test substitute'});print(codec,{k:sum(r[k] for r in rows)/len(rows) for k in ['teacher_KL_to_student_per_token','top1_teacher_agreement','teacher_top1_in_student_top5']},flush=True);del student;torch.cuda.empty_cache()
    write_json(out/'quality_diagnostics.json',{'model':name,'revision':identity['revision'],'results':results,'diagnostic_memory_note':'contains original dense teacher plus decoded dense student; permitted within32GB and explicitly not compressed residency evidence','capture_overhead':'activation/routing D2H copies for diagnostics; not used in latency timing','test_fitting':False})
