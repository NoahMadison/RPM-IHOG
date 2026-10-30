"""Real full-model quality with honest reference_decompressed execution and fresh reload."""
import hashlib,json,math,pathlib,subprocess,sys,time,torch
import torch.nn.functional as F
from common import ROOT,PREFIX,write_json,digest,gpu_guard,block_bootstrap,synchronize
from math_codec import encode,decode,to_device,tensor_bytes
from data_audit import latest_data

def identity_for(name):
    rows=json.loads(sorted((PREFIX/'results').glob('*_data_*/model_identities.json'))[-1].read_text())
    return next(r for r in rows if r['id']==name)

def load_model(name,revision):
    from transformers import AutoModelForCausalLM
    kw={'dtype':torch.float32 if 'granite' in name else torch.float16,'attn_implementation':'eager','local_files_only':True}
    if 'granite' in name:kw['experts_implementation']='eager'
    return AutoModelForCausalLM.from_pretrained(name,revision=revision,**kw).to('cuda:0').eval()

def get_tokens(name,revision,artifact):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    data=latest_data();ds=load_dataset(data['dataset'],data['configuration'],revision=data['revision']);tok=AutoTokenizer.from_pretrained(name,revision=revision,local_files_only=True)
    metadata={'tokenizer':name,'revision':revision,'splits':{}}
    for split in ['train','validation','test']:
        text='\n\n'.join(ds[split]['text']);tokens=torch.tensor(tok.encode(text,add_special_tokens=False),dtype=torch.long)
        p=artifact/(split+'.pt')
        with p.open('xb') as f:torch.save(tokens,f)
        metadata['splits'][split]={'tokens':len(tokens),'sha256':digest(p),'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
    return metadata

@torch.no_grad()
def evaluate(model,tokens,limit=None,teacher=None):
    # Nonoverlapping context blocks. First target per block is excluded; identical source targets for every variant.
    n=(len(tokens)+511)//512
    if limit is not None:n=min(n,limit)
    rows=[];top1=0;top5=0;kl=0.;teacher_count=0
    for block in range(n):
        c=tokens[block*512:min((block+1)*512,len(tokens))].unsqueeze(0).to('cuda:0')
        if c.shape[1]<2:continue
        logits=model(c,use_cache=False).logits[:,:-1].float();labels=c[:,1:]
        if not bool(logits.isfinite().all()):raise FloatingPointError(f'nonfinite untouched/decoded model logits at block {block}; reject this precision/configuration')
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='sum')
        rows.append({'block':block,'nll_sum':float(loss),'count':labels.numel()})
        if teacher is not None and block<8:
            # Stored calibration-independent teacher logits for fixed validation blocks only.
            target=teacher[block].to(logits.device).float();tp=target.argmax(-1)
            top1+=int((logits.argmax(-1)==tp).sum());top5+=int((logits.topk(5,-1).indices==tp[...,None]).any(-1).sum());teacher_count+=tp.numel()
            kl+=float(F.kl_div(F.log_softmax(logits,-1),F.softmax(target,-1),reduction='sum'))
        if (block+1)%100==0:print('eval',block+1,'/',n,flush=True)
    count=sum(r['count'] for r in rows);mean=sum(r['nll_sum'] for r in rows)/count
    result={'nll':mean,'ppl':math.exp(min(mean,700)),'tokens_scored':count,'rows':rows,'protocol':'512-token disjoint context blocks; first token per block not scored; eager attention; FP32 cross entropy','weight_dtype':str(next(model.parameters()).dtype)}
    if teacher_count:result['teacher_validation_diagnostics']={'top1_agreement':top1/teacher_count,'top5_agreement':top5/teacher_count,'kl_per_token':kl/teacher_count,'tokens':teacher_count}
    return result

def targets(model):
    return [(n,p) for n,p in model.named_parameters() if ('.experts.' in n and n.endswith(('gate_up_proj','down_proj'))) or ('.mlp.' in n and any(n.endswith(x+'.weight') for x in ['gate_proj','up_proj','down_proj']))]

@torch.no_grad()
def calibrate(model,tokens):
    stats={};hooks=[]
    for name,module in model.named_modules():
        if name.endswith('.experts') and hasattr(module,'gate_up_proj'):
            e,d=module.num_experts,module.hidden_dim
            stats[name]={'sum':torch.zeros(e,d,device='cuda:0'),'count':torch.zeros(e,device='cuda:0',dtype=torch.long)}
            def hook(mod,args,name=name):
                x,ti,tp=args;st=stats[name]
                for expert in range(mod.num_experts):
                    idx=(ti==expert).any(-1);st['sum'][expert]+=x[idx].float().square().sum(0);st['count'][expert]+=idx.sum()
            hooks.append(module.register_forward_pre_hook(hook))
        elif any(name.endswith('.mlp.'+x) for x in ['gate_proj','up_proj','down_proj']):
            stats[name]={'sum':torch.zeros(module.in_features,device='cuda:0'),'count':torch.zeros((),device='cuda:0',dtype=torch.long)}
            def hook(mod,args,name=name):
                x=args[0].reshape(-1,args[0].shape[-1]).float();stats[name]['sum']+=x.square().sum(0);stats[name]['count']+=x.shape[0]
            hooks.append(module.register_forward_pre_hook(hook))
    for i in range(128):model(tokens[i*128:(i+1)*128].unsqueeze(0).to('cuda:0'),use_cache=False)
    for hook in hooks:hook.remove()
    public={};rms={}
    for name,st in stats.items():
        count=st['count'];denom=count.clamp_min(1).float();mean=st['sum']/(denom[:,None] if count.ndim else denom)
        # Cold experts explicitly use an unweighted fallback rather than invented activations.
        r=mean.sqrt().clamp_min(1e-6)
        if count.ndim:r[count==0]=1
        public[name]={'token_coverage':count.cpu().tolist(),'cold_experts':int((count==0).sum()),'rms_min':float(r.min()),'rms_max':float(r.max())};rms[name]=r
    return rms,public

@torch.no_grad()
def compress_parameter(name,w,codec,rms):
    if w.ndim==2:
        p=encode(w,codec,activation=rms.get(name.rsplit('.',1)[0]));return {'parts':[to_device(p,'cpu')],'shape':list(w.shape),'mode':'matrix'}
    packages=[];activation=rms.get(name.rsplit('.',1)[0])
    for expert,matrix in enumerate(w):
        act=activation[expert] if activation is not None and name.endswith('gate_up_proj') else None
        if name.endswith('gate_up_proj'):
            # Independently encode complete gate and up matrices; preserve SiLU(gate)*up downstream.
            packages.append([to_device(encode(part,codec,activation=act),'cpu') for part in matrix.chunk(2,0)])
        else:packages.append([to_device(encode(matrix,codec),'cpu')])
    return {'parts':packages,'shape':list(w.shape),'mode':'bank'}

def reconstruct(package,device='cuda:0'):
    if package['mode']=='matrix':return decode(to_device(package['parts'][0],device))
    return torch.stack([torch.cat([decode(to_device(p,device)) for p in parts],0) for parts in package['parts']])

@torch.no_grad()
def main(out,args):
    torch.set_num_threads(4);name=args.model or 'ibm-granite/granite-3.1-1b-a400m-base';identity=identity_for(name)
    if identity['status']!='available':raise RuntimeError('verified model source unavailable')
    artifact=ROOT/'artifacts'/('pretrained_'+out.name);artifact.mkdir(exist_ok=False);start=time.perf_counter();model=load_model(name,identity['revision']);synchronize();load_seconds=time.perf_counter()-start
    tokenmeta=get_tokens(name,identity['revision'],artifact);tokens={s:torch.load(artifact/(s+'.pt'),weights_only=True) for s in ['train','validation','test']};ts=targets(model)
    if not ts:raise RuntimeError('no expert/MLP parameters discovered; do not silently skip conversion')
    # Store only run-owned CPU originals for restoring bounded validation pilots; never in the packed archive.
    original={n:p.detach().cpu().clone() for n,p in ts};names=set(original);shared={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n not in names}
    write_json(out/'architecture.json',{'config':model.config.to_dict(),'model_class':type(model).__name__,'target_parameters':[{'name':n,'shape':list(p.shape)} for n,p in ts],'unique_parameter_count':sum(p.numel() for p in model.parameters()),'expert_or_mlp_parameters':sum(p.numel() for _,p in ts),'load_seconds':load_seconds,'tokenizer':tokenmeta,'execution_backend':'FP16 eager attention/eager experts'})
    limit=8 if args.tier=='smoke' else None
    baseline={'validation':evaluate(model,tokens['validation'],limit=8),'test':evaluate(model,tokens['test'],limit=limit)};write_json(out/'baseline.json',baseline)
    rms,coverage=calibrate(model,tokens['train']);write_json(out/'calibration_coverage.json',coverage)
    # Bounded first-layer validation pilot chooses between independently specified residual codecs.
    first=[(n,p) for n,p in ts if '.layers.0.' in n];pilots=[]
    for codec in ['pure','ternary','int2','int2_sign','int4']:
        for n,p in first:p.copy_(reconstruct(compress_parameter(n,original[n].to('cuda:0'),codec,rms)).to(p.dtype))
        ev=evaluate(model,tokens['validation'],limit=8);pilots.append({'codec':codec,'validation':ev})
        for n,p in first:p.copy_(original[n].to('cuda:0'))
    write_json(out/'validation_pilots.json',pilots)
    # Keep a fixed INT4 baseline and choose the best sub-4-bit candidate from validation only.
    best=min([p for p in pilots if p['codec']!='int4'],key=lambda p:p['validation']['nll'])['codec']
    selected=[best,'int4'] if args.codec=='auto' else [args.codec]
    results={}
    for codec in selected:
        directory=artifact/('packed_'+codec);directory.mkdir(exist_ok=False)
        with (directory/'shared.pt').open('xb') as f:torch.save(shared,f)
        manifest={'model':name,'revision':identity['revision'],'config':model.config.to_dict(),'codec':codec,'parameters':[],'shared_sha256':digest(directory/'shared.pt'),'shared_bytes':(directory/'shared.pt').stat().st_size,'shared_tensor_bytes':tensor_bytes(shared),'representation':'packed archive; full model fidelity uses reference_decompressed GPU execution'};total_tensor=0;encode_start=time.perf_counter()
        for idx,(n,p) in enumerate(ts):
            package=compress_parameter(n,original[n].to('cuda:0'),codec,rms);path=directory/f'parameter_{idx:04d}.pt'
            with path.open('xb') as f:torch.save(package,f)
            size=tensor_bytes(package);total_tensor+=size
            manifest['parameters'].append({'name':n,'shape':list(p.shape),'file':path.name,'sha256':digest(path),'bytes':path.stat().st_size,'tensor_bytes':size})
            # Do not retain a second GPU dense copy. Full conversion is later loaded in a fresh process.
            if (idx+1)%12==0:print('encoded',codec,idx+1,'/',len(ts),flush=True)
        write_json(directory/'manifest.json',manifest)
        output=out/f'fresh_reload_{codec}.json';del package
        model.to('cpu');torch.cuda.empty_cache()
        cmd=[sys.executable,'-B','-c','import sys;from pretrained import reload_evaluate;reload_evaluate(*sys.argv[1:])',str(directory),str(artifact/'test.pt'),str(output),str(limit or 0)]
        subprocess.run(cmd,check=True)
        ev=json.loads(output.read_text());delta=ev['nll']-baseline['test']['nll'];blocks=[r['nll_sum']/r['count']-b['nll_sum']/b['count'] for r,b in zip(ev['rows'],baseline['test']['rows'])]
        serialized=sum(p.stat().st_size for p in directory.iterdir());params=sum(p.numel() for _,p in ts);whole=sum(p.numel() for p in model.parameters())
        results[codec]={'evaluation':ev,'delta_nll':delta,'ppl_ratio':math.exp(delta),'block_bootstrap_delta':block_bootstrap(blocks),'serialized_bytes':serialized,'tensor_bytes':total_tensor+tensor_bytes(shared),'expert_or_mlp_bpw':8*total_tensor/params,'whole_model_bpw':8*(total_tensor+tensor_bytes(shared))/whole,'archive_sha256':digest(directory/'manifest.json'),'artifact':str(directory),'encoding_seconds_including_reload':time.perf_counter()-encode_start,'execution':'reference_decompressed','direct_compressed_residency_claim':False}
        model.to('cuda:0')
    write_json(out/'pretrained_results.json',{'model':name,'revision':identity['revision'],'baseline':baseline,'selected_by_validation':selected,'results':results,'tier':args.tier,'calibration':'128 train-only sequences of length 128; independent calibration; Granite down-projection weighting omitted and explicitly unweighted','not_run':['closed-loop/distillation ablations','full downstream lm-evaluation-harness tasks','second domain','matched-byte low-rank full model']})

def reload_evaluate(directory,tokens_path,outpath,limit='0'):
    gpu_guard();from transformers import AutoConfig,AutoModelForCausalLM
    directory=pathlib.Path(directory);m=json.loads((directory/'manifest.json').read_text());cfg=AutoConfig.for_model(m['config']['model_type'],**{k:v for k,v in m['config'].items() if k!='model_type'});cfg._attn_implementation='eager'
    if 'granite' in m['model']:cfg._experts_implementation='eager'
    # Config-only random allocation: original model weights are never loaded/consulted in this process.
    model=AutoModelForCausalLM.from_config(cfg,dtype=torch.float32 if 'granite' in m['model'] else torch.float16).to('cuda:0').eval();named=dict(model.named_parameters());assert digest(directory/'shared.pt')==m['shared_sha256'];shared=torch.load(directory/'shared.pt',weights_only=True)
    loaded=set()
    with torch.no_grad():
        for name,p in shared.items():named[name].copy_(p.to('cuda:0'));loaded.add(name)
        for row in m['parameters']:
            path=directory/row['file'];assert digest(path)==row['sha256'];package=torch.load(path,weights_only=True);w=reconstruct(package);assert list(w.shape)==row['shape'];named[row['name']].copy_(w.to(named[row['name']].dtype));loaded.add(row['name'])
    assert loaded==set(named), 'archive omitted model parameters'
    torch.cuda.reset_peak_memory_stats();tokens=torch.load(tokens_path,weights_only=True);ev=evaluate(model,tokens,limit=int(limit) or None)
    ev.update(fresh_process=True,source_weights_loaded=False,all_unique_parameters_loaded=len(loaded),gpu_allocated=torch.cuda.memory_allocated(),gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),execution='reference_decompressed')
    write_json(outpath,ev)
