import json,math,pathlib,subprocess,sys,time,torch
import torch.nn.functional as F
from common import ROOT,PREFIX,write_json,digest,synchronize,source_identity,gpu_guard
from data_audit import latest_data
from models import Model,make_paired
from optimizers import RowAdamW,shrink
from math_codec import encode,decode,to_device,spectra,tensor_bytes,materialize

def training_identity():
    identity=source_identity()
    dependencies={'src/training.py','src/models.py','src/math_codec.py','src/optimizers.py','src/common.py','src/data_audit.py'}
    identity['sources']={k:v for k,v in identity['sources'].items() if k in dependencies}
    return identity

@torch.no_grad()
def evaluate(model,tokens,limit=None,batch=16,seq=128):
    model.eval();losses=[];counts=[]
    # Each block scores its next seq tokens, without duplicate targets. Drop final incomplete block explicitly.
    n=(tokens.numel()-1)//seq
    if limit is not None:n=min(n,limit)
    for start in range(0,n,batch):
        b=min(batch,n-start);positions=torch.arange(start,start+b,device=tokens.device)[:,None]*seq+torch.arange(seq+1,device=tokens.device)[None,:];c=tokens[positions]
        with torch.autocast('cuda',dtype=torch.float16):logits=model(c[:,:-1])
        l=F.cross_entropy(logits.float().reshape(-1,model.vocab),c[:,1:].reshape(-1),reduction='none').reshape(b,seq).mean(-1)
        losses.extend(l.cpu().tolist());counts.extend([seq]*b)
    model.train();mean=sum(l*c for l,c in zip(losses,counts))/sum(counts)
    return {'nll':mean,'ppl':math.exp(mean),'block_nll':losses,'tokens_scored':sum(counts),'incomplete_tail_tokens':tokens.numel()-1-n*seq}

def export_model(model,artifact,codec):
    # Complete packed archive; non-expert tensors preserved exactly at current training dtype.
    shared={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if '.moe.' not in k or '.router.' in k}
    banks={};stats=[]
    for l,block in enumerate(model.blocks):
        for direction in ['up','down']:
            moe=block.moe;bank=[]
            for e,w in enumerate(moe.weights(direction).detach()):
                if codec=='native_direct':
                    bank.append({'a':getattr(moe,'a_'+direction)[e].detach().half().cpu(),'b':getattr(moe,'b_'+direction)[e].detach().half().cpu(),'row':getattr(moe,'row_'+direction)[e].detach().half().cpu(),'meta':{'native_direct':True}})
                else:bank.append(to_device(encode(w,codec),'cpu'))
                if e in [0,7,15]:stats.append({'layer':l,'direction':direction,'expert':e,**spectra(w)})
            banks[f'{l}.{direction}']=bank
    package={'kind':model.kind,'rank':model.blocks[0].moe.rank,'codec':codec,'shared':shared,'banks':banks}
    path=artifact/f'packed_{codec}.pt'
    with path.open('xb') as f:torch.save(package,f)
    dense_equivalent=sum(v.numel() for k,v in shared.items() if k!='mask' and v.is_floating_point())+4*2*16*256*256
    return {'path':str(path),'sha256':digest(path),'serialized_bytes':path.stat().st_size,'tensor_bytes':tensor_bytes(package),'expert_tensor_bytes':tensor_bytes(banks),'expert_bpw':tensor_bytes(banks)*8/(4*2*16*256*256),'whole_model_bpw_dense_equivalent':tensor_bytes(package)*8/dense_equivalent,'dense_equivalent_parameter_count':dense_equivalent,'spectra':stats}

def reload_eval(checkpoint,outpath,artifact):
    gpu_guard();package=torch.load(checkpoint,weights_only=True);model=Model(package['kind'],rank=package.get('rank',8)).to('cuda:0');model.load_state_dict(package['shared'],strict=False)
    # Native direct export keeps factors; generic quality path explicitly reconstructs dense matrices.
    for l,block in enumerate(model.blocks):
        override={}
        for direction in ['up','down']:
            bank=package['banks'][f'{l}.{direction}']
            if package['codec']=='native_direct':
                with torch.no_grad():
                    for e,p in enumerate(bank):
                        for letter in ['a','b','row']:getattr(block.moe,letter+'_'+direction)[e].copy_(p[letter].to('cuda:0'))
            else:override[direction]=torch.stack([decode(to_device(p,'cuda:0')) for p in bank])
        if override:block.moe.override=override
    tokens=torch.load(pathlib.Path(artifact)/'test.pt',weights_only=True).to('cuda:0');result=evaluate(model,tokens)
    result.update(codec=package['codec'],execution='native_factors_materialized_for_training_forward' if package['codec']=='native_direct' else 'reference_decompressed',fresh_process=True)
    write_json(outpath,result)

def main(out,args):
    torch.set_num_threads(4);kind=args.arm or 'adamw';seed=args.seed or 11;steps=args.steps or (10 if args.tier=='smoke' else 1500)
    data=latest_data();artifact=ROOT/'artifacts'/('training_'+out.name);artifact.mkdir(exist_ok=False)
    tokens={s:torch.load(pathlib.Path(data['artifact'])/(s+'.pt'),weights_only=True).to('cuda:0') for s in ['train','validation','test']}
    model=make_paired(kind,seed,function_matched=args.function_matched,rank=args.rank).to('cuda:0');expert=[p for name,p in model.named_parameters() if '.moe.w_' in name]
    optimizer=RowAdamW(model.parameters(),[id(p) for p in expert],lr=args.lr) if kind in ['row','row_tail'] else torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.05)
    scaler=torch.amp.GradScaler('cuda');batch_gen=torch.Generator(device='cuda:0').manual_seed(seed+200000);qr_gen=torch.Generator(device='cuda:0').manual_seed(seed+300000)
    starts=torch.randint(0,len(tokens['train'])-129,(max(6000,steps),16),device='cuda:0',generator=batch_gen);offsets=torch.arange(129,device='cuda:0');q={};successful=0;skipped=0;first_step=0;previous_elapsed=0.;torch.cuda.reset_peak_memory_stats();synchronize();begin=time.perf_counter();curve=[]
    if args.checkpoint:
        cp=torch.load(args.checkpoint,weights_only=True,map_location='cuda:0')
        for key in ['kind','seed']:
            assert cp[key]=={'kind':kind,'seed':seed}[key], 'checkpoint identity mismatch'
        assert cp['hyperparameters']=={'lr':args.lr,'rank':args.rank,'tail_decay':args.tail_decay,'function_matched':args.function_matched}, 'configuration changed on resume'
        current=training_identity()
        assert cp['identity']['base']==current['base'] and cp['identity']['plan_sha256']==current['plan_sha256']
        assert all(current['sources'].get(k)==v for k,v in cp['identity']['sources'].items()), 'source changed on resume'
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler']);qr_gen.set_state(cp['qr_generator'].cpu());q=cp['qr'];first_step=cp['step'];successful=cp.get('successful_updates',first_step);skipped=cp.get('skipped_updates',0);previous_elapsed=cp.get('elapsed_training_seconds',0.);curve=cp.get('curve',[])
    initial_hashes={n:digest_tensor(p) for n,p in model.named_parameters() if '.moe.' not in n or '.router.' in n}
    if args.checkpoint:initial_hashes=cp.get('initial_shared_hashes',initial_hashes)
    for step in range(first_step,steps):
        model.train();c=tokens['train'][starts[step,:,None]+offsets];optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(c[:,:-1]).reshape(-1,50257),c[:,1:].reshape(-1))
        scaler.scale(loss).backward();scaler.unscale_(optimizer);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        previous=scaler.get_scale();scaler.step(optimizer);scaler.update();ok=scaler.get_scale()>=previous
        if ok:
            successful+=1
            if kind=='legacy_blend':
                with torch.no_grad():
                    for p in expert:
                        v=optimizer.state[p]['exp_avg_sq'];v.copy_(.9*v.mean(-1,keepdim=True)+.1*v)
            if kind in ['tail','row_tail','legacy_blend']:
                decay=.06 if kind=='legacy_blend' else args.tail_decay*min(1,max(0,(step-300)/300))
                if decay:
                    for idx,p in enumerate(expert):q[idx]=shrink(p,q.get(idx),decay=decay,generator=qr_gen)
        else:skipped+=1
        if (step+1)%100==0 or step+1==steps:
            synchronize();elapsed=previous_elapsed+time.perf_counter()-begin
            stats=[{'counts':b.moe.stats['counts'].cpu().tolist(),'routing_entropy':float(b.moe.stats['routing_entropy']),'activation_rms':float(b.moe.stats['activation_rms'])} for b in model.blocks]
            row={'step':step+1,'training_nll_last_batch':float(loss),'wall_seconds_including_validation':elapsed,'tokens_consumed':(step+1)*2048,'successful_updates':successful,'skipped_updates':skipped,'gradient_norm':float(gn),'amp_scale':scaler.get_scale(),'routing':stats}
            if (step+1)%300==0 or step+1==steps:row['validation_selection_64_blocks']=evaluate(model,tokens['validation'],limit=64)
            curve.append(row)
            with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(kind,seed,step+1,float(loss),elapsed,flush=True)
        checkpoint_interval=1500 if steps>1500 else 500
        if (step+1)%checkpoint_interval==0 or step+1==steps:
            cp=artifact/f'checkpoint_{step+1:06d}.pt'
            with cp.open('xb') as f:torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step+1,'successful_updates':successful,'skipped_updates':skipped,'elapsed_training_seconds':previous_elapsed+time.perf_counter()-begin,'curve':curve,'initial_shared_hashes':initial_hashes,'batch_generator':batch_gen.get_state(),'qr_generator':qr_gen.get_state(),'qr':q,'kind':kind,'seed':seed,'identity':training_identity(),'hyperparameters':{'lr':args.lr,'rank':args.rank,'tail_decay':args.tail_decay,'function_matched':args.function_matched}},f)
            write_json(out/f'checkpoint_{step+1:06d}.json',{'path':str(cp),'sha256':digest(cp),'bytes':cp.stat().st_size})
    synchronize();train_seconds=previous_elapsed+time.perf_counter()-begin
    baseline={'validation':evaluate(model,tokens['validation'])}
    if not args.selection_only:baseline['test']=evaluate(model,tokens['test'])
    exports={}
    for codec in ([] if args.selection_only else (['native_direct','pure','ternary'] if kind=='native' else ['pure','ternary'])):
        exports[codec]=export_model(model,artifact,codec)
        resultpath=out/f'fresh_reload_{codec}.json'
        cmd=[sys.executable,'-B','-c','import sys;from training import reload_eval;reload_eval(*sys.argv[1:])',exports[codec]['path'],str(resultpath),data['artifact']]
        subprocess.run(cmd,check=True)
        exports[codec]['evaluation']=json.loads(resultpath.read_text())
    result={'seed':seed,'kind':kind,'steps':steps,'tokens_consumed':steps*2048,'successful_updates':successful,'skipped_updates':skipped,'training_seconds_including_periodic_validation_checkpointing':train_seconds,'parameters':sum(p.numel() for p in model.parameters()),'expert_parameters':sum(p.numel() for n,p in model.named_parameters() if '.moe.' in n and '.router.' not in n),'optimizer_tensor_bytes':tensor_bytes(optimizer.state_dict()),'initial_shared_hashes':initial_hashes,'baseline':baseline,'exports':exports,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'curve':curve,'tier':args.tier}
    write_json(out/'training_results.json',result)

def digest_tensor(p):
    import hashlib
    return hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
