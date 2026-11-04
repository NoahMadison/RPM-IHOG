"""Full zero-shot task evaluations through pinned lm-evaluation-harness tasks."""
import json,pathlib,time,torch
from common import PREFIX,ROOT,write_json,digest
from pretrained import identity_for,load_model,reconstruct

def archive_model(directory):
    from transformers import AutoConfig,AutoModelForCausalLM
    directory=pathlib.Path(directory);m=json.loads((directory/'manifest.json').read_text());cfg=AutoConfig.for_model(m['config']['model_type'],**{k:v for k,v in m['config'].items() if k!='model_type'});cfg._attn_implementation='eager'
    if 'granite' in m['model']:cfg._experts_implementation='eager'
    model=AutoModelForCausalLM.from_config(cfg,dtype=torch.float32 if 'granite' in m['model'] else torch.float16).to('cuda:0').eval();params=dict(model.named_parameters());loaded=set();assert digest(directory/'shared.pt')==m['shared_sha256'];shared=torch.load(directory/'shared.pt',weights_only=True)
    with torch.no_grad():
        for name,p in shared.items():params[name].copy_(p.to('cuda:0'));loaded.add(name)
        for row in m['parameters']:
            path=directory/row['file'];assert digest(path)==row['sha256'];p=torch.load(path,weights_only=True);params[row['name']].copy_(reconstruct(p).to(params[row['name']].dtype));loaded.add(row['name'])
    assert loaded==set(params),'incomplete packed model';return model,m

def scalar(x):
    if hasattr(x,'item'):return x.item()
    raise TypeError(type(x).__name__)

def main(out,args):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.task import ConfigurableTask
    from transformers import AutoTokenizer
    torch.set_num_threads(4);name=args.model or 'ibm-granite/granite-3.1-1b-a400m-base';identity=identity_for(name);task_records=json.loads(sorted((PREFIX/'configs').glob('downstream_sources_*.json'))[-1].read_text());tasks=[ConfigurableTask(config=r['config']) for r in task_records if r['status']=='available'];blocked=[r for r in task_records if r['status']!='available']
    if not tasks:write_json(out/'downstream_results.json',{'model':name,'status':'blocked','sources':task_records});return
    if args.codec=='original':model=load_model(name,identity['revision']);manifest=None
    else:
        if args.codec=='distilled':paths=sorted((ROOT/'artifacts').glob('distillation_*/packed_int2_sign/manifest.json'))
        else:paths=[p for p in sorted((ROOT/'artifacts').glob('pretrained_*/packed_'+args.codec+'/manifest.json')) if json.loads(p.read_text())['model']==name]
        if not paths:raise RuntimeError('no complete fidelity archive for requested downstream variant')
        model,manifest=archive_model(paths[-1].parent)
    tokenizer=AutoTokenizer.from_pretrained(name,revision=identity['revision'],local_files_only=True);artifact=ROOT/'artifacts'/('downstream_'+out.name);artifact.mkdir(exist_ok=False)
    lm=HFLM(pretrained=model,tokenizer=tokenizer,backend='causal',device='cuda:0',batch_size=8,max_length=512,softmax_dtype=torch.float32,offload_folder=str(artifact/'offload'),logits_cache=True)
    start=time.perf_counter();result=lm_eval.simple_evaluate(model=lm,tasks=tasks,num_fewshot=0,limit=4 if args.tier=='smoke' else None,bootstrap_iters=2000,log_samples=True,apply_chat_template=False,random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
    # Preserve scores/decisions only. Dataset text/prompt bodies remain unlogged/unpublished.
    decisions={}
    for task,samples in result.get('samples',{}).items():
        decisions[task]=[{'doc_id':s['doc_id'],'doc_hash':s.get('doc_hash'),'acc':s.get('acc'),'acc_norm':s.get('acc_norm'),'filtered_resps':s.get('filtered_resps')} for s in samples]
    output={'model':name,'revision':identity['revision'],'codec':args.codec,'tier':args.tier,'status':'completed','results':result['results'],'n_samples':result.get('n-samples'),'decisions':decisions,'sources':task_records,'blocked_tasks':blocked,'harness_version':lm_eval.__version__,'zero_shot':True,'max_length':512,'batch_size':8,'seconds':time.perf_counter()-start,'execution':'original' if args.codec=='original' else 'reference_decompressed','archive_manifest_sha256':digest(paths[-1]) if manifest else None}
    output=json.loads(json.dumps(output,default=scalar));write_json(out/'downstream_results.json',output);print(json.dumps(output['results']),flush=True)
