"""Future-environment DDP launcher; requires distinct authorized GPU UUIDs.

Example: RPM_GPU_UUIDS=GPU-...,GPU-... torchrun --standalone --nproc_per_node=2
  multigpu_launcher.py --method native --out TASK_ROOT/artifacts/NEW_OUTPUT
Run dense, native, powersgd, candidate with identical --steps/global-batch/seed.
This file is a launcher, not evidence of multi-GPU execution.
"""
import argparse,hashlib,json,os,pathlib,time,subprocess
import torch,torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState,powerSGD_hook
from common import ROOT,write_json
from models import make_paired
from data_audit import latest_data
from math_codec import unfold,fold

class Candidate:
    def __init__(self,model):
        self.names={id(p):n for n,p in model.named_parameters()};self.state={};self.rounds=0;self.logical_bytes=0;self.refresh_interval=100

def candidate_hook(state,bucket):
    """Shared-basis K projection. Full gradient refresh and explicit basis broadcast counted; dense fallback elsewhere.

    Synchronous reference hook deliberately measures lost overlap. Coefficients are
    numeric FP32; workers share refreshed bases. Per-worker error feedback stays local.
    """
    world=dist.get_world_size()
    for p,g in zip(bucket.parameters(),bucket.gradients()):
        name=state.names[id(p)];st=state.state.setdefault(name,{'steps':0,'error':torch.zeros_like(g)})
        if g.ndim==3 and g.shape[-2:]==(256,256):
            corrected=g+st['error'];k=unfold(corrected,16,16)
            if st['steps']%state.refresh_interval==0:
                mean=k.clone();dist.all_reduce(mean);mean/=world;state.rounds+=1;state.logical_bytes+=mean.numel()*4
                # One identical random probe and QR of globally averaged gradient.
                seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')+st['steps'];gen=torch.Generator(device=g.device).manual_seed(seed)
                probe=torch.randn(k.shape[0],k.shape[-1],8,device=g.device,generator=gen)
                basis=torch.linalg.qr(mean@probe).Q.contiguous() if dist.get_rank()==0 else torch.empty(k.shape[0],k.shape[-2],8,device=g.device,dtype=g.dtype)
                dist.broadcast(basis,src=0);st['basis']=basis;state.rounds+=1;state.logical_bytes+=basis.numel()*basis.element_size()
            basis=st['basis'];c=basis.transpose(-1,-2)@k;local=basis@c;st['error'].copy_(corrected-fold(local,16,16,16,16));dist.all_reduce(c);c/=world;state.rounds+=1;state.logical_bytes+=c.numel()*4;g.copy_(fold(basis@c,16,16,16,16))
        else:
            dist.all_reduce(g);g/=world;state.rounds+=1;state.logical_bytes+=g.numel()*g.element_size()
        st['steps']+=1
    fut=torch.futures.Future();fut.set_result(bucket.buffer());return fut

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['dense','native','powersgd','candidate'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=6000);ap.add_argument('--global-batch',type=int,default=16);ap.add_argument('--seed',type=int,default=11);ap.add_argument('--target-nll',type=float,default=7.0);a=ap.parse_args()
    uuids=os.environ.get('RPM_GPU_UUIDS','').split(',');local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);local_world=int(os.environ.get('LOCAL_WORLD_SIZE',world));assert world>=2 and len(uuids)>=local_world and len(set(uuids))==len(uuids),'need distinct explicitly authorized UUIDs on each host'
    # CUDA is not initialized until local visibility is set. Never use duplicate-device ranks.
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,memory.total,compute_cap','--format=csv,noheader,nounits'],text=True)
    expected=next(row.split(', ') for row in inventory.splitlines() if row.split(', ')[2]==uuids[local])
    os.environ['CUDA_VISIBLE_DEVICES']=uuids[local];torch.cuda.set_device(0);prop=torch.cuda.get_device_properties(0);assert str(prop.uuid).replace('GPU-','')==uuids[local].replace('GPU-','')
    assert prop.name==expected[1] and f'{prop.major}.{prop.minor}'==expected[5]
    assert abs(prop.total_memory/2**20-float(expected[4]))<512
    actual_pci=f'{prop.pci_domain_id:08x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0';assert actual_pci.lower()==expected[3].lower()
    assert a.global_batch%world==0
    dist.init_process_group('nccl');identities=[None]*world;dist.all_gather_object(identities,str(prop.uuid));assert len(set(identities))==world,'duplicate physical GPU detected across ranks'
    rank=dist.get_rank();out=pathlib.Path(a.out).resolve()
    assert out.is_relative_to(ROOT),'all outputs must remain in the task root'
    if local==0:out.mkdir(parents=True,exist_ok=False) # host-local or host-unique output path required
    dist.barrier()
    from lineage import training_data_guard
    data_identity=training_data_guard()
    from training import training_identity
    canonical={'settings':{key:value for key,value in vars(a).items() if key!='out'},'data':data_identity['identity'],'code':training_identity(),'launcher_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};fingerprint=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest();fingerprints=[None]*world;dist.all_gather_object(fingerprints,fingerprint);assert len(set(fingerprints))==1,'rank code/data/configuration mismatch'
    write_json(out/f'rank_{rank:04d}_identity.json',{'rank':rank,'world_size':world,'inventory_before_mask':inventory,'selected_uuid':uuids[local],'torch_properties':str(prop),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'data_identity':data_identity,'command_arguments':vars(a),'precision':'FP32 training; common AMP evaluation','method_scope':'candidate compresses eligible expert banks, PowerSGD uses its standard eligible-parameter policy; compare actual wholepayload/quality rather than assuming matched totalbytes'})
    kind='native' if a.method=='native' else 'adamw';model=make_paired(kind,a.seed).to('cuda:0');model=DDP(model,device_ids=[0]);hook_state=None
    if a.method=='powersgd':model.register_comm_hook(PowerSGDState(process_group=dist.group.WORLD,matrix_approximation_rank=8,start_powerSGD_iter=10,use_error_feedback=True,warm_start=True),powerSGD_hook)
    elif a.method=='candidate':hook_state=Candidate(model.module);model.register_comm_hook(hook_state,candidate_hook)
    # Identical FP32 training across methods keeps local error feedback independent
    # of dynamic loss-scale changes. AMP requires a separately validated rescaling protocol.
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.05);data=latest_data();tokens=torch.load(pathlib.Path(data['artifact'])/'train.pt',weights_only=True).to('cuda:0');val=torch.load(pathlib.Path(data['artifact'])/'validation.pt',weights_only=True).to('cuda:0');gen=torch.Generator().manual_seed(a.seed+200000);local_batch=a.global_batch//world;torch.cuda.synchronize();dist.barrier();t0=time.perf_counter();reached=None
    for step in range(a.steps):
        starts=torch.randint(0,len(tokens)-129,(a.global_batch,),generator=gen)[rank*local_batch:(rank+1)*local_batch].to('cuda:0');chunk=tokens[starts[:,None]+torch.arange(129,device='cuda:0')];opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(chunk[:,:-1]).reshape(-1,50257),chunk[:,1:].reshape(-1));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%300==0:
            from training import evaluate
            ev=evaluate(model.module,val,limit=64);value=torch.tensor(ev['nll'],device='cuda:0');dist.all_reduce(value);value/=world;torch.cuda.synchronize();row={'step':step+1,'validation_nll':float(value),'seconds':time.perf_counter()-t0,'world':world,'method':a.method}
            if reached is None and float(value)<=a.target_nll:reached=row['seconds']
            if rank==0:
                with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if rank==0:write_json(out/'result.json',{'method':a.method,'world_size':world,'target_nll':a.target_nll,'time_to_quality_seconds':reached,'target_status':'reached' if reached is not None else 'target_not_reached','hook_rounds':getattr(hook_state,'rounds',None),'hook_logical_payload_bytes':getattr(hook_state,'logical_bytes',None),'note':'actual ring/network traffic requires profiler; no host network changes; same global batches and quality target; candidate reference loses overlap'})
    dist.destroy_process_group()
if __name__=='__main__':main()
