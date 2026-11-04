"""Controlled underdetermined regression; optimizer update geometry, not entropy rhetoric."""
import json,math,time,torch
from common import write_json,paired_ci,synchronize
from optimizers import RowAdamW,shrink
from math_codec import spectra,unfold,fold

def main(out,args):
    torch.set_num_threads(4);rows=[];curves=[]
    seeds=[11] if args.tier=='smoke' else [11,22,33,44,55]
    for seed in seeds:
        gen=torch.Generator(device='cuda:0').manual_seed(seed)
        # Known 16-dimensional input row space in R^64, condition number fixed.
        q=torch.linalg.qr(torch.randn(64,64,device='cuda:0',generator=gen,dtype=torch.float64)).Q
        x=torch.diag(torch.logspace(0,-1,16,device='cuda:0',dtype=torch.float64))@q[:,:16].T
        true=torch.randn(64,64,device='cuda:0',generator=gen,dtype=torch.float64)*.1;y=x@true.T;proj=q[:,:16]@q[:,:16].T;null=torch.eye(64,device='cuda:0',dtype=torch.float64)-proj
        for init in ['zero','random']:
            w0=torch.zeros_like(true) if init=='zero' else torch.randn(64,64,device='cuda:0',generator=gen,dtype=torch.float64)*.1
            for arm in ['sgd','adamw','row','tail','isotropic']:
                w=torch.nn.Parameter(w0.clone());lr=.03
                if arm=='sgd':opt=torch.optim.SGD([w],lr=lr,momentum=.9,weight_decay=0)
                elif arm=='row':opt=RowAdamW([w],[id(w)],lr=lr,weight_decay=0)
                else:opt=torch.optim.AdamW([w],lr=lr,weight_decay=0)
                sums={'gradient_row_norm':0.,'gradient_null_norm':0.,'update_row_norm':0.,'update_null_norm':0.,'optimizer_null_norm':0.,'intervention_null_norm':0.};qbuf=None
                # Identical deterministic batch sequence across arms, fixed controlled noise=0.
                for step in range(300):
                    idx=torch.tensor([(step*7+j)%16 for j in range(8)],device='cuda:0');loss=(x[idx]@w.T-y[idx]).square().mean();opt.zero_grad();loss.backward();g=w.grad.detach().clone();before=w.detach().clone();opt.step();after=w.detach().clone()
                    if arm=='tail':
                        # One expert bank; spectral structural regularization is a separate intervention.
                        qbuf=shrink(w.unsqueeze(0),qbuf,rank=2,decay=.01,p1=8,q1=8)
                    if arm=='isotropic':
                        # Isotropic decay control of declared fixed strength (not norm matched).
                        with torch.no_grad():w.mul_(.99)
                    delta=w.detach()-before
                    sums['gradient_row_norm']+=float((g@proj).norm());sums['gradient_null_norm']+=float((g@null).norm());sums['update_row_norm']+=float((delta@proj).norm());sums['update_null_norm']+=float((delta@null).norm());sums['optimizer_null_norm']+=float(((after-before)@null).norm());sums['intervention_null_norm']+=float(((w.detach()-after)@null).norm())
                    if (step+1)%50==0:curves.append({'seed':seed,'init':init,'arm':arm,'step':step+1,'loss':float(loss),'null_weight_norm':float((w@null).norm())})
                full=float((x@w.T-y).square().mean());diag=spectra(w.detach(),8,8,rank=2)
                # Permute coordinates and change valid factor shapes; neither preserves Kronecker rank.
                perm=torch.randperm(64,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed+99));permuted=spectra(w.detach()[perm][:,perm],8,8,2)
                alternative=spectra(w.detach(),4,16,2)
                row={'seed':seed,'initialization':init,'arm':arm,'loss':full,'weight_norm':float(w.norm()),'final_null_norm':float((w@null).norm()),**sums,**diag,'permuted_kron_stable_rank':permuted['kron_stable_rank'],'alternative_shape_kron_stable_rank':alternative['kron_stable_rank'],'steps':300,'noise':0,'dtype':'float64_parameters; RowAdamW FP32 states'}
                rows.append(row);print(seed,init,arm,full,sums['update_null_norm'],flush=True)
    summary={}
    for init in ['zero','random']:
        for arm in ['sgd','adamw','row','tail','isotropic']:
            a=[r for r in rows if r['initialization']==init and r['arm']==arm]
            summary[init+'_'+arm]={k:paired_ci([r[k] for r in a]) for k in ['loss','gradient_null_norm','update_null_norm','final_null_norm','histogram_body_bits']}
    write_json(out/'mechanism.json',{'rows':rows,'curves':curves,'summary':summary,'scope':'known exact linear input null space; no nonlinear null-space or universal entropy claim','limitations':['isotropic control fixed-strength, not norm-matched','no noisy-batch sweep in this first pass']})
