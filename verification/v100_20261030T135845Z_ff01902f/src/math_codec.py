"""Reference Kronecker codec. All storage and transform conventions are explicit."""
import math
import torch
import torch.nn.functional as F

def unfold(w, p1, q1):
    o,i=w.shape[-2:];assert o%p1==0 and i%q1==0
    return w.reshape(*w.shape[:-2],p1,o//p1,q1,i//q1).transpose(-3,-2).reshape(*w.shape[:-2],p1*q1,(o//p1)*(i//q1))

def fold(k,p1,p2,q1,q2):
    return k.reshape(*k.shape[:-2],p1,q1,p2,q2).transpose(-3,-2).reshape(*k.shape[:-2],p1*p2,q1*q2)

def materialize(a,b):
    return torch.einsum('...rij,...rkl->...ikjl',a,b).flatten(-4,-3).flatten(-2,-1)

def kron_forward(x,a,b,row=None,col=None):
    # x [...,q1*q2]; y[p1,p2] = sum_r A_r[p1,q1] X[q1,q2] B_r[p2,q2]^T.
    if col is not None: x=x*col
    temp=torch.einsum('rij,...jk->...rik',a,x.reshape(*x.shape[:-1],a.shape[-1],b.shape[-1]))
    y=torch.einsum('...rik,rpk->...ip',temp,b).flatten(-2,-1)
    return y if row is None else y*row

def fwht(x, block=None):
    n=x.shape[-1];block=n if block is None else block
    assert block>0 and block&(block-1)==0 and n%block==0
    shape=x.shape;y=x.reshape(*shape[:-1],n//block,block)
    h=1
    while h<block:
        z=y.reshape(*y.shape[:-1],block//(2*h),2,h)
        y=torch.stack((z[...,0,:]+z[...,1,:],z[...,0,:]-z[...,1,:]),-2).flatten(-3)
        h*=2
    return y.reshape(shape)/math.sqrt(block)

def rotate(w,block):
    return fwht(fwht(w,block).transpose(-1,-2),block).transpose(-1,-2)

def pack_codes(codes,bits):
    assert bits in (1,2,4,8)
    c=codes.flatten().to(torch.int64)
    if bool(((c<0)|(c>=(1<<bits))).any()): raise ValueError('invalid code')
    per=8//bits
    c=F.pad(c,(0,(-c.numel())%per)).reshape(-1,per)
    shifts=torch.arange(per,device=c.device)*bits
    return (c<<shifts).sum(-1).to(torch.uint8)

def unpack_codes(packed,bits,n):
    assert packed.dtype==torch.uint8 and bits in (1,2,4,8)
    if packed.numel() != (n*bits+7)//8: raise ValueError('packed length mismatch')
    shifts=torch.arange(8//bits,device=packed.device)*bits
    return ((packed.to(torch.int64)[:,None]>>shifts)&((1<<bits)-1)).flatten()[:n]

@torch.no_grad()
def fit_core(w,rank=8,p1=16,q1=16,balance=True,activation=None):
    w=w.float();o,i=w.shape
    padded=F.pad(w,(0,(-i)%q1,0,(-o)%p1))
    if activation is None: activation=torch.ones(padded.shape[-1],device=w.device)
    else: activation=F.pad(activation.float(),(0,padded.shape[-1]-i),value=1).clamp_min(1e-6)
    z=padded*activation
    row=z.square().mean(-1).sqrt().clamp_min(1e-8) if balance else torch.ones(z.shape[0],device=w.device)
    col=(z/row[:,None]).square().mean(0).sqrt().clamp_min(1e-8) if balance else torch.ones(z.shape[1],device=w.device)
    z=z/(row[:,None]*col)
    u,s,v=torch.linalg.svd(unfold(z,p1,q1),full_matrices=False)
    r=min(rank,len(s));rt=s[:r].sqrt()
    a=(u[:,:r].T*rt[:,None]).reshape(r,p1,q1)
    b=(v[:r]*rt[:,None]).reshape(r,z.shape[0]//p1,z.shape[1]//q1)
    return a,b,row,col/activation,z,activation

@torch.no_grad()
def encode(w,kind='ternary',rank=8,p1=16,q1=16,balance=True,rotation=True,activation=None):
    if not bool(w.isfinite().all()): raise ValueError('finite weights required')
    o,i=w.shape
    if kind=='int4':
        # Simple groupwise symmetric 4-bit baseline, 64 entries/FP16 scale.
        z=F.pad(w.float().flatten(),(0,(-w.numel())%64)).reshape(-1,64)
        scale=(z.abs().amax(-1)/7).clamp_min(torch.finfo(torch.float16).tiny).half()
        c=(z/scale.float()[:,None]).round().clamp(-7,7).long()+8
        return {'meta':{'version':1,'kind':kind,'shape':[o,i],'bits':4,'group':64},'codes':pack_codes(c,4),'scales':scale}
    a,b,row,col,z,act=fit_core(w,rank,p1,q1,balance,activation)
    # Storage rounding precedes residual fitting, avoiding hidden high-precision factors.
    a=a.half();b=b.half();row=row.clamp_min(torch.finfo(torch.float16).tiny).half();col=col.clamp_min(torch.finfo(torch.float16).tiny).half()
    # Work in stored scales so the residual represents the actual rounded payload.
    padded=F.pad(w.float(),(0,col.numel()-i,0,row.numel()-o))
    z=padded/(row.float()[:,None]*col.float()[None,:]).clamp_min(1e-20)
    core=materialize(a.float(),b.float())
    result={'meta':{'version':1,'kind':kind,'shape':[o,i],'padded_shape':list(z.shape),'block':16 if rotation else 1,'rank':a.shape[0],'bits':0},'a':a,'b':b,'row':row,'col':col}
    if kind=='pure': return result
    res=rotate(z-core,result['meta']['block'])
    if kind=='ternary': levels=torch.tensor([-1.224,0,1.224],device=w.device);bits=2
    elif kind in ('int2','int2_sign'): levels=torch.tensor([-1.51,-.453,.453,1.51],device=w.device);bits=2
    else: raise ValueError(kind)
    scale=res.square().mean(-1).sqrt().clamp_min(torch.finfo(torch.float16).tiny).half()
    codes=(res[:,:,None]/scale.float()[:,None,None]-levels).abs().argmin(-1)
    quant=levels[codes]*scale.float()[:,None]
    result.update(codes=pack_codes(codes,bits),scales=scale,levels=levels.half())
    result['meta']['bits']=bits
    if kind=='int2_sign':
        error=res-quant;gain=error.abs().mean(-1).half()
        result.update(sign_codes=pack_codes((error>=0).long(),1),gain=gain)
    return result

def residual_dense(payload):
    m=payload['meta'];o,i=m['padded_shape']
    codes=unpack_codes(payload['codes'],m['bits'],o*i).reshape(o,i)
    r=payload['levels'].float()[codes]*payload['scales'].float()[:,None]
    if 'sign_codes' in payload:
        r=r+(2*unpack_codes(payload['sign_codes'],1,o*i).reshape(o,i)-1)*payload['gain'].float()[:,None]
    return r

def decode(payload):
    m=payload['meta'];o,i=m['shape']
    if m['kind']=='int4':
        n=payload['scales'].numel()*64
        z=(unpack_codes(payload['codes'],4,n)-8).float().reshape(-1,64)*payload['scales'].float()[:,None]
        return z.flatten()[:o*i].reshape(o,i)
    w=materialize(payload['a'].float(),payload['b'].float())
    if m['bits']: w=w+rotate(residual_dense(payload),m['block'])
    return (w*payload['row'].float()[:,None]*payload['col'].float()[None,:])[:o,:i]

def reference_forward(x,payload):
    m=payload['meta'];o,i=m['shape']
    if m['kind']=='int4': return F.linear(x,decode(payload).to(x.dtype))
    x=F.pad(x.float(),(0,payload['col'].numel()-i))*payload['col'].float()
    y=kron_forward(x,payload['a'].float(),payload['b'].float())
    if m['bits']:
        # Explicit reference_decompressed fallback, never a direct packed kernel claim.
        y=y+fwht(F.linear(fwht(x,m['block']),residual_dense(payload)),m['block'])
    return (y*payload['row'].float())[...,:o]

def tensor_bytes(obj):
    seen=set();total=0
    def visit(x):
        nonlocal total
        if isinstance(x,torch.Tensor):
            s=x.untyped_storage();key=(str(x.device),s.data_ptr())
            if key not in seen: seen.add(key);total+=s.nbytes()
        elif isinstance(x,dict):
            for v in x.values(): visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x: visit(v)
    visit(obj);return total

def to_device(payload,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in payload.items()}

def spectra(w,p1=16,q1=16,rank=8):
    w=w.float();s=torch.linalg.svdvals(w);k=torch.linalg.svdvals(unfold(w,p1,q1));std=w.std().clamp_min(1e-12);z=w/std
    def entropy(a,lo,hi):
        h=torch.histc(a.clamp(lo,hi),256,lo,hi);p=h[h>0]/h.sum();return float(-(p*p.log2()).sum())
    return {'ordinary_stable_rank':float(s.square().sum()/s[0].square().clamp_min(1e-30)),
            'kron_stable_rank':float(k.square().sum()/k[0].square().clamp_min(1e-30)),
            'kron_energy_at_rank':float(k[:rank].square().sum()/k.square().sum().clamp_min(1e-30)),
            'histogram_body_bits':entropy(z,-3,3),'histogram_full_bits':entropy(w,float(w.min()),max(float(w.max()),float(w.min())+1e-12)),
            'body_clipped_fraction':float((z.abs()>3).float().mean()),'weight_std':float(std),'frobenius':float(w.norm())}
