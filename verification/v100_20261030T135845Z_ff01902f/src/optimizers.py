import torch
from math_codec import unfold,fold

class RowAdamW(torch.optim.Optimizer):
    """FP32 m; FP32 genuinely pooled v for selected expert matrices; AdamW elsewhere."""
    def __init__(self,params,expert_ids=(),lr=3e-4,betas=(.9,.95),eps=1e-8,weight_decay=.05):
        self.expert_ids=set(expert_ids)
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))

    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            b1,b2=group['betas']
            for p in group['params']:
                if p.grad is None: continue
                g=p.grad.float();s=self.state[p]
                if not s:
                    s['step']=0;s['m']=torch.zeros_like(p,dtype=torch.float32)
                    s['v']=torch.zeros_like(p[...,:1] if id(p) in self.expert_ids else p,dtype=torch.float32)
                s['step']+=1;t=s['step'];s['m'].lerp_(g,1-b1)
                g2=g.square().mean(-1,keepdim=True) if id(p) in self.expert_ids else g.square()
                s['v'].lerp_(g2,1-b2)
                p.mul_(1-group['lr']*group['weight_decay'])
                p.addcdiv_(s['m']/(1-b1**t),(s['v']/(1-b2**t)).sqrt().add_(group['eps']),value=-group['lr'])

@torch.no_grad()
def shrink(w,q=None,rank=8,decay=.003,p1=16,q1=16,generator=None):
    o,i=w.shape[-2:];k=unfold(w.float(),p1,q1)
    if q is None:
        q=torch.randn(*k.shape[:-2],k.shape[-1],rank,device=w.device,dtype=torch.float32,generator=generator)
        q=torch.linalg.qr(q).Q
    q=torch.linalg.qr(k.transpose(-1,-2)@(k@q)).Q
    proj=(k@q)@q.transpose(-1,-2)
    w.copy_(fold(proj+(1-decay)*(k-proj),p1,o//p1,q1,i//q1).to(w.dtype))
    return q
