import math,torch
from torch import nn
import torch.nn.functional as F
from math_codec import materialize

class MoE(nn.Module):
    def __init__(self,kind='adamw',d=256,e=16,rank=8):
        super().__init__();self.kind=kind;self.d=d;self.e=e;self.rank=rank
        self.router=nn.Linear(d,e,bias=False);self.override=None;self.stats={}
        if kind=='native':
            for direction in ['up','down']:
                for letter in ['a','b']:setattr(self,f'{letter}_{direction}',nn.Parameter(torch.randn(e,rank,16,16)*(1/(rank*d))**.25))
                setattr(self,f'row_{direction}',nn.Parameter(torch.ones(e,d)))
        elif kind=='lowrank':
            for direction in ['up','down']:
                setattr(self,f'a_{direction}',nn.Parameter(torch.randn(e,d,rank)*(1/(rank*d))**.25))
                setattr(self,f'b_{direction}',nn.Parameter(torch.randn(e,rank,d)*(1/(rank*d))**.25))
                setattr(self,f'row_{direction}',nn.Parameter(torch.ones(e,d)))
        else:
            self.w_up=nn.Parameter(torch.randn(e,d,d)/math.sqrt(d));self.w_down=nn.Parameter(torch.randn(e,d,d)/math.sqrt(d))

    def weights(self,direction):
        if self.override is not None:return self.override[direction]
        if self.kind=='native':w=materialize(getattr(self,'a_'+direction),getattr(self,'b_'+direction))
        elif self.kind=='lowrank':w=getattr(self,'a_'+direction)@getattr(self,'b_'+direction)
        else:return getattr(self,'w_'+direction)
        return w*getattr(self,'row_'+direction)[:,:,None]

    def forward(self,x):
        shape=x.shape;xf=x.reshape(-1,self.d);logits=self.router(xf);prob=F.softmax(logits.float(),-1)
        tp,ti=prob.topk(2,-1);tp=tp/tp.sum(-1,keepdim=True)
        out=torch.zeros_like(xf);wu=self.weights('up');wd=self.weights('down')
        # Combine top-2 memberships per expert; preserves the complete router/SiLU/up/down computation.
        for e in range(self.e):
            idx,slot=(ti==e).nonzero(as_tuple=True)
            xe=xf[idx];y=F.linear(F.silu(F.linear(xe,wu[e])),wd[e])
            out.index_add_(0,idx,(y*tp[idx,slot,None]).to(out.dtype))
        with torch.no_grad():
            counts=torch.bincount(ti.flatten(),minlength=self.e).float();self.stats={'counts':counts,'routing_entropy':-(prob*prob.clamp_min(1e-12).log()).sum(-1).mean(),'activation_rms':xf.detach().float().square().mean().sqrt()}
        return out.reshape(shape)

class Block(nn.Module):
    def __init__(self,kind,d,e,rank):
        super().__init__();self.ln1=nn.LayerNorm(d);self.attn=nn.MultiheadAttention(d,4,batch_first=True,dropout=0);self.ln2=nn.LayerNorm(d);self.moe=MoE(kind,d,e,rank)
    def forward(self,x,mask):
        h=self.ln1(x);x=x+self.attn(h,h,h,attn_mask=mask,need_weights=False)[0]
        return x+self.moe(self.ln2(x))

class Model(nn.Module):
    def __init__(self,kind='adamw',vocab=50257,d=256,layers=4,experts=16,rank=8,seq=128):
        super().__init__();self.vocab=vocab;self.seq=seq;self.kind=kind
        self.tok_emb=nn.Embedding(vocab,d);self.pos_emb=nn.Embedding(seq,d);self.blocks=nn.ModuleList([Block(kind,d,experts,rank) for _ in range(layers)]);self.ln=nn.LayerNorm(d)
        # Main-recipe embeddings use standard .02 initialization, distinct from legacy std=1.
        nn.init.normal_(self.tok_emb.weight,std=.02);nn.init.normal_(self.pos_emb.weight,std=.02)
        self.register_buffer('mask',torch.triu(torch.full((seq,seq),float('-inf')),diagonal=1))
    def forward(self,tokens):
        x=self.tok_emb(tokens)+self.pos_emb(torch.arange(tokens.shape[1],device=tokens.device))
        for block in self.blocks:x=block(x,self.mask[:tokens.shape[1],:tokens.shape[1]])
        return F.linear(self.ln(x),self.tok_emb.weight)

def make_paired(kind,seed,function_matched=False,rank=8):
    torch.manual_seed(seed);base=Model('adamw')
    if kind not in ['native','lowrank'] and not function_matched:return base
    torch.manual_seed(seed+100000);model=Model(kind,rank=rank)
    shared={k:v for k,v in base.state_dict().items() if '.moe.' not in k or '.router.' in k}
    model.load_state_dict(shared,strict=False)
    if function_matched and kind!='native':
        torch.manual_seed(seed+100000);native=Model('native',rank=rank)
        with torch.no_grad():
            for dst,src in zip(model.blocks,native.blocks):
                dst.moe.w_up.copy_(src.moe.weights('up'));dst.moe.w_down.copy_(src.moe.weights('down'))
    return model
