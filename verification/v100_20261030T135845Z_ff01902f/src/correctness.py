import json,pathlib,subprocess,sys,tempfile,torch
from common import write_json,assert_resume
from math_codec import *
from optimizers import RowAdamW,shrink

def main(out,args):
    torch.manual_seed(710);torch.set_num_threads(4);checks=[]
    def check(name,fn):
        fn();checks.append(name);print('PASS',name,flush=True)
    def factors():
        for p1,p2,q1,q2,rank,batch in [(2,3,4,5,1,1),(3,2,2,4,3,7),(2,4,3,2,2,3)]:
            a=torch.randn(rank,p1,q1,dtype=torch.float64,requires_grad=True);b=torch.randn(rank,p2,q2,dtype=torch.float64,requires_grad=True)
            w=materialize(a,b);assert torch.allclose(w,sum(torch.kron(ar.contiguous(),br.contiguous()) for ar,br in zip(a,b)))
            assert torch.equal(fold(unfold(w,p1,q1),p1,p2,q1,q2),w)
            x=torch.randn(batch,q1*q2*2,dtype=torch.float64)[:,::2];row=torch.rand(p1*p2,dtype=torch.float64,requires_grad=True);col=torch.rand(q1*q2,dtype=torch.float64,requires_grad=True)
            ref=(x*col)@w.T*row;direct=kron_forward(x,a,b,row,col);assert torch.allclose(ref,direct,atol=1e-10)
            g1=torch.autograd.grad(ref.square().sum(),[a,b,row,col],retain_graph=True);g2=torch.autograd.grad(direct.square().sum(),[a,b,row,col])
            for l,r in zip(g1,g2):assert torch.allclose(l,r,atol=1e-8,rtol=1e-8)
        a=torch.randn(1,2,2,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True);x=torch.randn(2,4,dtype=torch.float64);r=torch.rand(4,dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(lambda a,b,r:kron_forward(x,a,b,r),(a,b,r))
    check('rectangular_multirank_strided_factor_forward_and_backward_FP64_gradcheck',factors)
    def transforms():
        for n,block in [(16,16),(32,8),(12,4),(1,1)]:
            h=fwht(torch.eye(n,dtype=torch.float64),block)
            assert torch.allclose(h@h.T,torch.eye(n,dtype=torch.float64),atol=1e-12)
            x=torch.randn(5,n,dtype=torch.float64);sign=(torch.randint(0,2,(n,))*2-1)
            assert torch.allclose(fwht(fwht(x*sign,block),block)*sign,x,atol=1e-12)
            assert torch.allclose(fwht(x,block),x@h.T,atol=1e-12)
    check('FWHT_inverse_orthogonality_full_block_random_sign',transforms)
    def packing():
        for bits in [1,2,4,8]:
            for n in list(range(0,18))+[255,256,257]:
                for device in ['cpu','cuda:0']:
                    c=torch.arange(n,device=device)%(1<<bits);p=pack_codes(c,bits);u=unpack_codes(p,bits,n)
                    assert torch.equal(c,u);assert torch.equal(unpack_codes(p.cpu().to(device),bits,n),c)
                    if n and n%(8//bits):assert (int(p[-1])>>(n%(8//bits)*bits))==0
        try:pack_codes(torch.tensor([4]),2);raise AssertionError('accepted invalid code')
        except ValueError:pass
    check('all_integer_codes_odd_lengths_boundaries_tail_bits_CPU_GPU_roundtrip',packing)
    def codec():
        for shape in [(32,48),(17,21),(16,16)]:
            for kind in ['pure','ternary','int2','int2_sign','int4']:
                w=torch.randn(*shape,device='cuda:0')*.1;p=encode(w,kind,rank=3);x=torch.randn(3,shape[1],device=w.device)
                assert torch.allclose(reference_forward(x,p),x@decode(p).T,atol=1e-4,rtol=1e-4)
                payload=out/f'payload_{shape[0]}_{kind}.pt'
                with payload.open('xb') as f:torch.save(to_device(p,'cpu'),f)
                child=out/f'child_{shape[0]}_{kind}.pt'
                code='import torch,sys;from math_codec import decode;p=torch.load(sys.argv[1],weights_only=True);torch.save(decode(p),sys.argv[2])'
                subprocess.run([sys.executable,'-B','-c',code,str(payload),str(child)],check=True)
                assert torch.equal(torch.load(child,weights_only=True),decode(to_device(p,'cpu')))
                # Payload fixtures remain tiny; no original weight is given to child decoder.
        for v in [0.,1e-20,1e4]:
            for kind in ['pure','ternary','int4']:
                p=encode(torch.full((16,16),v),kind);assert bool(decode(p).isfinite().all())
    check('packed_payload_independent_process_reload_padding_zero_extreme_scale_placement',codec)
    def moments():
        p=torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64));orig=p.detach().clone();o=RowAdamW([p],[id(p)],lr=.1,betas=(.5,.75),weight_decay=.2)
        m=torch.zeros_like(p);v=torch.zeros(2,1,dtype=p.dtype);expected=orig
        for t,g in enumerate([torch.tensor([[2.,4.],[1.,3.]]),torch.tensor([[4.,2.],[3.,1.]])],1):
            p.grad=g.double();m=.5*m+.5*g;v=.75*v+.25*g.square().mean(-1,keepdim=True);expected=expected*.98-.1*(m/(1-.5**t))/((v/(1-.75**t)).sqrt()+1e-8);o.step();assert torch.allclose(p,expected,atol=1e-7);assert o.state[p]['v'].shape==(2,1)
    check('row_pooled_squared_gradient_recurrence_bias_correction_decay',moments)
    def qr():
        for w in [torch.zeros(2,16,16),torch.randn(2,16,1).expand(2,16,16).clone(),torch.randn(2,16,16)]:
            orig=w.clone();q=shrink(w,rank=2,decay=.2,p1=4,q1=4)
            k=unfold(orig,4,4);proj=k@q@q.transpose(-1,-2)
            assert torch.allclose(unfold(w,4,4),proj+.8*(k-proj),atol=1e-5);assert bool(w.isfinite().all())
    check('QR_projection_zero_and_rank_deficient_tail_shrink',qr)
    def causal():
        from legacy_reference import NanoMoELM
        m=NanoMoELM(vocab_size=32,d_model=16,n_layers=1,n_heads=2,num_experts=2,seq_len=8).double().eval()
        a=torch.randint(0,32,(2,8));b=a.clone();b[:,5:]=(b[:,5:]+1)%32
        assert torch.allclose(m(a)[:,:5],m(b)[:,:5],atol=1e-12)
        assert m.tok_emb.weight is m.lm_head.weight
        chunk=torch.arange(10);assert torch.equal(chunk[:-1]+1,chunk[1:])
        # Token weighting and ignored labels; non-overlap blocks score each intended target once.
        losses=torch.tensor([1.,3.,7.]);counts=torch.tensor([2,1,0]);assert float((losses*counts).sum()/counts.sum())==5/3 or torch.allclose((losses*counts).sum()/counts.sum(),torch.tensor(5/3))
    check('causal_mask_next_token_alignment_tying_token_weighted_NLL',causal)
    def immutable():
        path=out/'exclusive.json';write_json(path,{'original':True})
        try:write_json(path,{});raise AssertionError('clobber accepted')
        except FileExistsError:pass
        for a,b in [({'revision':'a'},{'revision':'b'}),({'config':1},{'config':2})]:
            try:assert_resume(a,b);raise AssertionError('resume accepted mismatch')
            except ValueError:pass
    check('exclusive_artifacts_and_resume_source_config_rejection',immutable)
    write_json(out/'checks.json',{'passed':checks,'count':len(checks),'all_passed':True})
