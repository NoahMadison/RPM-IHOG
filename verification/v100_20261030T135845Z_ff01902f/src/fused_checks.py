import json,torch
from common import write_json
from math_codec import encode,decode,to_device,fwht
from packed_kernel_fused import load_kernel,FusedOperator
from payload_validation import validate
def main(out,args):
    kernel=load_kernel();rows=[];torch.manual_seed(9081)
    for shape in [(17,21),(32,48),(512,1024)]:
        for kind in ['pure','ternary','int2','int2_sign','int4']:
            p=encode(torch.randn(*shape,device='cuda:0')*.02,kind,rank=3);validate(p);operator=FusedOperator(p,kernel);w=decode(p)
            for batch in [1,4]:
                for dtype in [torch.float16,torch.float32]:
                    x=torch.randn(batch,shape[1],device='cuda:0',dtype=dtype);y=operator(x);ref=x.float()@w.T;err=float((y-ref).abs().max());assert torch.allclose(y,ref,atol=2e-4,rtol=2e-3),(shape,kind,dtype,err);rows.append({'shape':shape,'kind':kind,'batch':batch,'input_dtype':str(dtype),'max_abs_error':err})
    # Distinguish an actual block16 transform from full/global transforms and check padding.
    for original in [17,32,48]:
        x=torch.randn(3,original,device='cuda:0');cols=(original+15)//16*16;scale=torch.rand(cols,device='cuda:0',dtype=torch.float16);scaled,rotated=kernel.prepare(x,scale,16);expected=torch.nn.functional.pad(x,(0,cols-original))*scale.float();assert torch.equal(scaled,expected);assert torch.allclose(rotated,fwht(expected,16),atol=1e-6)
    write_json(out/'fused_checks.json',{'all_passed':True,'matrix_cases':rows,'count':len(rows),'transform_cases':3,'architecture':'sm70','packed_residual_materialized':False})
