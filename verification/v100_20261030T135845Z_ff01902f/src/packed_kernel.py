"""Volta CUDA reference packed GEMV/GEMM: no dense residual allocation."""
import os,time,torch
from math_codec import kron_forward,fwht,to_device,tensor_bytes

CPP=r'''
#include <torch/extension.h>
torch::Tensor packed_cuda(torch::Tensor x, torch::Tensor c, torch::Tensor s,
 torch::Tensor levels, torch::Tensor sign, torch::Tensor gain,
 int64_t rows,int64_t cols,int64_t bits,int64_t group);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("packed",&packed_cuda);}
'''
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
__global__ void packed_mv(const float* x,const unsigned char* c,const half* s,
 const half* levels,const unsigned char* sign,const half* gain,float* y,
 int rows,int cols,int bits,int group){
 int row=blockIdx.x,batch=blockIdx.y,t=threadIdx.x;float sum=0;
 for(int j=t;j<cols;j+=blockDim.x){
  int idx=row*cols+j;int per=8/bits;int code=(c[idx/per]>>((idx%per)*bits))&((1<<bits)-1);
  float weight;
  if(group)weight=float(code-8)*__half2float(s[idx/group]);
  else{
   weight=__half2float(levels[code])*__half2float(s[row]);
   if(sign){int sg=(sign[idx/8]>>(idx%8))&1;weight+=(2*sg-1)*__half2float(gain[row]);}
  }
  sum=fmaf(weight,x[batch*cols+j],sum);
 }
 for(int offset=16;offset>0;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
 __shared__ float warp[8];
 if(t%32==0)warp[t/32]=sum;
 __syncthreads();
 if(t<32){sum=t<8?warp[t]:0;
  for(int offset=16;offset>0;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
  if(t==0)y[batch*rows+row]=sum;
 }
}
torch::Tensor packed_cuda(torch::Tensor x,torch::Tensor c,torch::Tensor s,
 torch::Tensor levels,torch::Tensor sign,torch::Tensor gain,
 int64_t rows,int64_t cols,int64_t bits,int64_t group){
 TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==torch::kFloat32,"contiguous CUDA FP32 x required");
 TORCH_CHECK(c.is_cuda() && c.scalar_type()==torch::kUInt8 && c.is_contiguous(),"CUDA uint8 codes required");
 TORCH_CHECK(s.is_cuda() && s.scalar_type()==torch::kFloat16 && s.is_contiguous(),"FP16 scales required");
 TORCH_CHECK(bits==2 || bits==4,"only 2/4 bit codebooks");
 TORCH_CHECK(x.size(-1)==cols && rows>0 && cols>0,"shape mismatch");
 TORCH_CHECK(c.numel()==(rows*cols*bits+7)/8,"packed length mismatch");
 TORCH_CHECK(group?s.numel()==(rows*cols+group-1)/group:s.numel()==rows,"scale length mismatch");
 TORCH_CHECK(group || (levels.is_cuda() && levels.scalar_type()==torch::kFloat16 && levels.numel()>=(1<<bits)-1),"level metadata invalid");
 TORCH_CHECK(!sign.numel() || (sign.is_cuda() && sign.scalar_type()==torch::kUInt8 && sign.numel()==(rows*cols+7)/8 && gain.numel()==rows && gain.is_cuda() && gain.scalar_type()==torch::kFloat16),"sign metadata invalid");
 c10::cuda::CUDAGuard guard(x.device());
 auto y=torch::empty({x.numel()/cols,rows},x.options());
 packed_mv<<<dim3(rows,x.numel()/cols),256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),c.data_ptr<unsigned char>(),reinterpret_cast<half*>(s.data_ptr<at::Half>()),reinterpret_cast<half*>(levels.data_ptr<at::Half>()),sign.numel()?sign.data_ptr<unsigned char>():nullptr,gain.numel()?reinterpret_cast<half*>(gain.data_ptr<at::Half>()):nullptr,y.data_ptr<float>(),rows,cols,bits,group);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
'''

def load_kernel():
    from torch.utils.cpp_extension import load_inline
    os.environ['TORCH_CUDA_ARCH_LIST']='7.0';os.environ['MAX_JOBS']='2'
    # CUDA 12.4 supports GCC 13; the host's default GCC 15 is outside its support range.
    if os.path.exists('/usr/bin/g++-13'):
        os.environ['CXX']='/usr/bin/g++-13';os.environ['CC']='/usr/bin/gcc-13'
    return load_inline(name='rpm_packed_sm70_v1',cpp_sources=CPP,cuda_sources=CUDA,extra_cuda_cflags=['-O3','--use_fast_math'],extra_cflags=['-O2'],verbose=False)

class DirectOperator:
    def __init__(self,payload,kernel):
        self.p=to_device(payload,'cuda:0');self.kernel=kernel;self.m=self.p['meta']
        # Explicit resident FP32 factor copies for the readable PyTorch contractions; counted in residency.
        if 'a' in self.p:self.p['a']=self.p['a'].float();self.p['b']=self.p['b'].float()
        self.empty=torch.empty(0,device='cuda:0',dtype=torch.float16);self.empty_codes=torch.empty(0,device='cuda:0',dtype=torch.uint8)
    def __call__(self,x):
        p=self.p;m=self.m;o,i=m['shape']
        if x.numel()==0:return torch.empty((*x.shape[:-1],o),device=x.device,dtype=torch.float32)
        if m['kind']=='int4':return self.kernel.packed(x.float().contiguous(),p['codes'],p['scales'],self.empty,self.empty_codes,self.empty,o,i,4,64)
        x=torch.nn.functional.pad(x.float(),(0,p['col'].numel()-i))*p['col'].float()
        y=kron_forward(x,p['a'],p['b'])
        if m['bits']:
            xr=fwht(x,m['block']).contiguous();res=self.kernel.packed(xr,p['codes'],p['scales'],p['levels'],p.get('sign_codes',self.empty_codes),p.get('gain',self.empty),p['row'].numel(),p['col'].numel(),m['bits'],0)
            y=y+fwht(res,m['block'])
        return (y*p['row'].float())[...,:o]
    @property
    def bytes(self):return tensor_bytes(self.p)
