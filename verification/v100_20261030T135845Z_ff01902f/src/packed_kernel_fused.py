"""sm70 follow-up: warp-shuffle block16 FWHT fused with input/output scaling."""
import os,torch
from math_codec import kron_forward
from packed_kernel import CPP as BASE_CPP,CUDA as BASE_CUDA,DirectOperator

EXTRA_CUDA=r'''
template<typename T> __device__ float input_float(T x){return float(x);}
template<> __device__ float input_float<half>(half x){return __half2float(x);}
template<typename T> __global__ void prepare_kernel(const T* x,const half* col,
 float* scaled,float* rotated,int batch,int original_cols,int cols,int block){
 int idx=blockIdx.x*blockDim.x+threadIdx.x,total=batch*cols;
 if(idx>=total)return;
 int j=idx%cols,b=idx/cols;float v=(j<original_cols?input_float(x[b*original_cols+j]):0)*__half2float(col[j]);
 scaled[idx]=v;unsigned mask=__activemask();
 for(int h=1;h<block;h*=2){float other=__shfl_xor_sync(mask,v,h,16);v=(j&h)?other-v:v+other;}
 rotated[idx]=v*(block==16?.25f:1.f);
}
__global__ void finish_kernel(const float* residual,const float* core,const half* row,
 float* output,int total,int cols,int block){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=total)return;
 int j=idx%cols;float v=residual[idx];unsigned mask=__activemask();
 for(int h=1;h<block;h*=2){float other=__shfl_xor_sync(mask,v,h,16);v=(j&h)?other-v:v+other;}
 output[idx]=(v*(block==16?.25f:1.f)+core[idx])*__half2float(row[j]);
}
std::vector<torch::Tensor> prepare_cuda(torch::Tensor x,torch::Tensor col,int64_t block){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&(x.scalar_type()==torch::kFloat32||x.scalar_type()==torch::kFloat16),"CUDA contiguous FP16/32 input required");
 TORCH_CHECK(col.is_cuda()&&col.is_contiguous()&&col.scalar_type()==torch::kFloat16,"FP16 column scales required");
 TORCH_CHECK((block==1||block==16)&&col.numel()%16==0&&x.size(-1)<=col.numel(),"supported block/padding required");
 c10::cuda::CUDAGuard guard(x.device());int original=x.size(-1),batch=x.numel()/original,cols=col.numel();
 auto scaled=torch::empty({batch,cols},x.options().dtype(torch::kFloat32));auto rotated=torch::empty_like(scaled);int threads=256,blocks=(batch*cols+threads-1)/threads;
 if(x.scalar_type()==torch::kFloat16)prepare_kernel<<<blocks,threads,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),reinterpret_cast<const half*>(col.data_ptr<at::Half>()),scaled.data_ptr<float>(),rotated.data_ptr<float>(),batch,original,cols,block);
 else prepare_kernel<<<blocks,threads,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),reinterpret_cast<const half*>(col.data_ptr<at::Half>()),scaled.data_ptr<float>(),rotated.data_ptr<float>(),batch,original,cols,block);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {scaled,rotated};
}
torch::Tensor finish_cuda(torch::Tensor residual,torch::Tensor core,torch::Tensor row,int64_t block){
 TORCH_CHECK(residual.is_cuda()&&core.is_cuda()&&row.is_cuda()&&residual.is_contiguous()&&core.is_contiguous()&&row.is_contiguous(),"CUDA contiguous finish inputs required");
 TORCH_CHECK(residual.scalar_type()==torch::kFloat32&&core.scalar_type()==torch::kFloat32&&row.scalar_type()==torch::kFloat16,"finish dtypes invalid");
 TORCH_CHECK(residual.sizes()==core.sizes()&&row.numel()==core.size(-1)&&row.numel()%16==0&&(block==1||block==16),"finish shapes invalid");
 c10::cuda::CUDAGuard guard(core.device());auto output=torch::empty_like(core);int total=core.numel();finish_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(residual.data_ptr<float>(),core.data_ptr<float>(),reinterpret_cast<const half*>(row.data_ptr<at::Half>()),output.data_ptr<float>(),total,row.numel(),block);C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
'''

def load_kernel():
    from torch.utils.cpp_extension import load_inline
    os.environ['TORCH_CUDA_ARCH_LIST']='7.0';os.environ['MAX_JOBS']='2'
    if os.path.exists('/usr/bin/g++-13'):os.environ['CXX']='/usr/bin/g++-13';os.environ['CC']='/usr/bin/gcc-13'
    original='PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("packed",&packed_cuda);}'
    replacement='std::vector<torch::Tensor> prepare_cuda(torch::Tensor,torch::Tensor,int64_t);\ntorch::Tensor finish_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("packed",&packed_cuda);m.def("prepare",&prepare_cuda);m.def("finish",&finish_cuda);}'
    assert original in BASE_CPP
    cuda=BASE_CUDA.replace('c.numel()==(rows*cols*bits+7)/8','c.numel()==(group?(((rows*cols+group-1)/group)*group*bits+7)/8:(rows*cols*bits+7)/8)')+EXTRA_CUDA
    return load_inline(name='rpm_packed_sm70_fused_v2',cpp_sources=BASE_CPP.replace(original,replacement),cuda_sources=cuda,extra_cuda_cflags=['-O3','--use_fast_math'],extra_cflags=['-O2'],verbose=False)

class FusedOperator(DirectOperator):
    def __call__(self,x):
        p=self.p;m=self.m;o,i=m['shape'];shape=x.shape[:-1]
        if x.numel()==0:return torch.empty((*shape,o),device=x.device,dtype=torch.float32)
        if m['kind']=='int4':return super().__call__(x.reshape(-1,i)).reshape(*shape,o)
        scaled,rotated=self.kernel.prepare(x.reshape(-1,i).contiguous(),p['col'],m['block']);core=kron_forward(scaled,p['a'],p['b'])
        if m['bits']:
            residual=self.kernel.packed(rotated,p['codes'],p['scales'],p['levels'],p.get('sign_codes',self.empty_codes),p.get('gain',self.empty),p['row'].numel(),p['col'].numel(),m['bits'],0)
            y=self.kernel.finish(residual,core.contiguous(),p['row'],m['block'])
        else:y=core*p['row'].float()
        return y[...,:o].reshape(*shape,o)
