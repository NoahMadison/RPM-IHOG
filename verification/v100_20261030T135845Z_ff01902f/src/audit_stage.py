import importlib.metadata,json,os,platform,subprocess,torch
from common import write_json
def main(out,args):
    x=torch.randn(128,128,device='cuda:0',dtype=torch.float16,requires_grad=True)
    loss=(x@x).float().square().mean();loss.backward();torch.cuda.synchronize()
    assert bool(x.grad.isfinite().all())
    cmds={}
    for cmd in [['nvidia-smi'],['nvcc','--version'],['free','-b'],['df','-B1',str(out)],['lscpu']]:
        r=subprocess.run(cmd,capture_output=True,text=True);cmds[' '.join(cmd)]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    write_json(out/'environment.json',{'platform':platform.platform(),'commands':cmds,'packages':packages,'torch_build':torch.__config__.show(),'fp16_gemm_backward_finite':True,'loss':float(loss),'cache_variables':{k:v for k,v in os.environ.items() if k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TORCH_HOME','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME','PIP_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR','TMPDIR','CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','WANDB_MODE']},'environment_note':'fresh venv, new packages local; preexisting PyTorch/runtime imported read-only via task_torch_runtime.pth; system packages read-only; no existing environment changed'})
