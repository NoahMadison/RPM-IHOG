#!/usr/bin/env python3
"""Immutable stage orchestration: one isolated process, new outputs, validated resume."""
import argparse,importlib,json,os,pathlib,resource,subprocess,sys,time,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'src'))
from common import PREFIX,ROOT,write_json,new_run,journal,source_identity,gpu_guard,digest,assert_resume

STAGES={'audit':'audit_stage','data':'data_audit','correctness':'correctness','fused-checks':'fused_checks','legacy':'legacy_run','train':'training','train-select':'select_training','direct-factor-export':'direct_factor_export','mechanism':'mechanism','bitrate':'bitrate','pretrained':'pretrained','quality-diagnostics':'quality_diagnostics','distillation':'distillation','calibration-ablation':'calibration_ablation','residual-control':'residual_control','downstream':'downstream','runtime':'runtime','full-runtime':'full_runtime','full-int4-runtime':'full_int4_runtime','native-runtime':'native_runtime','interleaved-runtime':'interleaved_runtime','streaming':'streaming','large-streaming':'large_streaming','distributed':'distributed','report':'reporting'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=list(STAGES),required=True);ap.add_argument('--tier',choices=['smoke','main'],default='main');ap.add_argument('--dry-run',action='store_true');ap.add_argument('--resume');ap.add_argument('--seed',type=int);ap.add_argument('--arm');ap.add_argument('--model');ap.add_argument('--codec',default='ternary');ap.add_argument('--steps',type=int);ap.add_argument('--checkpoint');ap.add_argument('--lr',type=float,default=.0003);ap.add_argument('--rank',type=int,default=8);ap.add_argument('--tail-decay',type=float,default=.003);ap.add_argument('--function-matched',action='store_true');ap.add_argument('--selection-only',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--out');args=ap.parse_args()
    identity=source_identity()
    if args.resume:assert_resume(json.loads(pathlib.Path(args.resume).read_text())['identity'],identity)
    if args.dry_run:print(json.dumps({'stage':args.stage,'tier':args.tier,'identity':identity,'budget_gpu_hours':24},indent=2));return
    if args.worker:
        out=pathlib.Path(args.out)
        try:
            if args.stage not in ['data','report']:
                import fcntl
                gpu_lock=(ROOT/'gpu_execution.lock').open('a')
                fcntl.flock(gpu_lock,fcntl.LOCK_EX)
                write_json(out/'gpu.json',gpu_guard())
            import importlib.metadata,numpy
            write_json(out/'worker_identity.json',{'source_identity_at_worker_start':source_identity(),'orchestrator_sha256':digest(__file__),'numpy_imported_version':numpy.__version__,'numpy_module':numpy.__file__,'library_versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','datasets','safetensors','huggingface-hub']}})
            if args.stage in ['train','train-select','direct-factor-export','native-runtime','interleaved-runtime']:
                from lineage import training_data_guard
                write_json(out/'data_lineage.json',training_data_guard())
            module=importlib.import_module(STAGES[args.stage]);module.main(out,args)
        except Exception as e:
            write_json(out/'error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
        return
    # Enforce cumulative complete stage wall budget conservatively, including CPU/download time.
    used=sum(json.loads(p.read_text()).get('wall_seconds',0) for p in (PREFIX/'results').glob('*/exit.json'))
    if used>=24*3600:raise RuntimeError('24-hour conservative first-pass budget exhausted')
    out=new_run(args.stage);config={'identity':identity,'arguments':vars(args),'command':sys.argv,'budget_remaining_seconds':24*3600-used};write_json(out/'config.json',config)
    journal('stage_start',stage=args.stage,result=str(out.relative_to(PREFIX)))
    cmd=[sys.executable,'-B',__file__]+sys.argv[1:]+['--worker','--out',str(out)]
    t=time.perf_counter()
    with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
        proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,cwd=out,start_new_session=True)
        write_json(out/'process.json',{'pid':proc.pid,'command':cmd})
        try:code=proc.wait(timeout=max(1,24*3600-used))
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid,signal.SIGTERM);code=proc.wait(timeout=60)
    write_json(out/'exit.json',{'exit_code':code,'wall_seconds':time.perf_counter()-t})
    write_json(out/'artifact_hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
    journal('stage_exit',stage=args.stage,exit_code=code,result=str(out.relative_to(PREFIX)))
    print(json.dumps({'result':str(out),'exit_code':code}),flush=True)
    sys.exit(code)

if __name__=='__main__': main()
