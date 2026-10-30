"""Validation-only checkpoint selection within the fixed6000-update training budget."""
import json,pathlib,subprocess,sys,time,torch
from common import ROOT,PREFIX,write_json,digest
from data_audit import latest_data
from models import make_paired
from training import evaluate,export_model

def main(out,args):
    torch.set_num_threads(4);source=pathlib.Path(args.checkpoint);d=json.loads(source.read_text());cfg=json.loads((source.parent/'config.json').read_text())['arguments'];kind=d['kind'];seed=d['seed'];data=latest_data();validation=torch.load(pathlib.Path(data['artifact'])/'validation.pt',weights_only=True).to('cuda:0');model=make_paired(kind,seed,function_matched=cfg.get('function_matched',False)).to('cuda:0');choices=[]
    # The previous1500 endpoint and the continuation's3000/4500/6000 checkpoints.
    if cfg.get('checkpoint'):
        prior_cp=pathlib.Path(cfg['checkpoint']);prior_name=prior_cp.parent.name.removeprefix('training_');prior=PREFIX/'results'/prior_name/'training_results.json';previous=json.loads(prior.read_text());choices.append({'steps':1500,'validation_nll':previous['baseline']['validation']['nll'],'checkpoint':str(prior_cp),'reused_result':str(prior)})
    for steps in ([3000,4500,6000] if cfg.get('checkpoint') else [1500,3000,4500,6000]):
        cp_manifest=json.loads((source.parent/f'checkpoint_{steps:06d}.json').read_text());cp_path=pathlib.Path(cp_manifest['path']);assert digest(cp_path)==cp_manifest['sha256'],'checkpoint hash changed'
        if steps==6000:value=d['baseline']['validation']['nll'];reused=str(source)
        else:
            state=torch.load(cp_path,weights_only=True,map_location='cuda:0');model.load_state_dict(state['model']);value=evaluate(model,validation)['nll'];del state;reused=None
        choices.append({'steps':steps,'validation_nll':value,'checkpoint':str(cp_path),'reused_result':reused})
    chosen=min(choices,key=lambda r:r['validation_nll']);write_json(out/'selection_before_test.json',{'criterion':'minimum complete validation NLL, tie earliest checkpoint','choices':choices,'chosen':chosen,'test_metrics_consulted_for_selection':False})
    if chosen['reused_result']:
        selected=json.loads(pathlib.Path(chosen['reused_result']).read_text())
        for codec,package in selected['exports'].items():assert digest(package['path'])==package['sha256'],'selected packed artifact changed'
        reused=True
    else:
        state=torch.load(chosen['checkpoint'],weights_only=True,map_location='cuda:0');model.load_state_dict(state['model']);del state;tokens=torch.load(pathlib.Path(data['artifact'])/'test.pt',weights_only=True).to('cuda:0');baseline={'test':evaluate(model,tokens),'validation':{'nll':chosen['validation_nll']}};artifact=ROOT/'artifacts'/('selected_'+out.name);artifact.mkdir(exist_ok=False);exports={}
        for codec in (['native_direct','pure','ternary'] if kind=='native' else ['pure','ternary']):
            exports[codec]=export_model(model,artifact,codec);path=out/f'fresh_reload_{codec}.json';subprocess.run([sys.executable,'-B','-c','import sys;from training import reload_eval;reload_eval(*sys.argv[1:])',exports[codec]['path'],str(path),data['artifact']],check=True);exports[codec]['evaluation']=json.loads(path.read_text())
        selected={'baseline':baseline,'exports':exports};reused=False
    output={'kind':kind,'seed':seed,'function_matched':cfg.get('function_matched',False),'training_budget_updates':6000,'training_budget_tokens':6000*2048,'selected_steps':chosen['steps'],'selection_validation_nll':chosen['validation_nll'],'baseline':selected['baseline'],'exports':selected['exports'],'selected_evidence_reused_after_hash_validation':reused,'source_endpoint':str(source),'choices':choices,'selection':'validation only; all conditions offered same saved update checkpoints and LR pilot budget; final test never chooses checkpoint'}
    write_json(out/'selected_training_results.json',output);print(kind,seed,chosen['steps'],output['baseline']['test']['nll'],flush=True)
