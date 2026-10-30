"""Serial bounded main training campaign. Run with task venv after sourcing env.sh."""
import json,pathlib,subprocess,sys,time
from common import PREFIX,ROOT,write_json,journal

def main():
    campaign=ROOT/'artifacts'/('campaign_'+str(time.time_ns()));campaign.mkdir(exist_ok=False)
    arms=['adamw','row','tail','row_tail','native','lowrank','legacy_blend']
    specs=[]
    # Pilot tuning uses validation only. Equal small LR budget for all methods.
    for arm in arms:
        for lr in [.0001,.0003]: specs.append({'phase':'tuning','arm':arm,'seed':11,'lr':lr,'steps':600,'selection_only':True})
    # Rank and penalty pilots, secondary mechanism coverage, no final-test access.
    for arm in ['native','lowrank']:specs.append({'phase':'rank_pilot','arm':arm,'seed':11,'lr':.0003,'rank':4,'steps':600,'selection_only':True})
    for arm in ['tail','row_tail']:
        for penalty in [0,.0003]:specs.append({'phase':'penalty_pilot','arm':arm,'seed':11,'lr':.0003,'tail_decay':penalty,'steps':600,'selection_only':True})
    for seed in [11,22,33]:
        for arm in arms:specs.append({'phase':'common_recipe','arm':arm,'seed':seed,'lr':.0003,'steps':1500})
    for seed in [11,22,33]:
        for arm in ['adamw','row_tail']:specs.append({'phase':'function_matched','arm':arm,'seed':seed,'lr':.0003,'steps':1500,'function_matched':True})
    write_json(campaign/'plan.json',specs)
    # Written before launching any new main training or selection result.
    write_json(PREFIX/'configs'/('campaign_'+campaign.name+'.json'),{'specs':specs,'continuation':'if validation improves >0.02 between steps 1200 and 1500, continue selected comparison to 6000 steps before plateau claims','time_to_quality_nll':7.0,'tuning_choice':'minimum complete-validation NLL among 600-step LR pilots per arm; rank/penalty pilots secondary; common recipe remains primary'})
    for idx,spec in enumerate(specs):
        args=['bash',str(PREFIX/'run.sh'),'--stage','train','--arm',spec['arm'],'--seed',str(spec['seed']),'--lr',str(spec['lr']),'--steps',str(spec['steps'])]
        for k in ['rank','tail_decay']:
            if k in spec:args+=['--'+k.replace('_','-'),str(spec[k])]
        for k in ['selection_only','function_matched']:
            if spec.get(k):args+=['--'+k.replace('_','-')]
        journal('campaign_job_start',index=idx,spec=spec)
        with (campaign/f'job_{idx:03d}.stdout').open('x') as out,(campaign/f'job_{idx:03d}.stderr').open('x') as err:r=subprocess.run(args,stdout=out,stderr=err)
        write_json(campaign/f'job_{idx:03d}.json',{'spec':spec,'returncode':r.returncode})
        journal('campaign_job_exit',index=idx,returncode=r.returncode)
        print(idx,spec,r.returncode,flush=True)
        if r.returncode:raise RuntimeError(f'campaign job {idx} failed; inspect evidence before resuming')
    journal('campaign_complete',path=str(campaign))

if __name__=='__main__':main()
