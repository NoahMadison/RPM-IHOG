import json,math,numpy as np
from scipy.stats import t
from common import paired_ci,PREFIX

def ci97(a):
    a=np.asarray(a,float)
    if len(a)<2:return None
    delta=float(t.ppf(.9875,len(a)-1)*a.std(ddof=1)/len(a)**.5);return [float(a.mean())-delta,float(a.mean())+delta]

def add(lines,verdict):
    from reporting import completed,evidence
    runs=completed('train-select','selected_training_results.json');selected={};output={}
    for path,d in runs:
        if not d['function_matched']:selected[(d['kind'],d['seed'])]=(path,d)
    if not selected:return output
    lines+=['Validation-selected training comparisons','', 'These use the same6000-update budget and saved checkpoints1500/3000/4500/6000, selected solely by complete validation NLL before reading test metrics. The original final-endpoint common recipe remains above. Two-LR pilots offered the same tuning budget to each method; selection never chooses on test NLL. Conditional close-call replication adds44/55 only where declared;95%CIs are descriptive,97.5% decision intervals conservatively split alpha over n3/n5 looks per contrast. No cross-method multiplicity guarantee is claimed.','', '| arm | selected steps by seed | test NLL | ΔNLL vs paired tuned dense | descriptive95%CI | decision97.5%CI | PPL ratio |','|---|---|---:|---:|---|---|---:|']
    for kind in ['adamw','row','tail','row_tail','native','lowrank','legacy_blend']:
        values=[(seed,path,d) for (k,seed),(path,d) in selected.items() if k==kind]
        if not values:continue
        q=[d['baseline']['test']['nll'] for seed,path,d in values];delta=[d['baseline']['test']['nll']-selected[('adamw',seed)][1]['baseline']['test']['nll'] for seed,path,d in values if ('adamw',seed) in selected];ci=paired_ci(delta) if delta else None;bounds=ci97(delta) if delta else None
        output[kind]={'n':len(values),'steps':{str(seed):d['selected_steps'] for seed,path,d in values},'original_nll':paired_ci(q),'paired_original_delta':ci,'decision_ci97_5':bounds,'compression_delta':{},'sources':[str(path.relative_to(PREFIX)) for seed,path,d in values]}
        for codec in ['native_direct','pure','ternary']:
            ds=[d['exports'][codec]['evaluation']['nll']-d['baseline']['test']['nll'] for seed,path,d in values if codec in d['exports']]
            if ds:output[kind]['compression_delta'][codec]={**paired_ci(ds),'decision_ci97_5':ci97(ds),'ppl_ratio':math.exp(float(np.mean(ds)))}
        lines.append(f"| {kind} | {output[kind]['steps']} | {np.mean(q):.5f} | {ci['mean'] if ci else '—'} | {ci['ci95'] if ci else '—'} | {bounds} | {math.exp(ci['mean']) if ci else '—'} |")
    lines+=['Selected-checkpoint export fidelity and storage','', '| arm / export | seeds | ΔNLL vs same model | decision97.5%CI | geometric PPL ratio | expert tensor bpw | whole tensor bpw | serialized bytes |','|---|---:|---:|---|---:|---:|---:|---:|']
    for kind,record in output.items():
        for codec,metrics in record['compression_delta'].items():
            exports=[d['exports'][codec] for (k,seed),(path,d) in selected.items() if k==kind and codec in d['exports']]
            whole=[d.get('whole_model_bpw_dense_equivalent',d.get('whole_model_bpw')) for d in exports]
            lines.append(f"| {kind}/{codec} | {metrics['n']} | {metrics['mean']:+.6f} | {metrics['decision_ci97_5']} | {metrics['ppl_ratio']:.5f} | {np.mean([d['expert_bpw'] for d in exports]):.4f} | {np.mean(whole):.4f} | {np.mean([d['serialized_bytes'] for d in exports]):.0f} |")
    lines+=['','PPL ratios exponentiate paired mean NLL deltas; archive values average the selected per-seed payloads. Whole-model values include unchanged FP32 shared tensors here; allFP16 complete-container controls follow separately.','']
    statuses=[]
    for kind in ['row','tail','row_tail','native']:
        if kind not in output or output[kind]['n']<3 or not output[kind]['decision_ci97_5']:continue
        r=output[kind];q=r['decision_ci97_5'];f=[v['decision_ci97_5'] for v in r['compression_delta'].values() if v['decision_ci97_5']];quality_ok=q[1]<=.02;fidelity_ok=any(v[1]<=.01 for v in f)
        status='supported_in_tested_scope' if quality_ok and fidelity_ok else 'contradicted_in_tested_scope' if q[0]>.02 or (f and all(v[0]>.01 for v in f)) else 'inconclusive';r['joint_equal_quality_and_export_status']=status;statuses.append(status)
    if statuses:
        status='supported_in_tested_scope' if 'supported_in_tested_scope' in statuses else 'contradicted_in_tested_scope' if all(v=='contradicted_in_tested_scope' for v in statuses) else 'inconclusive';verdict('A',status,'Joint +.02 original-model noninferiority and+.01 same-model compression tolerance assessed against validation-selected dense within equal6000-update tuning budget; see per-arm intervals and seed coverage',[p/'selected_training_results.json' for p,d in selected.values()])
    lines+=['','All selected checkpoints and reused archive hashes are validated. Exact per-seed originals, exports and block losses remain in the linked raw results: '+', '.join(evidence(path/'selected_training_results.json') for path,d in selected.values()),'']
    factors=completed('direct-factor-export','direct_factor_results.json');factor_values={}
    if factors:
        lines+=['Matched direct native/ordinary factor containers','', '| arm | storage policy | seed | selected step | expert bpw | whole tensor bpw | serialized bytes | original NLL | reloaded NLL | ΔNLL |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for path,d in factors:
            factor_values[(d['kind'],d['seed'],d['variant'])]=d;lines.append(f"| {d['kind']} | {d['variant']} | {d['seed']} | {d['selected_steps']} | {d['expert_bpw']:.4f} | {d['whole_model_bpw']:.4f} | {d['serialized_bytes']} | {d['original_test_nll']:.6f} | {d['reloaded_test_nll']:.6f} | {d['delta_nll']:+.7f} |")
        output['native_minus_lowrank_matched_direct_reload']={}
        for policy in ['expert_fp16_shared_fp32','all_fp16']:
            delta=[factor_values[('native',seed,policy)]['reloaded_test_nll']-factor_values[('lowrank',seed,policy)]['reloaded_test_nll'] for seed in [11,22,33,44,55] if ('native',seed,policy) in factor_values and ('lowrank',seed,policy) in factor_values]
            output['native_minus_lowrank_matched_direct_reload'][policy]=paired_ci(delta) if delta else None
        lines+=['','Native and ordinary-low-rank export original learned A/B and row scalars in FP16 with4352 expert scalars/matrix and matched complete-bank containers. Policies separately keep unaffected parameters FP32 or round all parameters to FP16; the latter also includes a dense FP16 baseline. Compare whole-model bytes within a policy. Paired native-minus-ordinary-low-rank reloaded NLL by policy: `'+json.dumps(output['native_minus_lowrank_matched_direct_reload'])+'`. These quality runs materialize factors in the training model; no optimized speed or compressed residency claim follows.','']
    return output
