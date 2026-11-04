"""Generate new immutable reports from completed evidence; never promote partial jobs."""
import collections,datetime,json,math,pathlib,time,re
import numpy as np
from common import PREFIX,ROOT,write_json,paired_ci,block_bootstrap,digest,journal
from analysis_stats import weighted_block_delta

def completed(stage,filename):
    rows=[]
    for directory in sorted((PREFIX/'results').glob('*_'+stage+'_*')):
        exitfile=directory/'exit.json';path=directory/filename
        if not exitfile.exists() or not path.exists():continue
        if json.loads(exitfile.read_text())['exit_code']!=0:continue
        try:data=json.loads(path.read_text())
        except json.JSONDecodeError:continue
        rows.append((directory,data))
    return rows

def average(a):return float(np.mean(a)) if a else None
def fmt(x):return '—' if x is None else f'{x:.4f}'
def evidence(path):return '['+str(path.relative_to(PREFIX))+'](../'+str(path.relative_to(PREFIX))+')'

def main(out,args):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');sequence=len(list((PREFIX/'reports').glob('*.md')))+1;name=f'{sequence:03d}_{stamp}'
    lines=['RPM-IHOG verification: measured scope and limits','',f'Generated {stamp}; base commit `d50eec79832e6b3ec9ee943f45ac63d7e3f38b50`. The original repository files are preserved; every implementation/evidence addition is under this prefix.','']
    verdicts=json.loads((PREFIX/'CLAIMS.json').read_text());byid={c['id']:c for c in verdicts}
    def verdict(cid,status,reason,paths):
        byid[cid].update(status=status,reason=reason,evidence=[str(p.relative_to(PREFIX)) for p in paths])
    audit=completed('audit','environment.json')
    if audit:
        path,env=audit[-1];gpu=json.loads((path/'gpu.json').read_text());driver=re.search(r'Driver Version: ([0-9.]+)',env['commands']['nvidia-smi']['stdout']).group(1);lines+=['Hardware/software','',f"Verified requested Tesla V100-PCIE-32GB, UUID GPU-b1e26f14-9243-49b6-c5a7-51f7ce39672d, PCI 00000000:06:00.0, 32768 MiB inventory / {gpu['torch_device']}; compute capability 7.0. The RTX 4060 was excluded. PyTorch {gpu['torch']}, CUDA runtime {gpu['cuda']}, Python {gpu['python'].splitlines()[0]}. Driver {driver}; CUDA compiler 12.4; CPU Intel Core i7-13700H (20 logical CPUs), host RAM32,520,777,728 bytes. sm_70 present; FP16 GEMM/backward finite. GCC 13 selected for CUDA compilation. Fresh task venv; existing runtime imported read-only; new dependencies installed locally.",evidence(path/'environment.json'),'']
    data=completed('data','data.json')
    if data:
        path,d=data[-1];lines+=['Pinned sources','',f"WikiText-2 official splits: `{d['revision']}`; GPT-2 tokenizer `{d['tokenizer_revision']}`. Main token counts: "+', '.join(f"{s} {v['tokens']:,}" for s,v in d['splits'].items())+'. Content/token hashes and legacy corpus identity: '+evidence(path/'data.json'),evidence(path/'model_identities.json'),'']
    checks=completed('correctness','checks.json')
    if checks:lines+=['Correctness','',f"{checks[-1][1]['count']} GPU/reference test groups passed, including rectangular factor forward/backward, FP64 gradcheck, block/full transforms, packing boundaries/CPU-GPU round trips, independent-process decode, causal masking, recurrence, QR zero/rank-deficient cases and no-clobber/resume rejection. Sixteen separately recorded CPU tests cover initialization, pinned data lineage, invalid codec payloads, retries and ignored/malformed/truncated/source-changed HTTP ranges. A zero-FP16-scale underflow was found and fixed before measurements. Raw evidence: "+evidence(checks[-1][0]/'checks.json'),'']
    legacy=[r for r in completed('legacy','legacy_results.json') if r[1].get('steps')==300]
    if legacy:
        path,d=legacy[-1];results=d.get('runs',{})
        lines+=['Legacy 300-step reproduction (one training seed)','',f"Dense/native actual parameters: {d.get('parameters_dense'):,}/{d.get('parameters_native'):,}. FP16 autocast training, FP32 parameters and FP32 evaluation; hashed overlapping-byte-pair legacy task. AMP skips: {d['amp']['skipped_updates']}/{d['amp']['attempted_updates']}. The latest serial reproduction is used; the earlier overlapping pilot timing is retained only as earlier evidence. No packed reload underlies this table.",'','| Arm | Original NLL | Pure refit NLL | Core+ternary NLL | Kronecker stable rank | training seconds |','|---|---:|---:|---:|---:|---:|']
        for arm,v in results.items():lines.append(f"| {arm} | {v['val_loss_fp16']:.4f} | {v['val_loss_pure_kron_053bpw']:.4f} | {v['val_loss_rpm_ihog_212bpw']:.4f} | {v['kronecker_stable_rank']:.2f} | {v['wall_time_sec']:.2f} |")
        lines += ['', 'The supplied native original/refit/residual NLLs reproduce to printed precision when the same corpus/source semantics are used. The row+QR arm differs under the tested stack; its discrepancy cause was not isolated. Main experiments separately isolate RNG streams. See exact printed-value comparisons and implementation deviations. Middle-layer-only stable rank is not a model-wide metric. Generic 256x256 rank-8 core is 1.125 theoretical bpw; ideal ternary totals 2.7725, while actual 2-bit ternary plus scale metadata is larger. Historical field names containing 0.53/2.12 are retained only as legacy labels.',evidence(path/'legacy_results.json'),'']
        from legacy_comparison import compare
        legacy_comparison=compare(results)
        lines+=['Legacy printed-value verification','', '| arm | matching non-time printed metrics | original NLL difference from supplied | pure-refit NLL difference | ternary NLL difference |','|---|---:|---:|---:|---:|']
        for arm,metrics in legacy_comparison.items():
            count=sum(r['matches_printed_value'] for r in metrics.values());lines.append(f"| {arm} | {count}/{len(metrics)} | {metrics['val_loss_fp16']['delta_printed']:+.4f} | {metrics['val_loss_pure_kron_053bpw']['delta_printed']:+.4f} | {metrics['val_loss_rpm_ihog_212bpw']['delta_printed']:+.4f} |")
        lines+=['','This compares the twelve non-time values supplied in the mission at their printed precision, not hidden unrounded originals. Training timing is hardware/software dependent and is measured separately above. The claimed row+QR results are revised rather than silently called exact reproduction. Native dense-to-Kronecker stable-rank ratio in this one bank is64.95/5.41≈12.0, with the original-model NLL tradeoff shown above. Full per-metric deltas are in this report’s JSON.','']
        verdict('H13','inconclusive','Legacy diagnostics reproduce improved Kronecker compressibility with worse uncompressed NLL; exact historical 13x configuration differs and one middle bank is not a multi-layer equal-quality result',[path/'legacy_results.json'])
    training=completed('train','training_results.json');common={};matched={};pilots=[]
    for path,d in training:
        cfg=json.loads((path/'config.json').read_text())['arguments']
        if cfg.get('selection_only'):pilots.append((path,d,cfg));continue
        if d.get('tier')!='main' or d['steps']<1500:continue
        (matched if cfg.get('function_matched') else common)[(d['kind'],d['seed'])]=(path,d,cfg)
    training_summary={};paired={};lines+=['Main real-token training','', 'The common recipe uses four width-256 layers, 16 top-2 experts, 128-token sequences, batch 16, GPT-2 vocabulary, FP32 parameters and FP16 autocast. Dense optimizer arms share initial weights and batch RNG; native/ordinary low-rank controls share non-expert tensors and match expert parameter counts. Dropout 0, clipping 1, Adam betas .9/.95, decay .05, LR .0003. Tail shrinkage is per successful update, delayed/ramped from step 300 to 600. Legacy post-step blending is labeled separately. Two LR choices per arm use only validation. Main seeds are 11/22/33; incomplete seed coverage is explicit. No tuning pilot sees test results.','', '| Arm | seeds | steps | Original test NLL | Pure reload ΔNLL | Ternary reload ΔNLL | Native direct ΔNLL | Train seconds incl periodic val/checkpoints | Peak allocated MiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ['adamw','row','tail','row_tail','native','lowrank','legacy_blend']:
        values=[(p,d,c) for (k,s),(p,d,c) in common.items() if k==kind]
        if not values:continue
        get=lambda key:[d[key] for _,d,_ in values]
        nll=[d['baseline']['test']['nll'] for _,d,_ in values]
        deltas={codec:[d['exports'][codec]['evaluation']['nll']-d['baseline']['test']['nll'] for _,d,_ in values if codec in d['exports']] for codec in ['pure','ternary','native_direct']}
        training_summary[kind]={'seed_count':len(values),'nll':paired_ci(nll),'compression_delta':{k:paired_ci(v) for k,v in deltas.items() if v},'parameters':get('parameters'),'expert_parameters':get('expert_parameters'),'optimizer_bytes':get('optimizer_tensor_bytes'),'successful_updates':get('successful_updates'),'skipped_updates':get('skipped_updates'),'sources':[str(p.relative_to(PREFIX)) for p,_,_ in values]}
        lines.append(f"| {kind} | {len(values)} | {','.join(str(x) for x in get('steps'))} | {average(nll):.4f} | {fmt(average(deltas['pure']))} | {fmt(average(deltas['ternary']))} | {fmt(average(deltas['native_direct']))} | {average(get('training_seconds_including_periodic_validation_checkpointing')):.2f} | {average(get('peak_allocated'))/2**20:.1f} |")
    lines+=['','Paired equal-token comparisons against dense AdamW, 95% Student-t intervals across seeds. Noninferiority margin +0.02 NLL; PPL ratio is exp(ΔNLL). These test only the declared recipe/tuning scope.','', '| Arm | paired seeds | ΔNLL mean | 95% CI | PPL ratio | equal-quality verdict |','|---|---:|---:|---|---:|---|']
    for kind in training_summary:
        if kind=='adamw':continue
        diff=[common[(kind,s)][1]['baseline']['test']['nll']-common[('adamw',s)][1]['baseline']['test']['nll'] for s in [11,22,33,44,55] if (kind,s) in common and ('adamw',s) in common]
        if not diff:continue
        ci=paired_ci(diff);bounds=ci['ci95'];status='inconclusive' if bounds is None else 'supported_in_tested_scope' if bounds[1]<=.02 else 'contradicted_in_tested_scope' if bounds[0]>.02 else 'inconclusive';paired[kind]={**ci,'ppl_ratio':math.exp(ci['mean']),'status':status}
        lines.append(f"| {kind} | {len(diff)} | {ci['mean']:.4f} | {bounds} | {math.exp(ci['mean']):.4f} | {status} |")
    if common:
        evidence_paths=[p/'training_results.json' for p,_,_ in common.values()]
        verdict('A','inconclusive','See paired main-training table: compressibility and equal-quality are reported separately; common recipe, bounded tuning, and plateau/continuation coverage limit generalization',evidence_paths)
        # Determine the narrow tested claim only when at least three paired seeds exist.
        useful=[k for k,v in paired.items() if v['n']>=3 and v['status']=='supported_in_tested_scope' and k in ['row','tail','row_tail','native'] and any(c.get('ci95') and c['ci95'][1]<=.01 for c in training_summary[k]['compression_delta'].values())]
        if useful:verdict('A','supported_in_tested_scope','At least one structured arm met the declared paired NLL margin in the tested recipe; see compression/export deltas and actual bytes independently',evidence_paths)
        lines+=['','Every individual run retains per-update counts, gradient norms, routing/utilization, activation RMS, checkpoint hashes, multi-layer/projection spectra, and exact packed reload metrics. Source directories: '+', '.join(evidence(p/'training_results.json') for p,_,_ in common.values()),'']
    # Main storage accounting stays separate from full-path runtime.
    lines+=['Training archive accounting (means; complete archive includes unaffected parameters)','', '| Arm / export | Expert tensor bpw | Whole-model tensor bpw (dense equivalent) | Serialized bytes | Before NLL | Reload NLL | Execution |','|---|---:|---:|---:|---:|---:|---|']
    for kind in training_summary:
        vals=[d for (k,s),(p,d,c) in common.items() if k==kind]
        for codec in ['native_direct','pure','ternary']:
            ex=[(d,d['exports'][codec]) for d in vals if codec in d['exports']]
            if not ex:continue
            lines.append(f"| {kind}/{codec} | {average([e['expert_bpw'] for d,e in ex]):.4f} | {average([e.get('whole_model_bpw_dense_equivalent',e.get('whole_model_bpw')) for d,e in ex]):.4f} | {average([e['serialized_bytes'] for d,e in ex]):.0f} | {average([d['baseline']['test']['nll'] for d,e in ex]):.4f} | {average([e['evaluation']['nll'] for d,e in ex]):.4f} | reference_decompressed / native materialized training path |")
    lines+=['','Whole-model bpw uses represented dense-equivalent parameters, not the smaller trainable-factor count. Main shared tensors are FP32. Expert-only savings must not be relabeled whole-model savings. Baseline uncompressed and codec/reload memory include separate dense reconstruction and temporary workspaces; none demonstrates compressed complete-model GPU residency.','']
    lines+=['Training state and equal-wall-time observations','', '| arm | trainable parameters | expert parameters | optimizer tensor MiB | skipped AMP updates |','|---|---:|---:|---:|---|']
    for kind,d in training_summary.items():lines.append(f"| {kind} | {average(d['parameters']):.0f} | {average(d['expert_parameters']):.0f} | {average(d['optimizer_bytes'])/2**20:.2f} | {d['skipped_updates']} |")
    wall_summary={}
    for seed in [11,22,33]:
        runs={kind:run for (kind,s),run in common.items() if s==seed}
        if len(runs)<7:continue
        cutoff=min(d['training_seconds_including_periodic_validation_checkpointing'] for path,d,cfg in runs.values())
        wall_summary[str(seed)]={'common_seconds':cutoff,'arms':{}}
        for kind,(path,d,cfg) in runs.items():
            points=[row for row in d['curve'] if row['wall_seconds_including_validation']<=cutoff and 'validation_selection_64_blocks' in row]
            if points:
                row=points[-1];wall_summary[str(seed)]['arms'][kind]={'step':row['step'],'observed_seconds':row['wall_seconds_including_validation'],'tokens':row['tokens_consumed'],'selection_validation_nll':row['validation_selection_64_blocks']['nll'],'staleness_seconds':cutoff-row['wall_seconds_including_validation']}
    lines+=['','Descriptive equal-wall-time comparison uses the minimum completed training clock across seven arms within each seed; the last observed held-out validation checkpoint at/before that cutoff is reported without interpolation. Checkpoint staleness is explicit and these64-block selection metrics do not replace final test endpoints: `'+json.dumps(wall_summary)+'`.','']
    # Tuning, wall-time and function-matched evidence.
    tuning={}
    for kind in training_summary:
        options=[(p,d,c) for p,d,c in pilots if d['kind']==kind and d['steps']==600 and c.get('rank',8)==8 and c.get('tail_decay',.003)==.003]
        if options:
            best=min(options,key=lambda x:x[1]['baseline']['validation']['nll']);tuning[kind]={'selected_lr':best[2]['lr'],'selection_validation_nll':best[1]['baseline']['validation']['nll'],'tuning_runs':len(options),'source':str(best[0].relative_to(PREFIX))}
    lines+=['Tuning, initialization and time budgets','', 'Validation-only LR selection: `'+json.dumps(tuning,sort_keys=True)+'`. The common-recipe main table is untuned; when the selected LR equals .0003 it also supplies that chosen-recipe endpoint. Other selected settings need their own main replications. Rank 4/8 and zero/weaker/ramped tail pilots are secondary coverage, not a combinatorial search. Function-matched dense arms start from native-materialized expert weights and retain the native initialization distribution; they are separate from ordinary dense initialization. Matching copies FP32-materialized expert values exactly; native factor materialization and dense matrix casting have different AMP operation orders, so bitwise identical mixed-precision forwards are not asserted.','']
    function_summary={}
    for kind in ['adamw','row_tail']:
        vals=[]
        for seed in [11,22,33]:
            if (kind,seed) in matched and ('native',seed) in common and matched[(kind,seed)][1]['steps']==common[('native',seed)][1]['steps']:vals.append(matched[(kind,seed)][1]['baseline']['test']['nll']-common[('native',seed)][1]['baseline']['test']['nll'])
        if vals:function_summary[kind+'_minus_native']=paired_ci(vals)
    lines+=['Validation-only bounded rank/penalty pilots','', '| arm | rank | tail strength per update | LR | complete validation NLL | consumed tokens | measured train seconds |','|---|---:|---:|---:|---:|---:|---:|']
    for path,d,cfg in pilots:
        if cfg.get('lr')!=.0003:continue
        lines.append(f"| {d['kind']} | {cfg['rank']} | {cfg['tail_decay']} | {cfg['lr']} | {d['baseline']['validation']['nll']:.5f} | {d['tokens_consumed']} | {d['training_seconds_including_periodic_validation_checkpointing']:.2f} |")
    lines+=['','Each pilot is seed11 only at600 updates; none is a replicated final-quality claim. Rank variation and tail0/.0003/.003 are not combined into a grid. Pure ordinary low rank has the same factor scalar count as the native control; archive tables separately evaluate each learned representation.','']
    lines+=['Function-matched paired test NLL differences: `'+json.dumps(function_summary)+'`.','']
    time_targets={}
    for kind in training_summary:
        values=[]
        for (k,s),(p,d,c) in common.items():
            if k!=kind:continue
            reached=[r['wall_seconds_including_validation'] for r in d['curve'] if r.get('validation_selection_64_blocks',{}).get('nll',1e9)<=7.0]
            values.append(min(reached) if reached else None)
        time_targets[kind]={'target_nll':7.0,'measured_times_seconds':values,'status':'reached' if all(v is not None for v in values) else 'target_not_reached'}
    lines+=['Time to the predeclared common validation NLL 7.0 (checkpoints every 300 steps; no extrapolation): `'+json.dumps(time_targets)+'`. Equal-wall-time curves use measured selection validation endpoints; do not interpret periodic minibatch training loss as held-out quality.','']
    mechanism=completed('mechanism','mechanism.json')
    if mechanism:
        path,d=mechanism[-1];lines+=['Controlled optimizer mechanism','', '| zero-init arm | summed gradient null norm | summed update null norm | final null weight norm | fitted loss |','|---|---:|---:|---:|---:|']
        for arm in ['sgd','adamw','row','tail','isotropic']:
            rows=[r for r in d['rows'] if r['initialization']=='zero' and r['arm']==arm];lines.append('| '+arm+' | '+' | '.join(f"{average([r[k] for r in rows]):.6g}" for k in ['gradient_null_norm','update_null_norm','final_null_norm','loss'])+' |')
        lines+=['','Five seeds, known 16-dimensional row space in R64, zero and matched random initializations, fixed geometry and deterministic batch noise=0. Coordinatewise preconditioning moves actual updates outside the gradient row space; row pooling strongly reduces that effect in this fixture. This does not establish that AdamW maximizes entropy, that null-space motion causes language-model compression losses, or that nonlinear low-sensitivity modes are exact null directions. Histogram/body clipping, raw scales and permutation/factor-shape sensitivity are recorded. Isotropic decay is fixed-strength rather than norm matched; noisy geometry sweeps remain open.',evidence(path/'mechanism.json'),'']
        verdict('B','inconclusive','The limited preconditioning/null-update mechanism is observed; the stronger universal entropy/useless-null-space causal explanation is not established',[path/'mechanism.json'])
    bitrate=completed('bitrate','bitrate.json')
    if bitrate:
        path,d=bitrate[-1];lines+=['Bitrate and matched-byte matrix controls','', '| codec | tensor bpw | serialized bpw | SQNR dB | matched-byte ordinary LR rank | LR SQNR dB |','|---|---:|---:|---:|---:|---:|']
        for row in d['actual_real_matrix_codecs']:lines.append(f"| {row['kind']} | {row['tensor_payload_bpw']:.4f} | {row['serialized_bpw']:.4f} | {row['weight_sqnr_db']:.3f} | {row['matched_payload_lowrank_rank']} | {row['matched_payload_lowrank_sqnr_db']:.3f} |")
        lines+=['','Complete real gate matrix scope; this table is not full-model quality. Native direct synthetic fixture FP16 payload is 1.0625 bpw before framing; generic core 1.125 at 256x256. 2-bit residual plus sign is three residual bits per entry. File/container metadata and unaffected model tensors are counted separately. INT8 factors, FP8 compute, and ideal base-3 storage are not asserted.',evidence(path/'bitrate.json'),'']
        verdict('D','contradicted_in_tested_scope','Measured payload/framing and complete archive totals differ from historical bitrate labels; native/generic/ternary/sign budgets are explicitly distinguished',[path/'bitrate.json'])
        verdict('H05','contradicted_in_tested_scope','At the supplied 256x256 rank-8 shapes, generic/native factors alone exceed 0.5 bpw before framing; no 0.5-bpw no-loss result was reproduced',[path/'bitrate.json'])
    pretrained=completed('pretrained','pretrained_results.json');pretrained_status=[]
    lines+=['Real pretrained full-model fidelity','', '| model / codec | Baseline NLL | Reload NLL | ΔNLL / PPL ratio | Δ block 95% CI | Expert/MLP bpw | Whole bpw | Serialized bytes | Resident execution |','|---|---:|---:|---|---|---:|---:|---:|---|']
    for path,d in pretrained:
        for codec,result in d['results'].items():
            weighted_ci=weighted_block_delta(d['baseline']['test']['rows'],result['evaluation']['rows']);ci=weighted_ci['ci95'];status='supported_in_tested_scope' if ci[1]<=.01 else 'contradicted_in_tested_scope' if ci[0]>.01 else 'inconclusive';pretrained_status.append(status)
            lines.append(f"| {d['model']}/{codec} | {d['baseline']['test']['nll']:.4f} | {result['evaluation']['nll']:.4f} | {result['delta_nll']:+.4f} / {result['ppl_ratio']:.4f} | {ci} | {result['expert_or_mlp_bpw']:.4f} | {result['whole_model_bpw']:.4f} | {result['serialized_bytes']} | reference_decompressed |")
        lines+=['',f"{d['model']} immutable revision `{d['revision']}`; full actual model/tokenizer, 128 train-only calibration sequences length 128; codec selected on first-layer validation pilot, then entire target tensor set encoded and reloaded in a fresh process. Coverage, original/source hashes and full parameter shapes are recorded. Original weights are never loaded by the packed decoder. Calibration is independent; Granite gate/up uses real routed inputs, down is unweighted. Block-bootstrap CI treats contiguous contexts as units; the final short block is token-weighted in NLL. This is an engineering codec pilot with explicitly pending matched-byte full-model low-rank, second-domain coverage. Later fitting and full downstream results are reported separately below.",evidence(path/'pretrained_results.json'),'']
    if pretrained:
        status='contradicted_in_tested_scope' if all(s=='contradicted_in_tested_scope' for s in pretrained_status) else 'inconclusive'
        verdict('C',status,'See same-model before/after NLL and simple INT4 comparator; neither matrix SQNR nor agreement substitutes for full-model quality. The bounded calibration recipes and model/task scope limit competitive-quality conclusions',[p/'pretrained_results.json' for p,_ in pretrained])
    runtime=completed('runtime','runtime.json')
    if runtime:
        path,d=runtime[-1]
        if d.get('status')=='completed':
            lines+=['Measured direct execution','',f"sm_70 CUDA packed residual compiled in {d['compilation_seconds']:.2f} s. Kernel decodes packed weights on chip; readable FP32 factor contractions and block FWHT vectors avoid global dense residual allocation. Resident FP32 factor copies are counted. Same saved payloads as the fidelity archive; CUDA events and host time, random interleaved groups, equal warmup and profiler traces retained. Sustained copy diagnostic: {d['device_copy_bandwidth_bytes_per_second_counting_read_write']/1e9:.2f} GB/s counting reads+writes. Integer decode, FP32 FMA, FP16 conversion, transforms, intermediates and launches preclude assigning the tensor-core peak as the relevant compute ceiling.",'','| model/codec | matrix | batch | dense FP16 median ms | direct complete median ms | speed ratio | resident direct bytes | peak temporary bytes |','|---|---|---:|---:|---:|---:|---:|---:|']
            for r in d['rows']:lines.append(f"| {r['identity']['model']}/{r['identity']['codec']} | {r['identity']['shape']} | {r['batch']} | {np.median(r['device_ms']['dense_fp16']):.5f} | {np.median(r['device_ms']['direct_complete']):.5f} | {r['median_speedup_vs_fp16']:.4f} | {r['resident_direct_bytes']} | {r['direct_peak_temporary_bytes']} |")
            lines+=['','Complete routed/gated bank measurements (router, top-k weights, SiLU(gate)*up, down, scatter included; no attention or full-model decode):','','| codec | batch | expert union | dense median ms | direct median ms | speed ratio |','|---|---:|---:|---:|---:|---:|']
            for r in d.get('routed_gated_bank_rows',[]):lines.append(f"| {r['codec']} | {r['batch']} | {r['active_expert_union']} | {np.median(r['device_ms']['dense_fp16']):.5f} | {np.median(r['device_ms']['direct']):.5f} | {r['median_speedup_vs_dense_fp16']:.4f} |")
            lines+=['','CPU pure-core components, one fixed thread, same core as materialized comparator: `'+json.dumps([{'model':r['identity']['model'],'codec':r['identity']['codec'],'speed_ratio':r['median_core_speedup']} for r in d.get('cpu_core_components',[])])+'`. These omit residual/rotation and do not carry full-model fidelity. GPU/CPU matrix or bank timings cannot establish complete-model generation throughput, prefill/TTFT, KV residency, or distributed performance. Cache-resident weight reuse is measured; streaming working-set coverage remains not_run; separate complete-model autoregressive measurements follow below.',evidence(path/'runtime.json'),'']
            cis=[r['direct_minus_dense_fp16_ms_paired_ci']['ci95'] for r in d['rows'] if r['identity']['codec']!='int4'];slower=bool(cis) and all(ci and ci[0]>0 for ci in cis)
            verdict('E','contradicted_in_tested_scope' if slower else 'inconclusive','Complete matrix/bank operator results are bounded to these payloads, batches and readable kernel; full-model compressed decode is not run',[path/'runtime.json'])
            verdict('H83','not_run','Historical complete GPU cascade/generation speed is not reproduced by matrix/bank timing',[path/'runtime.json'])
            verdict('HCPU','inconclusive','CPU component ratios measured at actual shapes; no full-path 5x–10x claim or historical-workload identity established',[path/'runtime.json'])
        else:verdict('E','blocked',d.get('error','kernel unavailable'),[path/'runtime.json'])
    streaming=completed('streaming','streaming.json')
    if streaming:
        path,d=streaming[-1];lines+=['Streaming/archive verification','', '| fixture tensors | source bytes | downloaded bytes | peak live source bytes | total archive bytes | coverage/reload |','|---|---:|---:|---:|---:|---|']
        for r in d['complete_fixture_archives']:lines.append(f"| {r['fixture_tensor_count']} | {r['source_bytes']} | {r['downloaded_bytes']} | {r['peak_live_source_range_bytes']} | {r['archive_bytes_including_manifests']} | {r['coverage_complete']} / fresh process |")
        remote=d['real_tensor_streaming'];lines+=['',f"Real remote complete expert-matrix samples: {len(remote.get('samples',[]))}; downloaded {remote.get('downloaded_bytes','blocked')} bytes. Exact HTTP206/ranges/revisions, expert IDs/depths/projections, payload/source hashes and SQNR are recorded. No full large model archive or residency claim follows from samples.",evidence(path/'streaming.json'),'']
        verdict('F','supported_in_tested_scope','Complete fixture archives have full manifest coverage, bounded per-tensor source ranges, interruption/resume, raw fallback and fresh-process restoration; remote real matrices remain samples',[path/'streaming.json'])
    distributed=completed('distributed','distributed.json')
    if distributed:
        path,d=distributed[-1];lines+=['Distributed applicability','', '| representation | whole FP32 gradient bytes | expert FP32 gradient bytes | Adam moments bytes |','|---|---:|---:|---:|']
        for kind,v in d['payloads'].items():lines.append(f"| {kind} | {v['whole_gradient_bytes_fp32']} | {v['expert_gradient_bytes_fp32']} | {v['adam_moments_bytes_fp32']} |")
        lines+=['','Unequal sequential shards [2,1] match concatenated gradients and an optimizer update with stochastic effects disabled. Real trained dense gradients—not just learned weights—feed ordinary/shared-Kronecker coefficient projections and INT8/error-feedback diagnostics. Coefficients are numeric; independently fitted Kronecker factors cannot be averaged separately. The current shared-basis pilot uses an initial centralized SVD, explicitly not a proven distributed basis-refresh method; the future launcher counts full refresh traffic. Whole-model parameter payload dominates any expert-only ratio. DDP does not synchronize Adam optimizer states. Ring sensitivity estimates use actual full gradient bytes plus link bandwidth/hop latency, with no claim of measured allreduce or time-to-quality.',evidence(path/'distributed.json'),'Actual multi-GPU training is blocked: only one GPU is authorized. multigpu_launcher.py is provided for distinct authorized devices; no duplicate-device NCCL ranks or RTX 4060 runs occurred.','']
        verdict('G','blocked','Actual multi-GPU time-to-quality requires additional authorized devices; single-GPU algebra and bandwidth projections do not satisfy it',[path/'distributed.json'])
    verdict('H27','not_run','Large-model identities/configs verified, but full archive/coverage/load/residency and full execution were not produced',[(data[-1][0]/'model_identities.json')] if data else [])
    verdict('H998','not_run','No full-model 99.8% activation R-squared replication; complete-model NLL is evaluated separately',[])
    from report_supplement import add
    additional=add(lines,verdict)
    from report_selected import add as add_selected
    additional['validation_selected_training']=add_selected(lines,verdict)
    exits=[(path,json.loads(path.read_text())) for path in (PREFIX/'results').glob('*/exit.json')]
    charged=sum(data.get('wall_seconds',0) for path,data in exits);intervals=[]
    for path,data in exits:
        guard=path.parent/'gpu.json'
        if guard.exists():intervals.append({'result':str(path.parent.relative_to(PREFIX)),'guard_file_mtime_unix_ns':guard.stat().st_mtime_ns,'exit_file_mtime_unix_ns':path.stat().st_mtime_ns,'approximate_lock_held_wall_seconds':max(0,(path.stat().st_mtime_ns-guard.stat().st_mtime_ns)/1e9)})
    additional['resource_budget']={'ceiling_hours':24,'conservative_charged_completed_stage_wall_hours':charged/3600,'approximate_completed_GPU_slot_held_wall_hours':sum(row['approximate_lock_held_wall_seconds'] for row in intervals)/3600,'intervals':intervals,'limits':'file-mtime interval includes CPU/load/evaluation while GPU lock held, not sampled GPU-kernel busy time; excludes initial probes and orphaned parent result without exit; charged budget intentionally includes queue waits and CPU stages'}
    lines+=['Execution budget','',f"First-pass ceiling24 hours. Completed stage wall time conservatively charged {charged/3600:.3f} hours, including CPU/download/queue waits. Approximate completed GPU-slot-held wall time {additional['resource_budget']['approximate_completed_GPU_slot_held_wall_hours']:.3f} hours, inferred from captured guard/exit file mtimes and including in-lock CPU work. This is not utilization-weighted GPU kernel time; original timestamps and exclusions are in the report JSON.",'']
    lines+=['Claim ledger','', '| Claim | verdict | bounded interpretation |','|---|---|---|']
    for c in verdicts:lines.append(f"| {c['id']} | {c['status']} | {c.get('reason','Not run in completed evidence')} |")
    lines+=['','Smallest useful contribution','', 'A complete packed Granite archive executes without original expert matrices and agrees with its own decoded payload while using substantially less steady GPU allocation. Transform fusion improves the readable direct path, though dense reconstruction remains faster. Bounded real-activation fitting improves pretrained NLL without reaching the declared fidelity margin. Exact native-factor export and matched ordinary-factor controls separate learned representability from later SVD refitting. Streaming/restoration and gradient aggregation have independent correctness evidence; these findings remain useful without accepting the stronger entropy, latency or distributed claims.','', 'Remaining evidence','', 'Read all per-stage errors and incomplete coverage. Needed for stronger claims: selected/tuned and plateau-complete training replications where absent; noisy/norm-matched causal controls; more rank/factor-shape/permutation controls; full-model matched-byte alternatives, richer closed-loop/distillation/calibration and downstream domains/tasks; optimized transform/core fusion, fair decode/prefill and streaming working sets; complete large archive/restoration/residency; actual multi-GPU time to common quality. Blocked/not_run stages are not scientific failures.','', 'Reproduction and recovery','', 'Exact per-stage commands and source identities are in results/*/config.json and process.json. Invoke `bash run.sh --stage STAGE`; use a new immutable result directory for each retry. See README_VERIFICATION.md, latest progress file and RESUME_PROMPT*.md. All model/data/checkpoint artifact paths are local-only, with content hashes; never publish their bodies. Final integrity/publication evidence is added separately after staging.','']
    payload={'legacy_printed_comparison':legacy_comparison if legacy else {},'equal_wall_time_observations':wall_summary,'training':training_summary,'paired_equal_quality':paired,'tuning':tuning,'function_matched':function_summary,'time_to_quality':time_targets,'claims':verdicts,'additional':additional,'completed_training_common_runs':len(common),'completed_function_matched_runs':len(matched)}
    write_json(PREFIX/'reports'/(name+'.json'),payload);write_json(PREFIX/'reports'/(name+'_CLAIMS.json'),verdicts)
    if common:lines+=['',f'![Measured training, storage and quality Pareto plots]({name}.png)','']
    report=PREFIX/'reports'/(name+'.md')
    with report.open('x') as stream:stream.write('\n'.join(lines))
    plot(report.with_suffix('.png'),common)
    write_json(out/'report.json',{'report':str(report.relative_to(PREFIX)),'data':str((PREFIX/'reports'/(name+'.json')).relative_to(PREFIX))});journal('report_written',path=str(report.relative_to(PREFIX)))
    print(report,flush=True)

def plot(path,common):
    if not common:return
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for (kind,seed),(directory,d,cfg) in common.items():
        if seed!=11:continue
        curve=[r for r in d['curve'] if 'validation_selection_64_blocks' in r];axes[0].plot([r['tokens_consumed'] for r in curve],[r['validation_selection_64_blocks']['nll'] for r in curve],label=kind)
        for codec,e in d['exports'].items():
            axes[1].scatter(e['serialized_bytes']/1e6,e['evaluation']['nll']);axes[1].annotate(kind+'/'+codec,(e['serialized_bytes']/1e6,e['evaluation']['nll']),fontsize=5)
        axes[2].scatter(d['training_seconds_including_periodic_validation_checkpointing'],d['baseline']['test']['nll']);axes[2].annotate(kind,(d['training_seconds_including_periodic_validation_checkpointing'],d['baseline']['test']['nll']),fontsize=7)
    for directory,d in completed('direct-factor-export','direct_factor_results.json'):
        if d['seed']!=11 or d['function_matched'] or d['variant']!='all_fp16':continue
        axes[1].scatter(d['serialized_bytes']/1e6,d['reloaded_test_nll'],marker='^',s=55)
        axes[1].annotate(d['kind']+'/direct-allFP16-selected',(d['serialized_bytes']/1e6,d['reloaded_test_nll']),fontsize=5)
    axes[0].set(xlabel='tokens consumed',ylabel='selection validation NLL',title='Seed 11 training curves');axes[0].legend(fontsize=6)
    axes[1].set(xlabel='complete serialized MB',ylabel='fresh-reloaded test NLL',title='Storage / quality (seed 11)');axes[2].set(xlabel='training seconds incl periodic checks',ylabel='original test NLL',title='Training time / quality (seed 11)')
    fig.tight_layout()
    with path.open('xb') as stream:fig.savefig(stream,format='png',dpi=160)
    plt.close(fig)
