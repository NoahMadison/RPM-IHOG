This additions-only verification suite tests training, compression, direct matrix execution, streaming, and distributed applicability independently. Start with the latest numbered report and CLAIMS result snapshot. RESTORED_CONTEXT.md records the recovered history; EXPERIMENT_PLAN.json was frozen before new test results. Later protocol amendments are immutable files in configs/.

The base repository is d50eec79832e6b3ec9ee943f45ac63d7e3f38b50. Everything beneath that commit remains byte-identical. This suite lives under one new verification prefix. Environments, downloaded datasets/weights, trained checkpoints, and private PDF histories remain local outside the published prefix.

Run from this directory using a fresh independent clone at TASK_ROOT/repo, with this directory at TASK_ROOT/repo/verification/EXPERIMENT_ID. `env.sh` infers TASK_ROOT, preserves HOME, redirects the requested caches/temp paths, disables tracker uploads, and masks the verified V100 UUID. GPU subprocesses verify name/UUID/PCI/memory/capability against the pre-mask nvidia-smi inventory, then use only cuda:0. Another machine must explicitly audit and update its newly added experiment configuration before use; do not silently substitute a GPU.

```bash
bash setup_fresh.sh
bash run.sh --stage audit
bash run.sh --stage data
source env.sh
"$TASK_ROOT/venv/bin/python" -B -m unittest discover -s "$VERIFY_PREFIX/tests" -v
bash run.sh --stage correctness
bash run.sh --stage legacy
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/campaign.py"
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/download_models.py"
bash run.sh --stage pretrained --model ibm-granite/granite-3.1-1b-a400m-base --codec auto
bash run.sh --stage pretrained --model HuggingFaceTB/SmolLM2-1.7B --codec auto
bash run.sh --stage mechanism
bash run.sh --stage bitrate
bash run.sh --stage runtime
bash run.sh --stage streaming
bash run.sh --stage distributed
bash run.sh --stage report
```

The observed environment is a fresh venv importing the preinstalled pinned Torch runtime read-only through a task-owned .pth file; new packages were installed only in that venv. No old environment was upgraded. setup_fresh.sh describes an independent-install alternative; the official exact CUDA12.6 CPython3.14 wheel URL/SHA was verified during this session. It must remain available on a later host. Full installed versions/build and GPU driver/compiler details are in the audit evidence.

`run_verification.py` creates unique result directories and exclusive artifact names, launches an isolated worker, saves stdout/stderr, records its exit code and hashes, and appends events.jsonl. A task-owned lock serializes GPU stages. Its conservative 24-hour budget counts completed stage wall time including CPU/I/O/waits; training's internal clock excludes time waiting for that lock. `--dry-run` resolves the source identity without execution; `--tier smoke` is explicitly a short correctness/pipeline run, never a full scientific run. `--resume path/to/config.json` rejects changed code/base/plan. Training continuation uses `--checkpoint` and validates source/configuration, model/optimizer/scaler, QR state and RNG before creating new checkpoints. New stage results never overwrite old evidence. A restarted parent can leave a worker error without exit.json; preserve that directory and append recovery notes.

Main training uses a 4-layer width-256 top-2/16-expert causal model, 128-token sequences, batch 16, tied 50,257-token GPT-2 embeddings, and the official WikiText-2 splits. Dense arms share exact initial weights and batches. Native/ordinary-low-rank arms share every non-expert tensor by explicit copying. Both factor controls have matched expert scalar counts, including row scales. Dropout is zero; clipping is 1.0; Adam moments are FP32; betas are .9/.95, epsilon 1e-8, decay .05. The row optimizer pools mean(g²), not mean(g)². Tail shrinkage is per-successful-update, ramping after step 300; separate legacy reproduction keeps the original 6%/step. Batch and QR generators are independent. Main evaluation scores contiguous nonoverlapping next-token blocks without duplicate targets, with an explicit final-tail count. Validation selection uses fixed 64-block curves; complete validation and untouched test are separate. The campaign gives each arm the same two-LR pilot budget and preserves the common-recipe comparison. Pilot/rank/penalty results are validation-only.

Pretrained evaluation runs the actual entire Granite/SmolLM2 causal model with its actual tokenizer. It scores disjoint context-512 blocks, excludes each block's first token, and reports token-weighted NLL/PPL. Granite's untouched FP16 eager evaluation was nonfinite; its declared rerun uses FP32 for baseline and decoded variants. SmolLM2 uses FP16 for both. Calibration uses 128 train-only length-128 sequences. Expert gate/up/down composition is preserved by changing only discovered matrix parameters; packed expert banks are handled explicitly. Calibration coverage includes cold experts. Granite gate/up weighting uses actual routed inputs; down projections currently use unweighted fitting. Full archives contain every unique parameter; fresh-process loaders construct from config without loading original weights and assert complete parameter coverage. Initial quality execution reconstructs dense weights and is labeled reference_decompressed. Separate full-runtime stages execute the same complete saved Granite payload directly; their fresh-process quality checks, memory and generation timing are independent evidence. Complete Granite sequential functional fitting and full PIQA/ARC-Easy tasks are reported separately. A second language-model domain remains untested.

The codec stores FP16 factors/scales plus actual packed codes: ternary uses 2 bits, INT2+sign uses 3 residual bits, and groupwise INT4 stores FP16 scales per 64 entries. Rotation is a normalized block-16 FWHT on both matrix axes with identical activation/output conventions. Generic balancing/activation scales are collapsed into stored column scales. Padding, actual rank, codebook levels, metadata and file framing are counted. Native direct export preserves trained factors before refitting; it has only row scales. Main quality exports reload in fresh processes. The separate CUDA sm_70 residual kernel decodes packed tiles without a global dense residual. Its complete matrix operator includes FP32 factor contractions, both transforms and scales. Resident FP32 factor copies are counted explicitly. Matrix timings never establish complete-model decode speed. Profiler traces and temporary peaks audit hidden materialization; the reference_decompressed comparator is labeled separately.

Streaming validates HTTP 206, exact Content-Range, header/range limits, truncation and immutable source revisions. Complete fixture archives exercise increasing sizes, raw fallback, interruption/resume and independent restoration. Real remote samples are complete expert matrices across depths/projections, explicitly not full large-model archives. No licensed model weights or dataset bodies are published.

Distributed checks use unequal sequential microbatch shards on one V100 and real dense gradients. Payload projections count all model parameters; DDP does not synchronize Adam states. Candidate shared-basis coefficient aggregation is separate from independently fitted factors, whose averaging is tested with a counterexample. Sensitivity tables are projections, not measured communication or training speed. multigpu_launcher.py is an unexecuted later-environment launcher for dense/native DDP, supported ordinary PowerSGD and a synchronous candidate hook. It requires distinct explicitly authorized UUIDs and compares a common quality target without changing host networking. Single-GPU algebra cannot support the distributed time-to-quality claim.

Publish only this prefix after running integrity verification, explicit staging, staged-content scans and the relevant correctness checks. Keep bulky profiler/fixture artifacts local if the publication manifest excludes them. Never merge, force-push, alter an old repository file, upload private histories, or upload checkpoints/dataset bodies.

Additional named stages and dependency controllers used in this session:

```bash
# Launch these controllers once in dependency order; each waits for prerequisite journal events.
# Inspect live pids/events before resuming; do not duplicate an active controller.
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/continue_campaign.py"
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/selection_campaign.py"
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/closecall_campaign.py"
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/factor_export_campaign.py"
# This supplementary controller was used with the earlier live default-only factor controller.
# The current factor_export_campaign.py includes both storage policies itself.
# "$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/precision_export_campaign.py"
bash run.sh --stage distillation
bash run.sh --stage calibration-ablation
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/prepare_downstream.py"
"$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/downstream_campaign.py"
bash run.sh --stage fused-checks
bash run.sh --stage runtime --codec fused
bash run.sh --stage full-runtime --codec fused
bash run.sh --stage full-int4-runtime
bash run.sh --stage large-streaming
bash run.sh --stage quality-diagnostics --model ibm-granite/granite-3.1-1b-a400m-base
bash run.sh --stage quality-diagnostics --model HuggingFaceTB/SmolLM2-1.7B
bash run.sh --stage report
```

The main continuation preserves all27 paired common/function-matched trajectories through6000 updates when the declared validation-trend criterion triggers. Checkpoints1500/3000/4500/6000 are selected solely by complete validation NLL. Conditional close-call seeds44/55 use conservative97.5% decision intervals at the n3/n5 looks; descriptive95% intervals remain separate. All pilot LR winners happened to be the common3e-4 recipe. Full selected test/export results and direct native-versus-ordinary-factor containers are distinct from last-update comparisons.

The fused sm_70 kernel uses warp-shuffle vector transforms and on-chip packed residual decode. The complete Granite direct path replaces all24 expert banks (2304 matrix operators), retains shared FP32 weights, and includes routing, gate/up/down, attention, output head and KV caches. The INT4 full-runtime control uses its own complete simple groupwise archive and measured fidelity; it is a readable reference kernel rather than a vendor-optimized quantization claim. Profiles and original trace hashes are retained locally; concise summaries are published. Matrix/bank inputs are synthetic Gaussian fixtures on real weight payloads, whereas full-model prompts and quality inputs are real held-out text.

Publication helpers explicitly select additions, exclude model/data/private-history bodies and large traces, scan staged Git objects, and validate base/original hashes. Shell push authentication was unavailable in this environment; the connected GitHub Git-data API can publish the identical staged tree on the authorized new branch. A remote tree hash must match `git write-tree` before committing. Fetch/fast-forward preserves the independent local clone; no reset or force push is involved. See the latest numbered handoff and integrity/publication receipts. Strict containment of initial ensurepip bootstrap temporary files is qualified in PRESERVATION_AUDIT.md.

Generation timing scope: host TTFT measures first-token readiness on the GPU. The raw field end_to_end_seconds sums timed prefill and decode through synchronization; final token D2H/hash and application transport are excluded. The report labels this prefill+decode latency. There is no per-token logging transfer in the measured loop.

Function-matched initialization copies exact FP32-materialized native expert values into dense arms and copies shared tensors. Native materialization and dense casting use different AMP operation orders; the control matches the represented FP32 function, without claiming bitwise identical mixed-precision forwards.

Future multi-host execution uses standard torchrun rendezvous variables, for example `torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 --rdzv-id=NEW_ID --rdzv-backend=c10d --rdzv-endpoint=HOST:29500 multigpu_launcher.py --method dense --out TASK_ROOT/artifacts/NEW_OUTPUT`, with node-rank1 on the other host and a distinct explicitly authorized local RPM_GPU_UUIDS on each host. Run each of dense/native/powersgd/candidate into an exclusively new output directory under that host's task root. This is an unexecuted continuation template; it requires pinned data on each host, matching environment/code and authorized distinct devices. The launcher records rank identities, checks pre-mask inventory and content lineage, broadcasts a single refreshed candidate basis, counts logical collective payload/rounds, and performs no host network changes. Standard PowerSGD eligibility differs from the expert-only candidate policy, so whole-model payloads must be measured rather than called matched by rank alone.

For the future multi-host launcher, output paths must be host-local or host-unique: each local-rank0 creates its output directory exclusively. Ranks compare code/plan/data fingerprints before training; only global-rank0 writes the aggregate curve/result, and every rank writes its own identity.

A bounded native-runtime follow-up measures the actual validation-selected seed11 allFP16 learned-factor archive. Run `bash run.sh --stage native-runtime` after direct-factor exports, or launch `python -B native_runtime_campaign.py` once alongside the already running precision-export controller; it waits for its completion event. Configs024–026 freeze originalFP32, samepayload dense/directFP32 and samepayload dense/directFP16 policies, with144 actual-payload operator checks before timing. This small training model has no KV cache; generation recomputes prefixes and reports that cost. It is a one-model runtime check, not another training replication. Same router/gating/scatter code removes training-only diagnostic overhead from both comparators. Only the originalFP32 comparison process loads the original checkpoint; decoders use only their exact saved archive. Allprecisions, storage rounding and original quality are reported separately.

CLAIMS.json records the preregistered initial ledger; the latest numbered reports/*_CLAIMS.json contains the evidence-linked verdicts. Initial plans/configurations stay immutable. Direct-factor export revalidates the selected source checkpoint hash, and native-runtime revalidates both that checkpoint and its exact archive before the reference comparison.
