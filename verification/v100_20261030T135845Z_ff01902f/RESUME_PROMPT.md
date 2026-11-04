Resume RPM-IHOG verification from this independent task tree.

Repository: https://github.com/NoahMadison/RPM-IHOG
Base: d50eec79832e6b3ec9ee943f45ac63d7e3f38b50
Branch: verification/v100-20261030T135845Z-ff01902f
Task root: /models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f
Prefix: /models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f/repo/verification/v100_20261030T135845Z_ff01902f

First read README_VERIFICATION.md, RESTORED_CONTEXT.md, EXPERIMENT_PLAN.json,
configs amendments, latest numbered report/progress, and events.jsonl.
The user authorizes bounded experiments, additions-only commit/push/PR; no merge.
Preserve every pre-session file and every base-commit tracked byte. Only this new
prefix may enter the Git diff. No PDF histories, data bodies, weights, environments,
or large checkpoints may be staged. All new writes/caches/tmp remain in task root;
HOME stays unchanged. Never use reset/clean/stash, sudo, force push, or global Git config.

Hardware: requested device1 is Tesla V100-PCIE-32GB, UUID
GPU-b1e26f14-9243-49b6-c5a7-51f7ce39672d, PCI00000000:06:00.0, sm70.
Sandbox hides GPU/network; authorized GPU/network commands need host execution.
env.sh masks verified UUID; each GPU subprocess checks pre-mask inventory and
masked cuda:0. Never use laptop RTX4060 or cuda:1 after masking.

Completed: hardware/data/source audit; 8 correctness groups and5 range failure tests;
300-step legacy reproduction;5-seed causal regression; complete archive streaming
fixtures and real expert ranges; full Granite and SmolLM2 before/after packed-reload
NLL for selected int2+sign/core and INT4; initial sharded-gradient checks.
Both one-shot pretrained codecs miss+.01NLL tolerance. Granite FP16 baseline
was nonfinite, so its consistent baseline/decoded path is FP32; SmolLM2 is FP16.
GPU quality execution is reference_decompressed, not compressed model residency.

Active/queued: campaign.py (47 serial jobs:20 validation pilots,21 common3-seed
arms,6 function-matched runs); continue_campaign.py waits for those27 endpoints
and conditionally continues all to6000 based on predeclared validation improvement.
Runtime retry compiled sm70 and is timing exact payloads plus full gated/routed
bank; expanded distributed diagnostic and final correctness rerun queued.
Check process.json, exit.json, error.json before repeating. A background job isn't
complete. Collect exit codes. All results directories and checkpoints are unique;
never overwrite old files. Checkpoint manifests point to private ROOT/artifacts.
Training source dependencies are frozen for compatible resume; edits require
new lineage. Existing checkpoints use row/QR RNG state, actual optimizer/scaler,
initial shared hashes and cumulative curves/time. Continuation checkpoints are
every1500 updates to bound disk use (~65GB projected vs109GB initially free).

Inspect current state:
  source /models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f/repo/verification/v100_20261030T135845Z_ff01902f/env.sh
  "$TASK_ROOT/venv/bin/python" -B -c 'from common import PREFIX; print((PREFIX/"events.jsonl").read_text()[-8000:])'

Generate a fresh report from completed evidence:
  bash /models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f/repo/verification/v100_20261030T135845Z_ff01902f/run.sh --stage report

Before publication: run integrity source checks, final GPU correctness and CPU tests,
explicitly stage only prefix (exclude synthetic .pt fixtures/full profiler traces as
appropriate), inspect staged list/credential/private-history scan and additions-only
diff, then commit/push unique branch and open PR. Do not merge or change old docs.
If shell Git auth fails, connected GitHub tools have repository access; use them
within authorization. Retain additions-only patch and bundle if publication blocks.

This is handoff version1; append a new version when state changes.
