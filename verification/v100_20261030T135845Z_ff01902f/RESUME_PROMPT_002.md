Resume handoff version2. Root=/models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f; prefix=/models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f/repo/verification/v100_20261030T135845Z_ff01902f.
Base=d50eec79832e6b3ec9ee943f45ac63d7e3f38b50.
Branch=verification/v100-20261030T135845Z-ff01902f.
Read RESUME_PROMPT.md for all preservation/hardware rules, then progress/005_live_checkpoint.json,
latest reports, configs amendments and events.jsonl. Preserve all old files/base bytes.

Main training1500 campaign and validation-triggered6000 continuation controllers remain active.
Do not duplicate them: inspect process.json and host task-owned Python PIDs, plus exit.json.
Full downstream7-condition campaign also active; full direct-runtime child is executing
cached generation and then fullWikiText2test. A worker without exit.json but with error.json
from the initial restart is stale failed evidence, not a live GPU job.
All active GPU work selects only verified V100 UUID and runs cuda:0; RTX4060 forbidden.

Current evidence: recovered source/PDF hashes match;8 GPUreference groups,11+ CPUtests;
legacy native NLL exactly3.3066;5-seed null-update mechanism; fullGranite/SmolLM2 codecNLL;
24-layer real-activation closed-loopGranite fitting improvesNLL2.4613->2.3503 vsoriginal2.2219;
strictstreaming completefixtures and6 completebigQwen matrix/expert samples;
realgradient aggregation/compressor/payload checks; sm70CUDA compiled and measured
completeoperators/gatedbanks. Core+residual is slower in testedimplementations; actual
fullmodeldirectmeasurement in progress. Wholemodeldirect has no denseexpertparameters.
Do not mix olderone-shot orFP32-inputcomponent timings with distilledfullmodelresults.

Training math dependencies are frozen in checkpoint identities: training.py,models.py,
math_codec.py,optimizers.py,common.py,data_audit.py. No edits to those without newlineage.
New runner-side lineage.py and configs/013_training_data_pin.json check actualdataset/tokenizer
revisions and tokenbufferhashes before futuretrain/resume. Resume equivalence5->10 wasbitwise.
NumPy audit correction: actualinitialimport2.5.4 (oldmetadataenumeration reported2.3.5).
Optionalharness install briefly introduced2.3.5 in newvenv, thenrestored2.5.4 locally.
Trainingmath and all batch/QR/modelRNG areTorch; oldenvironments unchanged. Timeline
configs/008 and futureworker_identity.json record real importedversions.

Exact continuation command for a specific checkpoint (new outputs automatically):
  bash /models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f/repo/verification/v100_20261030T135845Z_ff01902f/run.sh --stage train --arm ARM --seed SEED --steps 6000 --checkpoint CHECKPOINT_PATH
Supply --function-matched when originalconfig has it. SameLR/rank/tail settings required.
Privatecheckpointpaths/hashes in results/RUN/checkpoint_*.json. Latestcompleteddata.json
points to run-ownedtrain/validation/testbuffers. Never stage those data/checkpointbodies.

Next after all scientific jobs finish:
  bash /models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f/repo/verification/v100_20261030T135845Z_ff01902f/run.sh --stage report
  source /models/Project/generators/workingdir/rpm_ihog_verification_20261030T135845Z_ff01902f/repo/verification/v100_20261030T135845Z_ff01902f/env.sh
  "$TASK_ROOT/venv/bin/python" -B -m unittest discover -s "$VERIFY_PREFIX/tests" -v
  "$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/verify_integrity.py"
Then explicitly stage only the new prefix, excluding all.pt/fulltraces/invalidpartialJSON;
scan staged content and base-relativeadditions-onlydiff; commit/publish via connectedGitHubAPI
if shellGit lackscredentials. GitHubrepo permissionadmin wasverified. CreateTree frombase tree
plus newpaths in boundedbatches, check treeSHA equals localgitwrite-tree, createcommit(parentBASE),
createnewbranch(BASE), update_ref(force=false,expected_sha=BASE), openPR(no merge).
Fetch newbranch via publicHTTPS and gitmerge--ff-only with identicalstagedtree; this was tested
in a task-ownedfixture and preservesbasebytes. No reset/clean/stash/forcepush.
Save patch/bundle under ROOT/audit and publicationreceipt. Append a new handoffversion.
