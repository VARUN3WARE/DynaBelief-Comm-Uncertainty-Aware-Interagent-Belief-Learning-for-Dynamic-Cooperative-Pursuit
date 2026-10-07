# Running on SageMaker

All training runs as SageMaker training jobs ([D22](decisions.md)). The local machine
only launches jobs, monitors them, and downloads results.

## One-time setup

```bash
.venv/bin/python -m pip install -e ".[aws]"     # boto3
aws sts get-caller-identity                     # or any check that ~/.aws credentials work
```

Defaults in `cloud/launch.py` (override with flags):

| Setting | Value |
|---|---|
| Region | from `~/.aws/config` (us-east-1) |
| Role | `AmazonSageMaker-ExecutionRole-20260927T013216` (`--role` to change) |
| Bucket | `sagemaker-<region>-<account>`, everything under `dynabelief/` |
| Image | AWS PyTorch DLC `2.5.1` / py311 (GPU or CPU build by instance family) |
| Instance | `ml.g4dn.8xlarge`: 32 vCPUs for env workers + T4 for updates ([D23](decisions.md)) |

## Workflow

```bash
git commit ... && git push                      # jobs run the committed HEAD only
.venv/bin/python cloud/launch.py train --config configs/no_comm.yaml --set experiment.seed=1
.venv/bin/python cloud/launch.py status
.venv/bin/python cloud/launch.py logs <job> --follow
.venv/bin/python cloud/launch.py download <job>    # -> runs/sagemaker/<job>/ (no .pt files)
.venv/bin/python cloud/launch.py stop <job>
```

Useful flags for `train`: `--instance`, `--spot` (managed spot, cheaper; a restarted
job resumes from `latest.pt`), `--max-hours` (default 6), `--eval-episodes` (default 50),
`--dry-run` (print the request).

## What a job does

1. `pip install .` from a `git archive` of the commit (recorded in tags and `metadata.yaml`).
2. Trains with `scripts/train.py` into `/opt/ml/checkpoints/runs/`. SageMaker syncs this to
   `s3://<bucket>/dynabelief/checkpoints/<job>/` while the job runs.
3. Evaluates `final.pt` on the held-out eval seeds with sampled and greedy actions, into `evals/`.
4. Plots `curves.png` and exports configs, summaries, curves and `final.pt` as `model.tar.gz`.

Env workers = the largest divisor of `num_envs` that leaves 2 vCPUs free
(32 envs on 32 vCPUs → 16 workers). Worker count does not change results ([D9](decisions.md)).

## Quotas (account, us-east-1, 2026-10-07)

Training-job quota is 1 each for `ml.g4dn.{xlarge,2xlarge,4xlarge,8xlarge,12xlarge,16xlarge}` and
2–4 for `ml.c5/c6i/c7i.4xlarge–9xlarge`. Different instance types can run at the same time.
Other projects in this account use g5/g6 instances.
