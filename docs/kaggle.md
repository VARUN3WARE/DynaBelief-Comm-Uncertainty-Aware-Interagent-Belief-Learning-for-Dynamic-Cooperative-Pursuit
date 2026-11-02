# Running jobs on Kaggle

From 2026-11-02 all remaining training and evaluation runs on Kaggle (D47). SageMaker
(`docs/sagemaker.md`) produced the M2–M5 pilot runs and is no longer used.

## One-time setup (each student, own account)

1. Phone-verify the Kaggle account (needed for GPU and internet), at kaggle.com/settings.
2. `pip install -e ".[dev,viz,kaggle]"` in the repo venv.
3. `kaggle auth login` (browser OAuth). Never share tokens: Kaggle's terms allow one
   account per person, so each student launches their own runs.
4. `kaggle quota` shows the GPU hours left this week (about 30; resets weekly).

## Launching

Jobs run a **pushed** commit: commit and `git push` first. The generated kernel clones
the repo at that hash, so every result maps to a commit.

```bash
.venv/bin/python cloud/kaggle_launch.py bench                 # time updates at 2 and 4 workers
.venv/bin/python cloud/kaggle_launch.py tests                 # full suite on Kaggle's stack (CPU)
.venv/bin/python cloud/kaggle_launch.py train --config configs/point_diabl.yaml --seed 1
.venv/bin/python cloud/kaggle_launch.py status                # your jobs
.venv/bin/python cloud/kaggle_launch.py download <job>        # outputs to runs/kaggle/<job>/output
```

`--dry-run` writes the kernel (`runs/kaggle/<job>/kernel/`) without pushing it.

## What a job does

- A private script kernel on a T4 x2 session (one GPU used), 4 vCPUs, internet on.
- `cloud/kaggle_entry.py` installs the missing dependencies (pettingzoo, gymnasium,
  pygame), puts `src` on `PYTHONPATH` (the Kaggle image's Python may be newer than
  `requires-python`), and writes `environment.json` (versions, GPU, CPU model, commit).
- `train` stops itself after 11.25 h, before Kaggle's 12 h limit. A run that does not
  finish is **rerun from scratch** with the same seed, never resumed (a resume drops
  in-progress episodes, so it is not bit-identical). Reruns are disclosed.
- Outputs: `/kaggle/working` becomes the version's output: `status.json`,
  `environment.json`, the run directory (config, metadata, metrics, episodes, summary,
  curves, `final/best/latest.pt`). The script exits 0 even on failure so that logs are
  kept; `status.json` holds the real outcome.

## Rules for confirmatory runs

- All confirmatory runs use the same Kaggle image and GPU type; `environment.json` is
  checked for every run, and any change is reported.
- Results do not depend on the number of env workers (D9), so 4 workers on Kaggle and
  16-32 on SageMaker train identically on the same stack.
