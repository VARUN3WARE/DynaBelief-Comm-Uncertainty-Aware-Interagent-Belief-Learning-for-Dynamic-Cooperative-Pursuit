"""Launch and manage DynaBelief-Comm SageMaker training jobs from this machine.

Every job runs the code of one clean, committed ``HEAD`` (shipped as ``git archive``),
so each result maps to an exact commit. Requires ``pip install -e ".[aws]"`` and AWS
credentials (``~/.aws``).

Examples:
    python cloud/launch.py train --config configs/smoke_mappo.yaml --instance ml.g4dn.xlarge
    python cloud/launch.py train --config configs/no_comm.yaml --set experiment.seed=1
    python cloud/launch.py status
    python cloud/launch.py logs <job> --follow
    python cloud/launch.py download <job>          # metrics, plots, eval records (no .pt)
    python cloud/launch.py stop <job>
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFIX = "dynabelief"  # S3 key prefix and job-name prefix
# AWS PyTorch DLC; 2.5.1 / py311 matches the local venv (verified to exist in us-east-1).
IMAGE = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/"
    "pytorch-training:2.5.1-{kind}-py311{cuda}-ubuntu22.04-sagemaker"
)
DEFAULT_INSTANCE = "ml.g4dn.8xlarge"  # 32 vCPUs for env workers + T4 for PPO updates
DEFAULT_ROLE_NAME = "AmazonSageMaker-ExecutionRole-20260927T013216"
ENTRY = "cloud/sagemaker_entry.py"


# --------------------------------------------------------------------- pure helpers
def image_uri(region: str, instance_type: str) -> str:
    gpu = instance_type.split(".")[1].startswith(("g", "p"))
    return IMAGE.format(region=region, kind="gpu" if gpu else "cpu", cuda="-cu124" if gpu else "")


def job_name(config_name: str, seed: int, now: datetime) -> str:
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", config_name).strip("-").lower()
    name = f"{PREFIX}-{slug}-s{seed}-{now:%Y%m%d-%H%M%S}"
    return name[:63].rstrip("-")


def training_job_request(
    *,
    name: str,
    region: str,
    role_arn: str,
    bucket: str,
    source_uri: str,
    commit: str,
    config: str,
    overrides: list[str],
    instance_type: str,
    spot: bool,
    max_hours: float,
    eval_episodes: int,
) -> dict:
    """``create_training_job`` arguments for script mode in the PyTorch DLC."""
    hyper = {
        "sagemaker_program": ENTRY,
        "sagemaker_submit_directory": source_uri,
        "sagemaker_region": region,
        "sagemaker_container_log_level": 20,  # must decode to an int (logging level)
        "config": config,
        "overrides": ";".join(overrides),
        "eval_episodes": eval_episodes,
    }
    max_seconds = int(max_hours * 3600)
    request = {
        "TrainingJobName": name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri(region, instance_type),
            "TrainingInputMode": "File",
        },
        # The DLC's training toolkit JSON-decodes every value, so encode native types
        # (a string "20" log level crashed the toolkit's logging setup). Empty values are
        # dropped: the toolkit turns them into a bare flag with no argument.
        "HyperParameters": {k: json.dumps(v) for k, v in hyper.items() if v != ""},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 30},
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/{PREFIX}/output"},
        "CheckpointConfig": {
            "S3Uri": f"s3://{bucket}/{PREFIX}/checkpoints/{name}",
            "LocalPath": "/opt/ml/checkpoints",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": max_seconds},
        "Environment": {"DYNABELIEF_GIT_COMMIT": commit, "PYTHONUNBUFFERED": "1"},
        "EnableManagedSpotTraining": spot,
        "Tags": [
            {"Key": "project", "Value": PREFIX},
            {"Key": "git_commit", "Value": commit},
            {"Key": "config", "Value": config},
        ],
    }
    if spot:
        request["StoppingCondition"]["MaxWaitTimeInSeconds"] = max_seconds * 2
    return request


# --------------------------------------------------------------------- git / aws
def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def clean_commit() -> str:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty: commit first, jobs must map to a commit")
    commit = git("rev-parse", "HEAD")
    try:
        if git("rev-parse", "@{u}") != commit:
            print("warning: HEAD is not pushed to the upstream branch", file=sys.stderr)
    except subprocess.CalledProcessError:
        print("warning: no upstream branch", file=sys.stderr)
    return commit


def session():
    import boto3

    return boto3.Session()


def defaults(sess) -> tuple[str, str, str]:
    region = sess.region_name or "us-east-1"
    account = sess.client("sts").get_caller_identity()["Account"]
    role = f"arn:aws:iam::{account}:role/service-role/{DEFAULT_ROLE_NAME}"
    return region, f"sagemaker-{region}-{account}", role


def upload_source(sess, bucket: str, commit: str) -> str:
    key = f"{PREFIX}/source/{commit}.tar.gz"
    s3 = sess.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError:
        archive = subprocess.run(["git", "archive", "--format=tar.gz", commit], cwd=REPO,
                                 check=True, capture_output=True).stdout  # fmt: skip
        s3.upload_fileobj(io.BytesIO(archive), bucket, key)
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------- commands
def cmd_train(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "src"))
    from dynabelief.config import load_config, parse_override

    overrides = list(args.set)
    config = load_config(REPO / args.config, dict(parse_override(o) for o in overrides))
    commit = clean_commit()
    sess = session()
    region, bucket, role = defaults(sess)
    name = args.name or job_name(config.experiment.name, config.experiment.seed, datetime.now(UTC))
    source = "s3://<uploaded on launch>" if args.dry_run else upload_source(sess, bucket, commit)
    request = training_job_request(
        name=name, region=region, role_arn=args.role or role, bucket=bucket,
        source_uri=source, commit=commit, config=args.config, overrides=overrides,
        instance_type=args.instance, spot=args.spot, max_hours=args.max_hours,
        eval_episodes=args.eval_episodes,
    )  # fmt: skip
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0
    sess.client("sagemaker").create_training_job(**request)
    print(f"launched {name}\n  commit   {commit}\n  instance {args.instance} spot={args.spot}")
    print(f"  live S3  {request['CheckpointConfig']['S3Uri']}")
    print(f"  console  https://{region}.console.aws.amazon.com/sagemaker/home?region={region}"
          f"#/jobs/{name}")  # fmt: skip
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    sm = session().client("sagemaker")
    jobs = sm.list_training_jobs(NameContains=PREFIX, SortBy="CreationTime",
                                 SortOrder="Descending", MaxResults=args.n)  # fmt: skip
    for job in jobs["TrainingJobSummaries"]:
        d = sm.describe_training_job(TrainingJobName=job["TrainingJobName"])
        secs = d.get("TrainingTimeInSeconds") or (
            (datetime.now(UTC) - d["CreationTime"]).total_seconds()
        )
        print(f"{job['TrainingJobName']:60s} {d['TrainingJobStatus']:11s} "
              f"{d.get('SecondaryStatus', ''):12s} {secs / 3600:5.2f} h  "
              f"{d['ResourceConfig']['InstanceType']}")  # fmt: skip
        if d.get("FailureReason"):
            print(f"    failure: {d['FailureReason'][:300]}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    sess = session()
    logs, sm = sess.client("logs"), sess.client("sagemaker")
    group = "/aws/sagemaker/TrainingJobs"
    seen: dict[str, str | None] = {}
    while True:
        streams = logs.describe_log_streams(logGroupName=group, logStreamNamePrefix=args.job)
        for stream in streams.get("logStreams", []):
            name = stream["logStreamName"]
            kwargs = {"logGroupName": group, "logStreamName": name, "startFromHead": True}
            if seen.get(name):
                kwargs["nextToken"] = seen[name]
            elif not args.follow:
                kwargs["limit"] = args.tail
                kwargs["startFromHead"] = False
            response = logs.get_log_events(**kwargs)
            for event in response["events"]:
                print(event["message"])
            seen[name] = response["nextForwardToken"]
        if not args.follow:
            return 0
        status = sm.describe_training_job(TrainingJobName=args.job)["TrainingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            print(f"[job {status}]")
            return 0
        time.sleep(15)


def cmd_download(args: argparse.Namespace) -> int:
    sess = session()
    _, bucket, _ = defaults(sess)
    s3 = sess.client("s3")
    prefix = f"{PREFIX}/checkpoints/{args.job}/"
    target = REPO / "runs" / "sagemaker" / args.job
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".pt") and not args.with_checkpoints:
                continue
            dest = target / key[len(prefix) :]
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            count += 1
    print(f"downloaded {count} files to {target}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    session().client("sagemaker").stop_training_job(TrainingJobName=args.job)
    print(f"stop requested for {args.job}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="launch a training job from the committed HEAD")
    train.add_argument("--config", required=True, help="repo-relative config path")
    train.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    train.add_argument("--instance", default=DEFAULT_INSTANCE)
    train.add_argument("--spot", action="store_true", help="managed spot (cheaper; may resume)")
    train.add_argument("--max-hours", type=float, default=6.0)
    train.add_argument("--eval-episodes", type=int, default=50)
    train.add_argument("--name", help="job name (default: dynabelief-<config>-s<seed>-<utc>)")
    train.add_argument("--role", help="execution role ARN (default: the SageMaker exec role)")
    train.add_argument("--dry-run", action="store_true", help="print the request only")
    train.set_defaults(func=cmd_train)

    status = sub.add_parser("status", help="list recent dynabelief jobs")
    status.add_argument("-n", type=int, default=10)
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="print CloudWatch logs of a job")
    logs.add_argument("job")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--tail", type=int, default=60)
    logs.set_defaults(func=cmd_logs)

    download = sub.add_parser("download", help="fetch a job's synced run files from S3")
    download.add_argument("job")
    download.add_argument("--with-checkpoints", action="store_true")
    download.set_defaults(func=cmd_download)

    stop = sub.add_parser("stop", help="stop a running job")
    stop.add_argument("job")
    stop.set_defaults(func=cmd_stop)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
