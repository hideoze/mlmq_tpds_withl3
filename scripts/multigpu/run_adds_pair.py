#!/usr/bin/env python3
"""Slurm-only ADDS1/MLMQ2 screening; warmups validated, failures retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_samples(stdout):
    return [dict(part.split("=", 1) for part in line.split()[1:])
            for line in stdout.splitlines() if line.startswith("BENCH ")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adds", required=True, type=Path)
    parser.add_argument("--reference-mlmq", action="store_true",
                        help="--adds is an MLMQ reference binary for ablation or explicit strong scaling")
    parser.add_argument("--reference-gpus", type=int, choices=[1, 2], default=2,
                        help="explicit MLMQ reference GPU count for strong scaling; candidate stays n=2")
    parser.add_argument("--mlmq", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--continue-after-timeout", action="store_true",
                        help="correctness sweeps only: retain timeout failure and try the next run after the GPU-idle check; never retry the timed-out sample")
    parser.add_argument("--delta", type=int, help="explicit positive MLMQ L2 bucket width; does not change ADDS")
    parser.add_argument("--work-blocks", type=int, help="work CTAs per GPU, preserving two GPU participation")
    parser.add_argument("--queue", choices=["L1V_L2DQ", "L1SLF_L2DQ"], default="L1SLF_L2DQ")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU tests must be launched inside Slurm")
    if not args.reference_mlmq and args.reference_gpus != 2:
        parser.error("--reference-gpus applies only with --reference-mlmq; ADDS remains n=1")
    if args.pairs < 1 or args.warmups < 0 or args.timeout < 1 or args.source < 0:
        parser.error("invalid count/source/timeout")
    if args.delta is not None and args.delta <= 0:
        parser.error("delta must be positive")
    if args.work_blocks is not None and args.work_blocks <= 0:
        parser.error("work-blocks must be positive")
    paths = {name: getattr(args, name).resolve(strict=True) for name in ("adds", "mlmq", "graph")}
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"inputs": {key: {"path": str(path), "sha256": sha256(path)}
                           for key, path in paths.items()},
                "configuration": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                "slurm_job": os.environ["SLURM_JOB_ID"],
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "host": os.uname().nodename,
                "environment": {"MLMQ_FINAL_AUDIT": os.environ.get("MLMQ_FINAL_AUDIT"),
                                "MLMQ_DQ_CLAMP_REPORT": os.environ.get("MLMQ_DQ_CLAMP_REPORT")}}
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = dict(os.environ, MLMQ_BENCH="1", BENCH_REPEATS="1",
               BENCH_WARMUPS=str(args.warmups), BENCH_SOURCE=str(args.source), BENCH_QUEUE=args.queue)
    if args.work_blocks is not None:
        env["MLMQ_WORK_BLOCKS"] = str(args.work_blocks)
    else:
        env.pop("MLMQ_WORK_BLOCKS", None)
    failures = 0
    for pair in range(args.pairs):
        for algorithm in (("adds", "mlmq") if pair % 2 == 0 else ("mlmq", "adds")):
            prefix = args.out / f"pair{pair:03d}_{algorithm}"
            snapshot = subprocess.check_output(["nvidia-smi"], text=True)
            prefix.with_suffix(".gpu.log").write_text(snapshot)
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                            "--format=csv,noheader"], text=True).strip()
            if apps:
                raise RuntimeError("GPU process detected; refusing to contend: " + apps)
            command = [str(paths[algorithm])]
            is_adds = algorithm == "adds" and not args.reference_mlmq
            gpu_count = 1 if is_adds else (args.reference_gpus if algorithm == "adds" else 2)
            command += ([str(paths["graph"])] if is_adds else
                        ["-i", str(paths["graph"]), "-n", str(gpu_count)])
            if not is_adds and args.delta is not None:
                command += ["-d", str(args.delta)]
            try:
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
                rc, stdout = result.returncode, result.stdout
            except subprocess.TimeoutExpired as error:
                rc, stdout = 124, error.stdout or b""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
            prefix.with_suffix(".log").write_text(stdout + f"\nRUN_RC={rc}\n")
            rows = parse_samples(stdout)
            expected = "ADDS" if is_adds else "MLMQ"
            valid = (rc == 0 and len(rows) == args.warmups + 1
                     and "Error at node" not in stdout
                     and all(row.get("correct") == "1" and row.get("algorithm") == expected
                             and row.get("gpu_count") == str(gpu_count)
                             and row.get("source") == str(args.source)
                             and row.get("repeat") == str(i)
                             and row.get("warmup") == str(int(i < args.warmups))
                             and (is_adds or row.get("queue") == args.queue)
                             for i, row in enumerate(rows)))
            failures += not valid
            prefix.with_suffix(".json").write_text(json.dumps(
                {"command": command, "rc": rc, "valid": valid, "samples": rows}, indent=2) + "\n")
            print(f"pair={pair} algorithm={algorithm} valid={valid} rc={rc}", flush=True)
            if rc == 124 and not args.continue_after_timeout:
                # Stop on timeout; do not retry and obscure the failure sample.
                return 1
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
