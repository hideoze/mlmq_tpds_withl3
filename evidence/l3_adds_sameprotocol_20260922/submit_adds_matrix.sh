#!/usr/bin/env bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=adds-samegraph
#SBATCH --output=tmp/l3_adds_sameprotocol_20260922/adds_matrix_%j.out
set -euo pipefail
cd /mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26
python3 scripts/multigpu/run_adds_samegraph_matrix.py tmp/l3_adds_sameprotocol_20260922
