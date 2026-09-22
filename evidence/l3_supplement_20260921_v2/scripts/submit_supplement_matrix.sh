#!/usr/bin/env bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=l3-samegraph
#SBATCH --output=tmp/l3_supplement_20260921/matrix_%j.out
set -euo pipefail
cd /mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26
python3 scripts/multigpu/run_supplement_matrix.py "${1:-tmp/l3_supplement_20260921}"
