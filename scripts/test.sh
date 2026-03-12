#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --account=dgm
#SBATCH --job-name=test_grounding
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=slurm-logs/output.%j.log
#SBATCH --error=slurm-logs/error.%j.log

set -euo pipefail
mkdir -p slurm-logs

NUM_GPUS=${NUM_GPUS:-2}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK

.venv/bin/python -m accelerate.commands.launch \
  --num_processes $NUM_GPUS \
  --main_process_port $MASTER_PORT \
  --main_process_ip $MASTER_ADDR \
  --num_machines 1 \
  --machine_rank 0 \
  inference.py
