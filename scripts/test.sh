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

CKPT="/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph-complete/checkpoint-2000" # 
# CKPT="/group2/dgm/yoonjeon/LaViDa-O"
NUM_GPUS=${NUM_GPUS:-2}

# Force single-node local rendezvous for this test job.
# This avoids inheriting stale cluster addresses (e.g. 10.0.0.1) that can
# cause c10d socket bind/connect timeouts.
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NUM_GPUS: ${NUM_GPUS}"

accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --num_processes "${NUM_GPUS}" \
  validate_thinkmorph_generation.py \
  --num_samples 10 \
  --model_path "${CKPT}" \
  --generation_batch_size 8
