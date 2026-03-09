#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=thinkmorph_sft_complete
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=slurm-logs/output.%j.log
#SBATCH --error=slurm-logs/error.%j.log
export DEBUG=0
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
BATCH_SIZE=32 NUM_GPUS=8 DATA_PATH=./scripts/data/thinkmorph_complete.yaml RUN_NAME=sft-lavidao-thinkmorph-complete bash scripts/train/train_thinkmorph_sft.sh
