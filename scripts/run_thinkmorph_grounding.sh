#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=thinkmorph_grounding_after_SFT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm-logs/output.%j.log
#SBATCH --error=slurm-logs/error.%j.log

CKPT=${CKPT:-"/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph/checkpoint-26000"} #"/group2/dgm/yoonjeon/LaViDa-O"}
IMAGE_ROOT=${IMAGE_ROOT:-"/group2/dgm/yoonjeon/data/"}
DATA_DIR=${DATA_DIR:-"/music-home-shared-disk/user/yoonjeon.kim/dLLM-RL/data/"}
CKPT_NAME=$(basename "$CKPT")
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/thinkmorph_grounding/${CKPT_NAME}"}
STEPS=${STEPS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_GPUS=${NUM_GPUS:-2}

# ── Environment setup ──
source /music-home-shared-disk/user/yoonjeon.kim/dLLM-RL/train_sft/.venv/bin/activate
cd /music-home-shared-disk/user/yoonjeon.kim/dLLM-RL/train_sft
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p slurm-logs
# ── Run ──
if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch --num_machines 1 --num_processes "$NUM_GPUS" \
        eval/eval_thinkmorph_grounding.py \
        --model "$CKPT" \
        --image_root "$IMAGE_ROOT" \
        --data_dir "$DATA_DIR" \
        --output "$OUTPUT_DIR" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        "$@"
else
    python eval/eval_thinkmorph_grounding.py \
        --model "$CKPT" \
        --image_root "$IMAGE_ROOT" \
        --data_dir "$DATA_DIR" \
        --output "$OUTPUT_DIR" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        "$@"
fi
