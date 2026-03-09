#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --account=dgm
#SBATCH --job-name=thinkmorph_inference_after_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm-logs/output.%j.log
#SBATCH --error=slurm-logs/error.%j.log

# ── Configurable variables (override from command line or environment) ──
CKPT=${CKPT:-"/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph/checkpoint-26000"}
DATASET=${DATASET:-"ThinkMorph"}
IMAGE_ROOT=${IMAGE_ROOT:-"/group2/dgm/yoonjeon/data/"}
DATA_DIR=${DATA_DIR:-"/music-home-shared-disk/user/yoonjeon.kim/dLLM-RL/data/"}
CKPT_NAME=$(basename "$CKPT")
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/thinkmorph_inference/${CKPT_NAME}"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
NUM_GPUS=${NUM_GPUS:-2}

# Image generation args (only used for interleaved / image_only modes)
EDIT_MODE=${EDIT_MODE:-0}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.2}
GUIDANCE_SCALE_IMAGE=${GUIDANCE_SCALE_IMAGE:-1.4}
CONF_POLICY=${CONF_POLICY:-"stratified"}
BATCH_SIZE=${BATCH_SIZE:-16}

# ── Environment setup ──
source /music-home-shared-disk/user/yoonjeon.kim/dLLM-RL/train_sft/.venv/bin/activate
cd /music-home-shared-disk/user/yoonjeon.kim/dLLM-RL/train_sft
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p slurm-logs

for MODE in "text_only" "image_only" "interleaved"; do # 
    CMD=(
        eval/eval_thinkmorph_inference.py
        --ckpt "$CKPT"
        --dataset_name "$DATASET"
        --mode "$MODE"
        --data_dir "$DATA_DIR"
        --output_dir "$OUTPUT_DIR"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --edit_mode "$EDIT_MODE"
        --guidance_scale "$GUIDANCE_SCALE"
        --guidance_scale_image "$GUIDANCE_SCALE_IMAGE"
        --conf_policy "$CONF_POLICY"
        --enable_stratified
        --batch_size "$BATCH_SIZE"
        --image_root "$IMAGE_ROOT"
    )

    # Append any extra flags passed to this script (e.g. --debug, --enable_stratified)
    CMD+=("$@")

    # ── Run ──
    if [ "$NUM_GPUS" -gt 1 ]; then
        accelerate launch --num_machines 1 --num_processes "$NUM_GPUS" "${CMD[@]}"
    else
        python "${CMD[@]}"
    fi
    sleep 1
done
