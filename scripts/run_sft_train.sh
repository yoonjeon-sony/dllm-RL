#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --account=dgm
#SBATCH --job-name=zebracot_thinkmorph_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=slurm-logs/output.%j.log
#SBATCH --error=slurm-logs/error.%j.log

MODEL_PATH="/group2/dgm/yoonjeon/LaViDa-O"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
DATA_PATH=./scripts/data/thinkmorph_zebracot_complete.yaml
IMG_PATH="/scratch2/yoonjeon.kim/"
RUN_NAME=sft-lavidao-thinkmorph-zebracot-complete
DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SELECT_ONE_INDEX=0
export DEBUG_FIX_PADDING=1
export SELECT_ALL_INDEX=1
export NOT_ALWASY_DO_2DPOOL=1
export SKIP_COMPLEMENTARY_MASKING=1
PROMPT_VERSION="llada"

NUM_CKPT=32

if [[ "${DEBUG}" == "1" ]]; then
    NUM_GPUS=4
    MAX_STEP=10
    LOG_STEP=5
    EVAL_STEP=5
    SAVE_STEP=10
    BATCH_SIZE=2
    EVAL_BATCH_SIZE=1
else
    NUM_GPUS=${NUM_GPUS:-8}
    MAX_STEP=${MAX_STEP:-100000}
    LOG_STEP=${LOG_STEP:-10}
    EVAL_STEP=${EVAL_STEP:-50}
    SAVE_STEP=${SAVE_STEP:-1000}
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
fi
# Default values for single-node training
WORLD_SIZE=1
RANK=0
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

.venv/bin/python -m torch.distributed.run \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --nproc_per_node="${NUM_GPUS}"  \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ./train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path $DATA_PATH \
    --load_vlm True \
    --image_folder $IMG_PATH \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_vision_resampler,mm_language_model,mm_language_model_vision_parms" \
    --mm_vision_tower_lr=2e-6 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --mm_patch_merge_type spatial_unpad \
    --mm_spatial_pool_mode conv \
    --mm_spatial_pool_stride 2 \
    --mm_resampler_type spatial_pool \
    --mm_spatial_pool_out_channels 1152 \
    --add_loc_tokens \
    --add_vision_tokens \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir "/group2/dgm/yoonjeon/ckpts/${RUN_NAME}" \
    --num_train_epochs 10 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps $SAVE_STEP \
    --save_total_limit 30 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --resume_from_checkpoint latest \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --eval_strategy "no" \
    --logging_steps $LOG_STEP \
    --vqvae /group2/dgm/yoonjeon/LaViDa-O/vqvae \
    --unified_gen \
    --dual_tower \
    --prompt_drop_rate 0.1 \
    --image_gen_size 1024 \
    --num_gen_image_tokens 1024 \
    --policy cosine \
    --dual_tower_layers 16 \
    --enc_use_image_branch True \
    --group_by_random_length \
    --num_gen_image_tokens_enc_ds 1 \
    --image_enc_drop_rate 0.5 \
    --lmms_eval_extra_tasks "" \
    --lmms_eval_limit 50 \
    ${@}
