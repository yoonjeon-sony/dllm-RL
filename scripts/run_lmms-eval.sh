#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --account=dgm
#SBATCH --job-name=lmmseval-text-vanilla
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=slurm-logs/output.%j.log
#SBATCH --error=slurm-logs/error.%j.log

CHAT_MODE=${CHAT_MODE:-text_gen} # text_gen,image_gen
BATCH_SIZE=${BATCH_SIZE:-16}
# CKPT="/group2/dgm/yoonjeon/ckpts/rl-lavidao-thinkmorph/thinkmorph-sft/checkpoint-250" # # CKPT="/group2/dgm/yoonjeon/LaViDa-O"
# CKPT="/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph/checkpoint-26000"
# CKPT="/group2/dgm/yoonjeon/LaViDa-O"
CKPT="/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph-complete/checkpoint-1000"
LIMIT=${LIMIT:-}
NUM_GPUS=${NUM_GPUS:-4}
TASKS="blink_jigsaw_cot_text_only,vstar_bench_cot_text_only,cv_bench_cot_text_only,VisualPuzzles_cot_text_only,chartqa_cot_text_only"
# TASKS="VisualPuzzles_cot_text_only"
if [[ "${CHAT_MODE}" != "text_gen" && "${CHAT_MODE}" != "image_gen" ]]; then
    echo "Invalid CHAT_MODE=${CHAT_MODE}. Must be text_gen or image_gen."
    exit 1
fi

if [[ -z "${TASKS}" ]]; then
    echo "TASKS must be explicitly provided (e.g., TASKS=chartqa_cot_text_only)."
    exit 1
fi

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
BLOCK_LENGTH=${BLOCK_LENGTH:-256}
STEP_PER_BLOCK=${STEP_PER_BLOCK:-${BLOCK_LENGTH}}
TEMPERATURE=${TEMPERATURE:-0}

export NOT_ALWASY_DO_2DPOOL=1
export DEBUG_PRINT_IMAGE_RES=1
export DEBUG_FIX_PADDING=1

PRE_PROMPT=$'\nLet\'s think step-by-step to solve the question. \nPut your final answer in <answer> </answer> tags.'
LMMS_SPECIFIC_KWARGS="llava_llada:pre_prompt=${PRE_PROMPT},post_prompt="
OUTPUT_DIR="outputs/eval_generate_logs/${CHAT_MODE}_tok${MAX_NEW_TOKENS}_blk${BLOCK_LENGTH}_step${STEP_PER_BLOCK}_t${TEMPERATURE}"
if [ "${NUM_GPUS}" -eq 1 ]; then
    LAUNCH_CMD="python"
    LAUNCH_ARGS="-m lmms_eval"
else
    # Force local rendezvous on single-node Slurm jobs to avoid inheriting
    # stale cluster addresses (e.g. 10.0.0.1:29500) from the environment.
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK
    echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NUM_GPUS=${NUM_GPUS}"

    LAUNCH_CMD="accelerate launch --num_machines=1 --machine_rank=0 --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} --num_processes=${NUM_GPUS}"
    LAUNCH_ARGS="-m lmms_eval"
fi

echo "Running with TASKS=${TASKS} CKPT=${CKPT} BATCH_SIZE=${BATCH_SIZE} CHAT_MODE=${CHAT_MODE}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" && "${LIMIT,,}" != "none" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

${LAUNCH_CMD} ${LAUNCH_ARGS} \
    --model llava_llada \
    --model_args pretrained=$CKPT,conv_template=llada,model_name=llava_llada${CHAT_MODE:+,chat_mode=${CHAT_MODE}},img_gen_save_dir=${OUTPUT_DIR}/gen_imgs,img_gen_conf_policy=stratified,img_gen_edit_mode=0,img_gen_guidance_scale=1.2,img_gen_guidance_scale_image=1.4,img_gen_n_steps=64,img_gen_temperature=0.8,img_gen_enable_stratified=True \
    --tasks "$TASKS" \
    --batch_size ${BATCH_SIZE} \
    --gen_kwargs prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},temperature=${TEMPERATURE} \
    --lmms_eval_specific_kwargs "${LMMS_SPECIFIC_KWARGS}" \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ${OUTPUT_DIR} --verbosity=DEBUG \
    --wandb_args "project=lmms-eval,job_type=eval,name=${EVAL_RUN:-debug}" \
    "${LIMIT_ARGS[@]}" \
    "$@"
