#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=rl_thinkmorph-sft-edit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=128:00:00
#SBATCH --requeue
#SBATCH --output=slurm-logs/output.%j.log
#SBATCH --error=slurm-logs/error.%j.log

export WANDB_ENTITY="jeoni"
export WANDB_PROJECT="rl-lavidao-thinkmorph"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-${USER}/${SLURM_JOB_ID:-$$}-${LOCAL_RANK:-0}"

mkdir -p "$TRITON_CACHE_DIR"
chmod 700 "$TRITON_CACHE_DIR"
DATASET="thinkmorph_edit" # thinkmorph, thinkmorph_edit, thinkmorph_grounding
RUN_NAME=${DATASET}-sft
# MODEL_PATH="/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph/checkpoint-26000"
MODEL_PATH="/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph-complete/checkpoint-2420"
# 64 prompts
# × 8 generations
# = 512 trajectories
# split into
# 4 PPO minibatches (128 traj each)
# 2 GEN minibatches (256 traj each)
BATCH_SIZE=64
NUM_GENERATION=8
NUM_PROCESSES=8
PER_DEVICE_BATCH_SIZE=8
PPO_MINIBATCH_SIZE=64
GEN_MINIBATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=$(
  echo $(( 
    BATCH_SIZE * NUM_GENERATION 
    / NUM_PROCESSES 
    / PER_DEVICE_BATCH_SIZE
  ))
)

accelerate launch \
    --config_file ./scripts/accelerate_configs/config.yaml \
    --num_processes $NUM_PROCESSES \
    diffu_grpo_train.py \
    --config ./scripts/rl_train.yaml \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ppo_mini_batch_size $PPO_MINIBATCH_SIZE \
    --num_generations $NUM_GENERATION \
    --generation_batch_size $GEN_MINIBATCH_SIZE \
    --num_iterations 2 \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --resume_from_checkpoint false \
    --guidance_scale 0 \
    --data_root /home/yoonjeon.kim/dLLM-RL/train_sft/data/ \
    --output_dir /group2/dgm/yoonjeon/ckpts/rl-lavidao-thinkmorph/$RUN_NAME
