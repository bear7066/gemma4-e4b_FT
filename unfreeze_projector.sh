#!/bin/bash
# Gemma 4 E4B — Caption Fine-tuning (projector-only)
# Freeze vision_tower + LLM completely; train only embed_vision (projector).
# Lightest-touch FT: ~few hundred MB trainable, no LoRA, no LLM disturbance.
# Best when data is small (~3K) and preserving base LLM ability matters.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load secrets from .env (WANDB_API_KEY, etc). gitignored.
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

# wandb settings
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-e4b-FT}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"  # set to "checkpoint" to upload model artifacts

set -e

MODEL_NAME="${MODEL_NAME:-google/gemma-4-e4b-it}"

DATA_PATH="${DATA_PATH:-./kinetic_3K.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$HOME/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/gemma4_e4b_caption}"
RUN_NAME="${RUN_NAME:-gemma4-e4b-caption}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_config/stage1.json}"

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

uv run deepspeed \
    --num_gpus "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    stage1/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    \
    --bf16 True \
    \
    --use_lora False \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --optim "paged_adamw_8bit" \
    \
    --learning_rate 5e-5 \
    --image_encoder_lr 0.0 \
    --projector_lr 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --report_to "wandb"
