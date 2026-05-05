#!/bin/bash
# Gemma 4 E4B - LoRA-only Fine-tuning
# Freeze: vision_tower + embed_vision (projector) + LLM base weights
# Train: ONLY LoRA adapters on LLM (q/k/v/o + gate/up/down)
# Use when projector is already aligned and you only want to specialize the LLM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-e4b-FT}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"

set -e

MODEL_NAME="${MODEL_NAME:-google/gemma-4-e4b-it}"

DATA_PATH="${DATA_PATH:-./dataset/kinetics_3k/kinetic_3K.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./dataset/kinetics_3k}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/gemma4_e4b_lora_only}"
RUN_NAME="${RUN_NAME:-gemma4-e4b-lora_only}"
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
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_mlp True \
    --freeze_projector True \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --optim "paged_adamw_8bit" \
    \
    --learning_rate 2e-4 \
    --image_encoder_lr 0.0 \
    --projector_lr 0.0 \
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

