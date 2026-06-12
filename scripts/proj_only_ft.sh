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

eval "$(python3 -m config.entry proj_only_ft)"

uv run deepspeed \
    --num_gpus "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    src/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    \
    --bf16 "$BF16" \
    \
    --use_lora "$USE_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --optim "$OPTIM" \
    \
    --learning_rate "$LEARNING_RATE" \
    --image_encoder_lr "$IMAGE_ENCODER_LR" \
    --projector_lr "$PROJECTOR_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    \
    --eval_strategy "$EVAL_STRATEGY" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --logging_steps "$LOGGING_STEPS" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --report_to "$REPORT_TO"
