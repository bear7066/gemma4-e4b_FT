"""
Gemma 4 Stage 1 Multimodal Fine-tuning
- Model: google/gemma-4-e4b-it (Gemma4ForConditionalGeneration)
- Training: SFT with LoRA via GemmaSFTTrainer + DeepSpeed
- Monitoring: Trackio (real-time metrics)
- Supports: image + video inputs

Usage:
    uv run deepspeed stage1/train.py --deepspeed scripts/stage1.json [args...]
"""

import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from transformers import (
    AutoProcessor,
    Gemma4ForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from stage1.ds_wrapper import make_data_module
from stage1.sft import GemmaSFTTrainer
from stage1.utils import _freeze_llm, _unfreeze_image_encoder, _print_trainable_parameters, _log


@dataclass
class ModelArguments:
    model_id: str = field(
        default="google/gemma-4-e2b-it",
        metadata={"help": "HuggingFace model ID or local path"},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for LLM backbone (recommended for stage 1)"},
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Training data JSON file path"})

    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Validation data JSON file path"},
    )

    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Image/video root directory"},
    )


@dataclass
class Stage1TrainingArguments(TrainingArguments):
    image_encoder_lr: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Vision tower learning rate (0 = frozen)"},
    )

    projector_lr: Optional[float] = field(
        default=1e-6,
        metadata={"help": "embed_vision (projector) learning rate"},
    )

    max_seq_length: int = field(
        default=3072,
        metadata={"help": "Max sequence length"},
    )

    cache_dir: Optional[str] = field(default=None)

    eval_sanity_check_only: bool = field(
        default=False,
        metadata={"help": "Run eval sanity check only and exit before training"},
    )

    eval_chunk_debug: bool = field(
        default=False,
        metadata={"help": "Run chunked eval debug and exit before training"},
    )

    eval_chunk_size: int = field(
        default=32,
        metadata={"help": "Chunk size for eval debug"},
    )

    eval_chunk_start: int = field(
        default=0,
        metadata={"help": "Start index for eval chunk debug"},
    )

    eval_chunk_end: int = field(
        default=-1,
        metadata={"help": "End index for eval chunk debug, -1 means len(eval_dataset)"},
    )


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, Stage1TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.bfloat16
    device = training_args.device

    _log(f"Loading model: {model_args.model_id}")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_args.model_id,
        dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="sdpa",
    )
    
    """
    # Stage 1: freeze LLM, unfreeze embed_vision (projector)
    _freeze_llm(model)
    _unfreeze_image_encoder(model, compute_dtype, device)
    """

    # Full fine-tuning: train all parameters
    for name, param in model.named_parameters():
        param.requires_grad = True

    _log("Full fine-tuning enabled: all parameters trainable")

    # Optionally add LoRA to LLM backbone (keeps it frozen but adds trainable adapters)
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            layers_to_transform=list(range(42)),  # gemma4-e4b has 42 text layers
            layers_pattern="language_model.layers",
        )
        model = get_peft_model(model, lora_config)
        _log("LoRA applied to LLM backbone")

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    model.config.use_cache = False
    model.config.image_encoder_lr = training_args.image_encoder_lr
    model.config.projector_lr = training_args.projector_lr

    # Processor handles image + video tokenization for Gemma 4
    processor = AutoProcessor.from_pretrained(model_args.model_id)

    data_module = make_data_module(
        processor=processor,
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        max_seq_length=training_args.max_seq_length,
    )

    _log(f"data_args.data_path: {data_args.data_path}")
    _log(f"data_args.eval_data_path: {data_args.eval_data_path}")
    _log(f"data_module keys: {list(data_module.keys())}")

    train_dataset = data_module["train_dataset"]
    data_collator = data_module.get("data_collator", None)

    eval_dataset = None

    if data_args.eval_data_path:
        if not os.path.exists(data_args.eval_data_path):
            raise FileNotFoundError(f"eval_data_path not found: {data_args.eval_data_path}")

        eval_data_module = make_data_module(
            processor=processor,
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            max_seq_length=training_args.max_seq_length,
        )

        _log(f"eval_data_module keys: {list(eval_data_module.keys())}")

        # make_data_module 通常只會回傳 train_dataset，
        # 即使用的是 val.json，也還是叫 train_dataset
        eval_dataset = eval_data_module.get("eval_dataset", None)

        if eval_dataset is None:
            eval_dataset = eval_data_module.get("train_dataset", None)

    # Debug：確認真的有載入
    _log(f"train_dataset size: {len(train_dataset)}")
    _log(f"eval_dataset size: {len(eval_dataset) if eval_dataset is not None else None}")

    if training_args.eval_strategy != "no" and eval_dataset is None:
        raise ValueError(
            f"eval_strategy is enabled, but eval_dataset is None. "
            f"data_args.eval_data_path={data_args.eval_data_path}. "
            f"Please check make_data_module return keys."
        )

    trainer = GemmaSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    output_dir = pathlib.Path(training_args.output_dir)
    resume = bool(list(output_dir.glob("checkpoint-*")))
    _log(f"{'Resuming' if resume else 'New training'} → {training_args.output_dir}")

    if training_args.eval_sanity_check_only:
        _log("Running eval sanity check only...")

        if eval_dataset is None:
            raise ValueError("eval_dataset is None")

        _log(f"eval_dataset size: {len(eval_dataset)}")

        bad_no_label = []
        bad_exception = []
        bad_nonfinite_tensor = []
        valid_token_counts = []

        # ============================================================
        # 1. 全掃 eval dataset：確認 labels / pixel_values 沒問題
        # ============================================================
        check_n = min(10, len(eval_dataset))

        for i in range(check_n):
            try:
                sample = eval_dataset[i]
                batch = data_collator([sample]) if data_collator is not None else sample

                if "labels" not in batch:
                    bad_no_label.append(i)
                    continue

                labels = batch["labels"]
                valid_tokens = (labels != -100).sum().item()
                valid_token_counts.append(valid_tokens)

                if valid_tokens == 0:
                    bad_no_label.append(i)

                # 檢查 pixel_values / floating tensor 是否有 nan 或 inf
                for key, value in batch.items():
                    if torch.is_tensor(value) and torch.is_floating_point(value):
                        if not torch.isfinite(value).all():
                            bad_nonfinite_tensor.append((i, key))

            except Exception as e:
                bad_exception.append((i, repr(e)))

            if i % 500 == 0:
                _log(f"checked eval samples: {i}/{len(eval_dataset)}")

        _log(f"num eval samples total: {len(eval_dataset)}")
        _log(f"num eval samples checked: {check_n}")
        _log(f"num bad_no_label: {len(bad_no_label)}")
        _log(f"bad_no_label first 50: {bad_no_label[:50]}")
        _log(f"num bad_exception: {len(bad_exception)}")
        _log(f"bad_exception first 10: {bad_exception[:10]}")
        _log(f"num bad_nonfinite_tensor: {len(bad_nonfinite_tensor)}")
        _log(f"bad_nonfinite_tensor first 10: {bad_nonfinite_tensor[:10]}")

        if valid_token_counts:
            _log(f"min valid label tokens: {min(valid_token_counts)}")
            _log(f"max valid label tokens: {max(valid_token_counts)}")
            _log(f"avg valid label tokens: {sum(valid_token_counts) / len(valid_token_counts)}")

        if bad_no_label:
            raise ValueError(
                f"Found eval samples with 0 valid label tokens. "
                f"First bad indices: {bad_no_label[:20]}"
            )

        if bad_exception:
            raise ValueError(
                f"Found eval samples that crash during collation. "
                f"First bad exceptions: {bad_exception[:5]}"
            )

        if bad_nonfinite_tensor:
            raise ValueError(
                f"Found eval samples with NaN/Inf tensor. "
                f"First bad tensors: {bad_nonfinite_tensor[:10]}"
            )

        _log("Eval label/collator sanity check passed.")

        # ============================================================
        # 2. 不要直接 model(**batch)
        #    DeepSpeed 下請用 trainer.evaluate()
        # ============================================================
        from torch.utils.data import Subset

        small_eval_dataset = Subset(
            eval_dataset,
            list(range(min(10, len(eval_dataset))))
        )

        trainer.eval_dataset = small_eval_dataset
        metrics = trainer.evaluate()

        _log(f"small trainer.evaluate() metrics: {metrics}")

        eval_loss = metrics.get("eval_loss", None)

        if eval_loss is None:
            raise ValueError("trainer.evaluate() did not return eval_loss.")

        if not torch.isfinite(torch.tensor(eval_loss)):
            raise ValueError(
                f"small trainer.evaluate() returned non-finite eval_loss: {eval_loss}"
            )

        _log("Small eval sanity check passed.")
        return
    
    if training_args.eval_chunk_debug:
        _log("Running eval chunk debug only...")

        if eval_dataset is None:
            raise ValueError("eval_dataset is None")

        from torch.utils.data import Subset

        chunk_size = training_args.eval_chunk_size
        debug_start = training_args.eval_chunk_start
        debug_end = (
            len(eval_dataset)
            if training_args.eval_chunk_end < 0
            else min(training_args.eval_chunk_end, len(eval_dataset))
        )

        bad_chunks = []

        for start in range(debug_start, debug_end, chunk_size):
            end = min(start + chunk_size, len(eval_dataset))

            _log(f"Evaluating eval chunk {start}-{end}...")

            subset = Subset(eval_dataset, list(range(start, end)))
            trainer.eval_dataset = subset

            metrics = trainer.evaluate()
            loss = metrics.get("eval_loss", None)

            _log(f"eval chunk {start}-{end}: loss={loss}")

            if loss is None or not torch.isfinite(torch.tensor(loss)):
                bad_chunks.append((start, end, loss))
                _log(f"[BAD] eval chunk {start}-{end}: loss={loss}")
                break

        _log(f"bad_chunks: {bad_chunks}")

        if bad_chunks:
            raise ValueError(f"Found bad eval chunks: {bad_chunks}")

        _log("All eval chunks passed.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

        return

    trainer.train(resume_from_checkpoint=resume)

    trainer.save_state()
    model.config.use_cache = True

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(training_args.output_dir)
    else:
        from peft import PeftModel
        if isinstance(trainer.model, PeftModel):
            # Save only LoRA adapter weights (~32 MB) instead of full model (~8 GB)
            trainer.model.save_pretrained(training_args.output_dir)
        else:
            state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
            trainer._save(training_args.output_dir, state_dict=state_dict)
            trainer.model.config.save_pretrained(training_args.output_dir)

    _log("Training complete, model saved to", training_args.output_dir)


if __name__ == "__main__":
    train()
