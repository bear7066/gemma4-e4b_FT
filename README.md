# GemmaFT — Gemma 4 E4B Action Recognition Fine-tuning

Video action-recognition SFT pipeline for `google/gemma-4-e4b-it`
(`Gemma4ForConditionalGeneration`). LoRA on the LLM backbone, trainable
`embed_vision` projector, frozen `vision_tower`, DeepSpeed ZeRO-2, PyAV
video decoding (bypasses system FFmpeg).

## Layout

```
stage1/
  train.py        entrypoint (Gemma4ForConditionalGeneration + LoRA)
  sft.py          GemmaSFTTrainer with per-group LRs
  ds_wrapper.py   SupervisedDataset (messages format, PyAV video I/O)
  utils.py        freeze/unfreeze helpers (model.model.language_model etc.)
  forward.py      no-op (Gemma 4 native forward handles video + loss)
deepspeed_config/stage1.json   ZeRO-2 config
prepare_videochat2.py          VideoChat2-IT → action messages JSON
prepare_k710_gemma4.py         Kinetics-710 → action messages JSON
prepare_webvid_gemmaFT.py      WebVid QA → action + temporal messages JSON
prepare_webvid_openai.py       OpenAI-rewritten short action phrases
upgrade_labels_gpt4o.py        GPT-4o label upgrade pass
smoke_test.py                  synthetic video + forward/backward sanity check
run.sh                         training launcher
```

## Data format (messages JSON)

```json
[
  {
    "video_metadata": {"fps": 25.0, "duration_sec": 8.3},
    "messages": [
      {"role": "user", "content": [
        {"type": "video", "video": "clips/xxx.mp4"},
        {"type": "text",  "text": "What action is performed?"}
      ]},
      {"role": "assistant", "content": [{"type": "text", "text": "riding a bicycle"}]}
    ]
  }
]
```

## Quickstart

```bash
# 1. Smoke test (synthetic MP4, no data needed)
cd ~/test && ~/.local/bin/uv run python3 -u GemmaFT/smoke_test.py

# 2. Prepare an action-recognition dataset (pick one)
~/.local/bin/uv run python3 -u GemmaFT/prepare_videochat2.py \
    --output-dir ~/data/videochat2_action --max-samples 5000
~/.local/bin/uv run python3 -u GemmaFT/prepare_k710_gemma4.py \
    --video-root ~/data/k400 --output ~/data/k710_action.json

# 3. Train
cd ~/test/GemmaFT
DATA_PATH=~/data/videochat2_action/videochat2_action.json \
OUTPUT_DIR=./output/gemma4_e4b_action_stage1 \
sh run.sh
```

## Gemma 4 E4B notes (design choices baked in)

- `Gemma4ForConditionalGeneration` wraps `Gemma4Model` at `.model`; LLM is
  at `model.model.language_model`, vision encoder at
  `model.model.vision_tower`, projector at `model.model.embed_vision`.
- LoRA uses `layers_to_transform=list(range(42))` +
  `layers_pattern="language_model.layers"` to skip
  `Gemma4ClippableLinear` inside the vision encoder. E4B has 42 text
  layers — do not change this unless you move to another Gemma 4 size.
- Video I/O uses PyAV (bundled FFmpeg). `SupervisedDataset._load_video_as_array`
  decodes to numpy `[T,H,W,C]`; the Gemma 4 video processor skips its
  internal FFmpeg path when it sees a 4-D array. Do not switch to
  torchcodec / torchvision video.
- `transformers 5.x` no longer exports `ALL_LAYERNORM_LAYERS`; `sft.py`
  defines it locally as `[nn.LayerNorm]`.
- `run.sh` defaults target a single 32 GB GPU; scale `NUM_GPUS`,
  `gradient_accumulation_steps`, and ZeRO stage as needed.

## Evaluation

Handled in a separate repo (not included here).
