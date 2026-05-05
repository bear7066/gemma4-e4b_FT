---
dataset_info:
  splits:
    - name: train
      num_examples: 3115
---

# gemma-3-4b-kinetics_3K

Kinetics action recognition dataset, formatted for Gemma 3-4B fine-tuning.

- **3,115 samples** — Kinetics video clips sampled from YouTube
- **Format**: 4 sequential frames per clip (JPEG 336×336) + descriptive answer
- **Frames**: stored in `frames/<clip_id>/frame_{1,2,3,4}.jpg`
- **Annotation**: `kinetic_3K.json` — messages format, ready for training

## File Sizes

| File | Size |
|---|---|
| `frames/` (12,460 JPEGs) | 274 MB |
| `kinetic_3K.json` | 3.0 MB |
| **Total** | **~277 MB** |

## Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "frames/<clip_id>/frame_1.jpg"},
        {"type": "image", "image": "frames/<clip_id>/frame_2.jpg"},
        {"type": "image", "image": "frames/<clip_id>/frame_3.jpg"},
        {"type": "image", "image": "frames/<clip_id>/frame_4.jpg"},
        {"type": "text",  "text": "Please analyze the sequence of frames from this video. What specific action or event is happening?"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "A man is in an office doing some kind of welding of an object"}
      ]
    }
  ]
}
```

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bear7011/gemma-3-4b-kinetics_3K",
    repo_type="dataset",
    local_dir="./kinetics_3k",
)
```

## Usage with GemmaFT

```python
from stage1.ds_wrapper import make_data_module
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
data_module = make_data_module(
    processor=processor,
    data_path="./kinetics_3k/kinetic_3K.json",
    image_folder="./kinetics_3k",
)
```
