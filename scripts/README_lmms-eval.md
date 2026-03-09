# LMMS-Eval Runner Guide (`llava_llada`)

This README explains how to run `train_sft/scripts/run_lmms-eval.sh`, what each chat mode does, and the prompt/output behavior used by `train_sft/lmms-eval/lmms_eval/models/llava_llada.py`.

## How to run

Run the orchestrator script:

```bash
bash train_sft/scripts/run_lmms-eval.sh
```

The script runs a single lmms-eval invocation per call. `TASKS` must be explicitly provided.

Each run calls:

```bash
accelerate launch ... -m lmms_eval \
  --model llava_llada \
  --model_args pretrained=$CKPT,conv_template=llada,model_name=llava_llada,chat_mode=$CHAT_MODE \
  --tasks $TASKS ...
```

Common overrides:

```bash
CKPT=/path/to/checkpoint CHAT_MODE=text_gen TASKS=chartqa_cot_text_only BATCH_SIZE=4 LIMIT=10 bash train_sft/scripts/run_lmms-eval.sh
CKPT=/path/to/checkpoint CHAT_MODE=image_gen TASKS=chartqa_cot_image_only BATCH_SIZE=2 LIMIT=10 bash train_sft/scripts/run_lmms-eval.sh
```

## Chat Modes Table

`context` below means `doc_to_text(...)` from task YAML (`question + post_prompt` for ChartQA).

| Chat mode | Task used by script | What the mode does | Prompt construction in `llava_llada.py` | Output format |
|---|---|---|---|---|
| `text_gen` | Explicit via `TASKS` | Standard text generation over original task inputs | If `<image>` token not in context and images exist: `<image>\n<context>` | Decoded assistant text (normalized with `lstrip('!')` and `strip()`) |
| `image_gen` | Explicit via `TASKS` | Two-stage flow: predict bbox -> generate edited image -> final text answer from edited image | First pass adds `image_gen_post_prompt` for bbox prediction only; final text pass excludes it | Returns text output and saves generated image (`gen_img_path`) |

Notes:

- Only `chat_mode=text_gen|image_gen` is supported.
- In `image_gen`, each sample must contain at least one image; otherwise generation raises an error.
- If bbox parsing fails in first pass, generation falls back to full-image edit masking.
- If `context` is JSON, `generate_until` still supports multi-turn-style serialization.

## Multi-round generation status

Task selection is explicit via `TASKS` and can target any lmms-eval task compatible with the selected chat mode.

Examples:

- `CHAT_MODE=text_gen TASKS=chartqa_cot_text_only`
- `CHAT_MODE=image_gen TASKS=chartqa_cot_image_only`
