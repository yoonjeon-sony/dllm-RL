#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError as e:
    raise SystemExit(
        "Missing dependency: cv2. Install with `pip install opencv-python`."
    ) from e

try:
    import numpy as np
except ImportError as e:
    raise SystemExit(
        "Missing dependency: numpy. Install with `pip install numpy`."
    ) from e

try:
    import torch
except ImportError as e:
    raise SystemExit(
        "Missing dependency: torch. Install with `pip install torch`."
    ) from e

try:
    from datasets import Dataset
except ImportError as e:
    raise SystemExit(
        "Missing dependency: datasets. Install with `pip install datasets`."
    ) from e

from PIL import Image

from data_utils import (
    EDIT_PROMPT,
    _build_grounding_prompt,
    _build_question_prompt,
    convert_to_rgb,
)
from diffu_grpo_config import DiffuGRPOConfig
from diffu_grpo_train import MyProcessor
from diffu_grpo_trainer import DiffuGRPOTrainer
from llava.mm_utils import pad_to_square_and_resize
from llava.model.builder import load_pretrained_model

VALIDATION_MODES = ("text_gen", "grounding", "image_gen")


def _get_model_device_stats(model) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "num_params": 0,
        "num_trainable_params": 0,
        "device_counts": {},
        "dtype_counts": {},
    }
    try:
        for p in model.parameters():
            n = int(p.numel())
            stats["num_params"] += n
            if p.requires_grad:
                stats["num_trainable_params"] += n
            dev = str(p.device)
            dt = str(p.dtype)
            stats["device_counts"][dev] = int(stats["device_counts"].get(dev, 0)) + n
            stats["dtype_counts"][dt] = int(stats["dtype_counts"].get(dt, 0)) + n
    except Exception as e:
        stats["error"] = f"Failed to inspect model parameters: {e}"
    return stats


def _print_accelerate_debug(stage: str, model, trainer=None, local_rank: Optional[int] = None) -> None:
    prefix = f"[accelerate-debug][{stage}]"
    print(
        f"{prefix} env: LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"RANK={os.environ.get('RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    print(
        f"{prefix} torch.cuda: available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} current_device="
        f"{torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'} "
        f"requested_local_rank={local_rank}"
    )
    if torch.cuda.is_available():
        try:
            print(f"{prefix} torch.cuda.current_device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
        except Exception:
            pass

    model_stats = _get_model_device_stats(model)
    print(
        f"{prefix} raw_model_stats: num_params={model_stats.get('num_params')} "
        f"num_trainable={model_stats.get('num_trainable_params')} "
        f"device_counts={model_stats.get('device_counts')} "
        f"dtype_counts={model_stats.get('dtype_counts')}"
    )
    if "error" in model_stats:
        print(f"{prefix} raw_model_stats_error={model_stats['error']}")

    if trainer is None:
        return

    acc = trainer.accelerator
    print(
        f"{prefix} accelerator: device={acc.device} "
        f"process_index={acc.process_index} local_process_index={acc.local_process_index} "
        f"num_processes={acc.num_processes} distributed_type={acc.distributed_type}"
    )

    wrapped_stats = _get_model_device_stats(trainer.model)
    print(
        f"{prefix} wrapped_model_stats: num_params={wrapped_stats.get('num_params')} "
        f"device_counts={wrapped_stats.get('device_counts')} "
        f"dtype_counts={wrapped_stats.get('dtype_counts')}"
    )
    if "error" in wrapped_stats:
        print(f"{prefix} wrapped_model_stats_error={wrapped_stats['error']}")

    try:
        unwrapped = acc.unwrap_model(trainer.model)
        unwrapped_stats = _get_model_device_stats(unwrapped)
        print(
            f"{prefix} unwrapped_model_stats: num_params={unwrapped_stats.get('num_params')} "
            f"device_counts={unwrapped_stats.get('device_counts')} "
            f"dtype_counts={unwrapped_stats.get('dtype_counts')}"
        )
    except Exception as e:
        print(f"{prefix} unwrap_error={e}")

    if torch.cuda.is_available():
        if acc.device.type != "cuda":
            print(
                f"{prefix} WARNING: CUDA is available but accelerator.device is {acc.device}. "
                "This usually means launch was not distributed/cuda-aware."
            )
        if "cpu" in str(wrapped_stats.get("device_counts", {})):
            print(
                f"{prefix} WARNING: wrapped model still has CPU params. "
                "Likely accelerator/model placement mismatch."
            )


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ThinkMorph text_gen/grounding/image_gen via DiffuGRPOTrainer._generate_and_score_completions"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph/checkpoint-26000",
    )
    parser.add_argument("--data_root", type=str, default="/home/yoonjeon.kim/dLLM-RL/train_sft/data")
    parser.add_argument("--image_root", type=str, default="/group2/dgm/yoonjeon/data")
    parser.add_argument("--output_dir", type=str, default="outputs/thinkmorph_mode_validation")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_resolution", type=int, default=1024)

    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    parser.add_argument("--guidance_scale_image", type=float, default=1.4)
    parser.add_argument("--edit_mode", type=int, default=0)

    parser.add_argument("--run_image_gen", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--save_raw_edited", type=str2bool, nargs="?", const=True, default=True)

    return parser.parse_args()


def noop_reward(prompts, completions, **kwargs) -> List[float]:
    return [0.0 for _ in completions]


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_image_path(image_path: str, image_root: str) -> str:
    if not image_path:
        raise ValueError("Missing problem_image_0 path.")
    img_path = Path(str(image_path))
    if img_path.is_absolute():
        return str(img_path)
    return str(Path(image_root) / img_path)


def _build_grounding_prompt_with_inference(
    question: str, grounding_inference: Optional[str]
) -> str:
    if grounding_inference is None:
        return _build_grounding_prompt(question)
    inference = str(grounding_inference).strip()
    if not inference:
        return _build_grounding_prompt(question)

    # If the inference already looks like a full prompt, pass it through.
    if "<|startoftext|>" in inference:
        return inference

    # Otherwise, treat inference as a LOC string to be inserted into the prompt.
    inference = inference.replace("<|eot_id|>", "").strip()
    template = _build_grounding_prompt(question)
    placeholder = "<LOC_BEGIN><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><LOC_END>"
    if "<LOC_BEGIN>" in inference and placeholder in template:
        return template.replace(placeholder, inference)
    return template


def load_thinkmorph_val_rows(
    jsonl_path: Path,
    image_root: str,
    dataset_name: str,
    limit: int,
    use_grounding_inference: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            question = example.get("question", "")
            if not question:
                continue

            image_path = _resolve_image_path(example.get("problem_image_0", ""), image_root)
            image = convert_to_rgb(image_path)
            prompt = _build_question_prompt(question)
            if use_grounding_inference:
                grounding_prompt = _build_grounding_prompt_with_inference(
                    question, example.get("grounding_inference")
                )
            else:
                grounding_prompt = _build_grounding_prompt(question)
            edit_prompt = f"{EDIT_PROMPT} {question}"

            row = {
                "prompt": prompt,
                "edit_prompt": edit_prompt,
                "grounding_prompt": grounding_prompt,
                "answer": example.get("answer"),
                "gt_bbox": example.get("raw_gt_box"),
                "image": image,
                "gen_type": "image_gen",
                "dataset_name": dataset_name,
                "pid": example.get("pid"),
                "problem_image_0": image_path,
            }
            if use_grounding_inference:
                row["grounding_inference"] = example.get("grounding_inference")

            rows.append(row)
            if len(rows) >= limit:
                break

    if len(rows) < limit:
        raise RuntimeError(
            f"{dataset_name}: requested {limit} samples but only {len(rows)} available in {jsonl_path}."
        )
    return rows



def dedupe_rows_by_sample_gen(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        gen_type = str(row.get("gen_type", "text_gen"))
        key = (sample_id, gen_type)
        if key not in dedup:
            dedup[key] = row
    return [dedup[key] for key in sorted(dedup.keys(), key=lambda x: (x[0], x[1]))]


def build_mode_rows(
    sampled: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
    mode: str,
) -> List[Dict[str, Any]]:
    mode_rows: List[Dict[str, Any]] = []
    for sample_id, (key, text_ex, ground_ex, image_ex) in enumerate(sampled):
        if mode == "text_gen":
            ex = dict(text_ex)
        elif mode == "grounding":
            ex = dict(ground_ex)
        elif mode == "image_gen":
            ex = dict(image_ex)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        ex["sample_id"] = sample_id
        ex["grounding_prompt_key"] = key
        ex["gen_type"] = mode
        image = ex.get("image")
        ex["images"] = [image] if image is not None else None
        ex["is_padding_row"] = False
        mode_rows.append(ex)
    return mode_rows


def build_concatenated_rows(
    sampled: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
    sampler_chunk_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode in VALIDATION_MODES:
        rows.extend(build_mode_rows(sampled, mode))

    if sampler_chunk_size <= 0:
        return rows
    if not rows:
        return rows

    remainder = len(rows) % sampler_chunk_size
    if remainder == 0:
        return rows

    pad_count = sampler_chunk_size - remainder
    source_len = len(rows)
    for i in range(pad_count):
        padded = dict(rows[i % source_len])
        padded["is_padding_row"] = True
        rows.append(padded)
    return rows


def group_examples_by_gen_type(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in VALIDATION_MODES}
    for ex in examples:
        gen_type = str(ex.get("gen_type", "text_gen"))
        if gen_type not in grouped:
            raise ValueError(f"Unsupported gen_type in batch: {gen_type}")
        grouped[gen_type].append(ex)
    return grouped


def parse_loc_bbox(text: str) -> Tuple[Optional[List[int]], Optional[str]]:
    vals = [int(v) for v in re.findall(r"<LOC_([0-9]+)>", text)]
    if len(vals) < 4:
        return None, "Could not parse 4 LOC tokens from completion."
    return vals[:4], None


def parse_any_bbox(bbox: Any) -> Tuple[Optional[List[float]], Optional[str]]:
    if bbox is None:
        return None, "gt_bbox is missing."
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+", bbox)
            if len(nums) < 4:
                return None, f"Could not parse bbox from string: {bbox!r}"
            bbox = [float(x) for x in nums[:4]]
    if isinstance(bbox, (list, tuple, np.ndarray)):
        if len(bbox) < 4:
            return None, f"Expected 4 bbox values, got {len(bbox)}."
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], None
        except Exception as e:
            return None, f"Failed to cast bbox values to float: {e}"
    return None, f"Unsupported bbox type: {type(bbox)}"


def project_bbox_to_1024_canvas(
    bbox_xyxy: List[float], image_size: Tuple[int, int], image_resolution: int
) -> List[int]:
    width, height = image_size
    max_side = float(max(width, height))
    pad_x = (max_side - float(width)) / 2.0
    pad_y = (max_side - float(height)) / 2.0

    x1, y1, x2, y2 = bbox_xyxy
    x1 = (x1 + pad_x) * image_resolution / max_side
    y1 = (y1 + pad_y) * image_resolution / max_side
    x2 = (x2 + pad_x) * image_resolution / max_side
    y2 = (y2 + pad_y) * image_resolution / max_side

    return [
        int(np.clip(round(x1), 0, image_resolution - 1)),
        int(np.clip(round(y1), 0, image_resolution - 1)),
        int(np.clip(round(x2), 0, image_resolution - 1)),
        int(np.clip(round(y2), 0, image_resolution - 1)),
    ]


def normalize_pred_bbox_1024(pred_bbox: Any, image_resolution: int) -> Optional[List[int]]:
    if not isinstance(pred_bbox, (list, tuple)) or len(pred_bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(pred_bbox[0]), float(pred_bbox[1]), float(pred_bbox[2]), float(pred_bbox[3])]
    except Exception:
        return None
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    x0 = int(np.clip(round(x0), 0, image_resolution - 1))
    y0 = int(np.clip(round(y0), 0, image_resolution - 1))
    x1 = int(np.clip(round(x1), 0, image_resolution - 1))
    y1 = int(np.clip(round(y1), 0, image_resolution - 1))
    if x1 <= x0:
        x1 = min(image_resolution - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(image_resolution - 1, y0 + 1)
    return [x0, y0, x1, y1]


def draw_bbox_overlay_1024(
    image,
    pred_bbox_xyxy: Optional[List[int]],
    gt_bbox_xyxy: Optional[List[int]],
    image_resolution: int,
) -> np.ndarray:
    vis = pad_to_square_and_resize(image.convert("RGB"), image_resolution)
    vis_np = np.array(vis)
    vis_bgr = cv2.cvtColor(vis_np, cv2.COLOR_RGB2BGR)

    if pred_bbox_xyxy is not None:
        x, y, w, h = pred_bbox_xyxy
        x = int(np.clip(x, 0, image_resolution - 1))
        y = int(np.clip(y, 0, image_resolution - 1))
        w = int(np.clip(w, 0, image_resolution - 1))
        h = int(np.clip(h, 0, image_resolution - 1))
        cv2.rectangle(vis_bgr, (x, y), (w, h), (0, 255, 0), 2)

    if gt_bbox_xyxy is not None:
        x, y, w, h = gt_bbox_xyxy
        x = int(np.clip(x, 0, image_resolution - 1))
        y = int(np.clip(y, 0, image_resolution - 1))
        w = int(np.clip(w, 0, image_resolution - 1))
        h = int(np.clip(h, 0, image_resolution - 1))
        cv2.rectangle(vis_bgr, (x, y), (w, h), (0, 0, 255), 2)

    return vis_bgr


def draw_mask_overlay_64(
    image,
    mask_flat: List[int],
    grid_shape: Tuple[int, int],
    image_resolution: int,
) -> np.ndarray:
    gh, gw = grid_shape
    if gh * gw != len(mask_flat):
        raise ValueError(f"mask_flat length mismatch: got {len(mask_flat)}, expected {gh*gw}.")

    base_1024 = pad_to_square_and_resize(image.convert("RGB"), image_resolution)
    base_64 = base_1024.resize((gw, gh), resample=Image.BILINEAR)
    arr = np.array(base_64).astype(np.float32)

    mask_arr = np.array(mask_flat, dtype=np.uint8).reshape(gh, gw) > 0
    gray = np.array([220.0, 220.0, 220.0], dtype=np.float32)
    arr[mask_arr] = 0.7 * arr[mask_arr] + 0.3 * gray

    out_rgb = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def align_samples_three(
    ds_text: Dataset, ds_ground: Dataset, ds_image: Dataset
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    text_map: Dict[str, deque] = defaultdict(deque)
    ground_map: Dict[str, deque] = defaultdict(deque)
    image_map: Dict[str, deque] = defaultdict(deque)

    for ex in ds_text:
        text_map[ex["grounding_prompt"]].append(ex)
    for ex in ds_ground:
        ground_map[ex["grounding_prompt"]].append(ex)
    for ex in ds_image:
        image_map[ex["grounding_prompt"]].append(ex)

    aligned: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    common_keys = sorted(set(text_map.keys()) & set(ground_map.keys()) & set(image_map.keys()))
    for key in common_keys:
        n = min(len(text_map[key]), len(ground_map[key]), len(image_map[key]))
        for _ in range(n):
            aligned.append(
                (
                    key,
                    text_map[key].popleft(),
                    ground_map[key].popleft(),
                    image_map[key].popleft(),
                )
            )
    return aligned


def build_trainer(
    model_path: str,
    output_dir: Path,
    max_completion_length: int,
    block_length: int,
    diffusion_steps: int,
    temperature: float,
    generation_batch_size: int,
    num_generations: int,
    guidance_scale: float,
    guidance_scale_image: float,
    edit_mode: int,
    train_dataset: Dataset,
) -> Tuple[DiffuGRPOTrainer, MyProcessor]:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        "llava_llada",
        attn_implementation="sdpa",
        device_map="cpu",
        torch_dtype="bfloat16",
    )

    model.tie_weights()
    model.to(torch.bfloat16)
    model.to(device)
    model.config.use_cache = False

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    processor = MyProcessor(model, tokenizer, image_processor)

    cfg = DiffuGRPOConfig(
        output_dir=str(output_dir / "_tmp_trainer"),
        run_name="thinkmorph_mode_validation",
        dataset="thinkmorph",
        remove_unused_columns=False,
        max_prompt_length=512,
        max_completion_length=max_completion_length,
        block_length=block_length,
        diffusion_steps=diffusion_steps,
        temperature=temperature,
        generation_batch_size=generation_batch_size,
        num_generations=num_generations,
        num_iterations=1,
        beta=0.0,
        log_completions=False,
        report_to=[],
        logging_steps=10_000_000,
        save_strategy="no",
        eval_strategy="no",
        per_device_train_batch_size=1,
        bf16=torch.cuda.is_available(),
        guidance_scale=guidance_scale,
        guidance_scale_image=guidance_scale_image,
        edit_mode=edit_mode,
    )
    trainer = DiffuGRPOTrainer(
        args=cfg,
        model=model,
        reward_funcs=[noop_reward],
        train_dataset=train_dataset,
        processing_class=processor,
    )
    trainer.model.eval()
    return trainer, processor


def main() -> None:
    args = parse_args()
    if args.num_generations < 2:
        raise ValueError("--num_generations must be >= 2 for GRPO.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_out_dir = Path(args.output_dir)
    model_parts = Path(args.model_path).parts
    if len(model_parts) >= 2:
        run_suffix = f"{model_parts[-2]}_{model_parts[-1]}"
    else:
        run_suffix = Path(args.model_path).name or "run"
    out_dir = base_out_dir / run_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    grounding_overlay_dir = out_dir / "grounding_overlays"
    grounding_overlay_dir.mkdir(parents=True, exist_ok=True)

    image_stage1_dir = out_dir / "image_gen_stage1_bbox"
    image_stage2_dir = out_dir / "image_gen_stage2_edited_bbox"
    image_stage3_dir = out_dir / "image_gen_stage3_mask64"
    image_stage2_raw_dir = out_dir / "image_gen_stage2_raw"
    image_stage1_dir.mkdir(parents=True, exist_ok=True)
    image_stage2_dir.mkdir(parents=True, exist_ok=True)
    image_stage3_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw_edited:
        image_stage2_raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.run_image_gen:
        print("`--run_image_gen=false` is ignored in unified mode. Running text_gen/grounding/image_gen together.")

    dataset_specs = [
        ("Jigsaw_Assembly", "ThinkMorph-Jigsaw_Assembly_loc_val.jsonl", False),
        ("Visual_Search", "ThinkMorph-Visual_Search_loc_val.jsonl", True),
        ("Spatial_Navigation", "ThinkMorph-Spatial_Navigation_loc_val.jsonl", True),
        ("Chart_Refocus", "ThinkMorph-Chart_Refocus_loc_val.jsonl", True),
    ]

    samples_per_dataset = args.num_samples
    all_rows: List[Dict[str, Any]] = []
    for dataset_name, filename, use_grounding_inference in dataset_specs:
        jsonl_path = Path(args.data_root) / filename
        if not jsonl_path.exists():
            raise SystemExit(
                f"Dataset file not found: {jsonl_path}. Check --data_root."
            )
        dataset_rows = load_thinkmorph_val_rows(
            jsonl_path=jsonl_path,
            image_root=args.image_root,
            dataset_name=dataset_name,
            limit=samples_per_dataset,
            use_grounding_inference=use_grounding_inference,
        )
        all_rows.extend(dataset_rows)

    sampled_rows: List[Dict[str, Any]] = []
    for sample_id, ex in enumerate(all_rows):
        row = dict(ex)
        row["sample_id"] = sample_id
        row["grounding_prompt_key"] = row.get("grounding_prompt")
        image = row.get("image")
        row["images"] = [image] if image is not None else None
        row["is_padding_row"] = False
        sampled_rows.append(row)
    sampled_dataset = Dataset.from_list(sampled_rows)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    trainer, processor = build_trainer(
        model_path=args.model_path,
        output_dir=out_dir,
        max_completion_length=args.max_completion_length,
        block_length=args.block_length,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature,
        generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations,
        guidance_scale=args.guidance_scale,
        guidance_scale_image=args.guidance_scale_image,
        edit_mode=args.edit_mode,
        train_dataset=sampled_dataset,
    )

    accelerator = trainer.accelerator
    world_size = int(accelerator.num_processes)
    rank = int(accelerator.process_index)
    local_rank = int(accelerator.local_process_index)

    text_rows_local: List[Dict[str, Any]] = []
    grounding_rows_local: List[Dict[str, Any]] = []
    image_rows_local: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[int, str]] = set()
    total_samples = len(sampled_rows)
    target_keys = {(sample_id, "image_gen") for sample_id in range(total_samples)}
    sampler_chunk_size: Optional[int] = None
    effective_sampler_chunk_size: Optional[int] = None
    concatenated_rows = sampled_rows
    checked_dataloader_contract = False

    train_dataloader = trainer.get_train_dataloader()
    for batch in train_dataloader:
        examples = batch
        if len(examples) == 0:
            continue

        if not checked_dataloader_contract:
            required_keys = {"prompt", "grounding_prompt", "edit_prompt", "images", "gen_type"}
            missing_keys = sorted(required_keys - set(examples[0].keys()))
            if missing_keys:
                raise RuntimeError(f"Missing required dataloader keys: {missing_keys}")
            checked_dataloader_contract = True

        mode = str(examples[0].get("gen_type", "text_gen"))
        mode_inputs = examples
        if mode != "image_gen":
            raise RuntimeError(f"Unsupported mode in this validator: {mode}. Expected only image_gen rows.")

        if mode == "image_gen":
            mode_outputs = trainer._generate_and_score_completions(
                mode_inputs, return_debug_artifacts=True
            )
        else:
            mode_outputs = trainer._generate_and_score_completions(mode_inputs)
    

        final_generated_texts = processor.tokenizer.batch_decode(
            mode_outputs["completion_ids"],
            skip_special_tokens=True,
        )
        pred_bboxes = mode_outputs.get("pred_bboxes", [None] * len(mode_inputs))
        edited_images = mode_outputs.get("edited_images", [None] * len(mode_inputs))
        image_gen_debug = mode_outputs.get("image_gen_debug", [None] * len(mode_inputs))
        grounding_bbox_texts = mode_outputs.get(
            "grounding_bbox_completion_text", [None] * len(mode_inputs)
        )

        for local_idx, ex in enumerate(mode_inputs):
            sample_id = int(ex["sample_id"])
            key = (sample_id, mode)
            if key in seen_keys:
                continue

            pred_bbox_1024 = normalize_pred_bbox_1024(pred_bboxes[local_idx], args.image_resolution)
            gt_raw = ex.get("gt_bbox")
            gt_parsed, gt_parse_error = parse_any_bbox(gt_raw)
            gt_bbox_1024 = None
            if gt_parsed is not None:
                gt_bbox_1024 = project_bbox_to_1024_canvas(
                    gt_parsed, ex["image"].size, args.image_resolution
                )

            stage1_path = image_stage1_dir / f"sample_{sample_id:03d}.png"
            stage1_bgr = draw_bbox_overlay_1024(
                ex["image"],
                pred_bbox_xyxy=pred_bbox_1024,
                gt_bbox_xyxy=gt_bbox_1024,
                image_resolution=args.image_resolution,
            )
            cv2.imwrite(str(stage1_path), stage1_bgr)

            edited_image = edited_images[local_idx] if local_idx < len(edited_images) else None
            if not isinstance(edited_image, Image.Image):
                raise RuntimeError(f"image_gen sample {sample_id}: edited image is missing or invalid type.")

            stage2_path = image_stage2_dir / f"sample_{sample_id:03d}.png"
            stage2_bgr = draw_bbox_overlay_1024(
                edited_image,
                pred_bbox_xyxy=pred_bbox_1024,
                gt_bbox_xyxy=gt_bbox_1024,
                image_resolution=args.image_resolution,
            )
            cv2.imwrite(str(stage2_path), stage2_bgr)

            raw_stage2_path = None
            if args.save_raw_edited:
                raw_stage2_path = image_stage2_raw_dir / f"sample_{sample_id:03d}.png"
                edited_image.save(raw_stage2_path)

            debug_item = image_gen_debug[local_idx] if local_idx < len(image_gen_debug) else None
            if not isinstance(debug_item, dict):
                raise RuntimeError(f"image_gen sample {sample_id}: missing debug payload.")

            mask_shape = debug_item.get("mask_idx_2d_shape")
            if not isinstance(mask_shape, (list, tuple)) or len(mask_shape) != 2:
                raise RuntimeError(f"image_gen sample {sample_id}: invalid mask_idx_2d_shape={mask_shape}.")
            if int(mask_shape[1]) != 4096:
                raise RuntimeError(
                    f"image_gen sample {sample_id}: expected mask second dim 4096, got {mask_shape}."
                )

            grid_shape = debug_item.get("grid_shape", [64, 64])
            if not isinstance(grid_shape, (list, tuple)) or len(grid_shape) != 2:
                raise RuntimeError(f"image_gen sample {sample_id}: invalid grid_shape={grid_shape}.")
            gh, gw = int(grid_shape[0]), int(grid_shape[1])

            tokens = debug_item.get("tokens", [])
            if not isinstance(tokens, list):
                raise RuntimeError(f"image_gen sample {sample_id}: tokens must be a list.")
            if any((not isinstance(t, int)) or t < 0 or t >= 4096 for t in tokens):
                raise RuntimeError(f"image_gen sample {sample_id}: token index out of range in tokens.")

            n_mask_tokens = int(debug_item.get("n_mask_tokens", len(tokens)))
            if n_mask_tokens != len(tokens):
                raise RuntimeError(
                    f"image_gen sample {sample_id}: n_mask_tokens ({n_mask_tokens}) != len(tokens) ({len(tokens)})."
                )

            mask_flat = debug_item.get("mask_idx_2d_flat")
            if mask_flat is None:
                mask_flat = [0] * (gh * gw)
                for tok in tokens:
                    if 0 <= tok < len(mask_flat):
                        mask_flat[tok] = 1
            if len(mask_flat) != gh * gw:
                raise RuntimeError(
                    f"image_gen sample {sample_id}: mask length mismatch {len(mask_flat)} vs {gh*gw}."
                )
            mask_token_indices = [idx for idx, val in enumerate(mask_flat) if int(val) > 0]
            if set(mask_token_indices) != set(tokens):
                raise RuntimeError(
                    f"image_gen sample {sample_id}: tokens do not match mask_idx_2d active indices."
                )

            grid_bbox = debug_item.get("grid_bbox_xyxy")
            if isinstance(grid_bbox, (list, tuple)) and len(grid_bbox) == 4:
                gx0, gy0, gx1, gy1 = [int(v) for v in grid_bbox]
                expected_tokens = {
                    r * gw + c
                    for r in range(max(0, gy0), min(gh, gy1))
                    for c in range(max(0, gx0), min(gw, gx1))
                }
                if expected_tokens and set(tokens) != expected_tokens:
                    raise RuntimeError(
                        f"image_gen sample {sample_id}: tokens do not match rectangular grid_bbox region."
                    )

            stage3_path = image_stage3_dir / f"sample_{sample_id:03d}.png"
            stage3_bgr = draw_mask_overlay_64(
                ex["image"],
                mask_flat=[int(v) for v in mask_flat],
                grid_shape=(gh, gw),
                image_resolution=args.image_resolution,
            )
            cv2.imwrite(str(stage3_path), stage3_bgr)

            final_text = final_generated_texts[local_idx] if local_idx < len(final_generated_texts) else ""
            if not final_text.strip():
                raise RuntimeError(f"image_gen sample {sample_id}: final generated text is empty.")

            parse_error = gt_parse_error if gt_parse_error is not None else None

            image_rows_local.append(
                {
                    "sample_id": sample_id,
                    "gen_type": mode,
                    "grounding_prompt_key": ex["grounding_prompt_key"],
                    "grounding_prompt_after_template": ex.get("grounding_prompt"),
                    "edit_prompt_after_template": ex.get("edit_prompt"),
                    "prompt_after_template": ex.get("prompt"),
                    "grounding_bbox_completion_text_full": grounding_bbox_texts[local_idx],
                    "pred_bbox_xyxy_1024": pred_bbox_1024,
                    "gt_bbox": gt_raw,
                    "gt_bbox_xyxy_1024": gt_bbox_1024,
                    "grid_bbox_xyxy": grid_bbox,
                    "mask_idx_2d_shape": mask_shape,
                    "grid_shape": [gh, gw],
                    "n_mask_tokens": n_mask_tokens,
                    "tokens": tokens,
                    "xt_mask_count_after_apply": debug_item.get("xt_mask_count_after_apply"),
                    "image_sizes": debug_item.get("image_sizes"),
                    "final_generated_text": final_text,
                    "answer": ex.get("answer"),
                    "stage1_overlay_path": str(stage1_path),
                    "stage2_overlay_path": str(stage2_path),
                    "stage3_mask64_path": str(stage3_path),
                    "stage2_raw_path": str(raw_stage2_path) if raw_stage2_path is not None else None,
                    "parse_error": parse_error,
                }
            )
            seen_keys.add(key)

        if len(seen_keys) == len(target_keys):
            break

    missing_keys = target_keys - seen_keys
    if missing_keys and world_size == 1:
        raise RuntimeError(
            f"Generation loop ended before collecting all target keys. Missing count={len(missing_keys)}."
        )

    rank_cache_dir = out_dir / "_rank_cache"
    rank_cache_dir.mkdir(parents=True, exist_ok=True)
    text_local_path = rank_cache_dir / f"text_rank{rank:03d}.jsonl"
    grounding_local_path = rank_cache_dir / f"grounding_rank{rank:03d}.jsonl"
    image_local_path = rank_cache_dir / f"image_rank{rank:03d}.jsonl"

    save_jsonl(text_local_path, text_rows_local)
    save_jsonl(grounding_local_path, grounding_rows_local)
    save_jsonl(image_local_path, image_rows_local)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        text_rows: List[Dict[str, Any]] = []
        grounding_rows: List[Dict[str, Any]] = []
        image_rows: List[Dict[str, Any]] = []

        for r in range(world_size):
            text_path_r = rank_cache_dir / f"text_rank{r:03d}.jsonl"
            grounding_path_r = rank_cache_dir / f"grounding_rank{r:03d}.jsonl"
            image_path_r = rank_cache_dir / f"image_rank{r:03d}.jsonl"

            if text_path_r.exists():
                text_rows.extend(load_jsonl(text_path_r))
            if grounding_path_r.exists():
                grounding_rows.extend(load_jsonl(grounding_path_r))
            if image_path_r.exists():
                image_rows.extend(load_jsonl(image_path_r))

        global_seen_keys = {(int(row["sample_id"]), str(row.get("gen_type", ""))) for row in image_rows}
        missing_keys = target_keys - global_seen_keys
        if missing_keys:
            raise RuntimeError(
                "Generation loop ended before collecting all target keys across ranks. "
                f"Missing count={len(missing_keys)}."
            )

        text_rows = dedupe_rows_by_sample_gen(text_rows)
        grounding_rows = dedupe_rows_by_sample_gen(grounding_rows)
        image_rows = dedupe_rows_by_sample_gen(image_rows)
        stage1_count = len([row for row in image_rows if row.get("stage1_overlay_path")])
        stage2_count = len([row for row in image_rows if row.get("stage2_overlay_path")])
        stage3_count = len([row for row in image_rows if row.get("stage3_mask64_path")])

        save_jsonl(out_dir / "text_gen.jsonl", text_rows)
        save_jsonl(out_dir / "grounding.jsonl", grounding_rows)
        save_jsonl(out_dir / "image_gen.jsonl", image_rows)

        if len(image_rows) != total_samples:
            raise RuntimeError(
                f"image_gen rows mismatch: expected {total_samples}, got {len(image_rows)}."
            )
        if stage1_count != total_samples:
            raise RuntimeError(
                f"image_gen stage1 image count mismatch: expected {total_samples}, got {stage1_count}."
            )
        if stage2_count != total_samples:
            raise RuntimeError(
                f"image_gen stage2 image count mismatch: expected {total_samples}, got {stage2_count}."
            )
        if stage3_count != total_samples:
            raise RuntimeError(
                f"image_gen stage3 image count mismatch: expected {total_samples}, got {stage3_count}."
            )

        output_files: Dict[str, Any] = {
            "text_gen": str(out_dir / "text_gen.jsonl"),
            "grounding": str(out_dir / "grounding.jsonl"),
            "grounding_overlays_dir": str(grounding_overlay_dir),
            "image_gen": str(out_dir / "image_gen.jsonl"),
            "image_gen_stage1_bbox_dir": str(image_stage1_dir),
            "image_gen_stage2_edited_bbox_dir": str(image_stage2_dir),
            "image_gen_stage3_mask64_dir": str(image_stage3_dir),
            "image_gen_stage2_raw_dir": str(image_stage2_raw_dir) if args.save_raw_edited else None,
        }

        run_config = {
            "args": vars(args),
            "distributed": {
                "world_size": world_size,
                "rank": rank,
                "local_rank": local_rank,
            },
            "num_loaded_text": 0,
            "num_loaded_grounding": 0,
            "num_loaded_image_gen": total_samples,
            "num_aligned": 0,
            "num_sampled": total_samples,
            "num_concatenated_rows": len(concatenated_rows),
            "sampler_chunk_size": sampler_chunk_size,
            "effective_sampler_chunk_size": effective_sampler_chunk_size,
            "output_files": output_files,
        }
        with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

        print(f"Saved text logs: {out_dir / 'text_gen.jsonl'}")
        print(f"Saved grounding logs: {out_dir / 'grounding.jsonl'}")
        print(f"Saved grounding overlays: {grounding_overlay_dir}")
        print(f"Saved image_gen logs: {out_dir / 'image_gen.jsonl'}")
        print(f"Saved image_gen stage1 overlays: {image_stage1_dir}")
        print(f"Saved image_gen stage2 overlays: {image_stage2_dir}")
        print(f"Saved image_gen stage3 mask overlays: {image_stage3_dir}")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
