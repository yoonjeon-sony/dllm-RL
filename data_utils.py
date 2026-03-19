
from datasets import load_dataset, Dataset
import pandas as pd
from reward_func import extract_hash_answer
import io
import json
import random
import numpy as np
import torch
import os
from PIL import Image

def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return data  # type: ignore


def get_code_questions(split="train"):
    data = load_dataset("KodCode/KodCode-Light-RL-10K", split=split)
    data = data.train_test_split(test_size=0.1, seed=42)[
        "train"
    ]  # NOTE: 10% of the data was used for a different experiment
    data = data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a coding expert. You will be given a coding problem to solve. Solve it step by step. \n\n{x['question']}",
                }
            ],
            "answer": {"solution": x["solution"], "tests": x["test"]},
        }
    )
    return data

def convert_to_rgb(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


THINKMORPH_HF_REPO_ID = "yjyjyj98/mvot-rl-dataset"
THINKMORPH_HF_CONFIGS = (
    "ThinkMorph-Spatial_Navigation_loc",
    "ThinkMorph-Visual_Search_loc",
    "ThinkMorph-Chart_Refocus_loc",
)


COT_PROMPT = (
        "Let's think step-by-step to solve the question."
        "Put your final answer in <answer> <\answer> tags. "
    )
GROUNDING_PROMPT = (
    "Your job is to identify the region where auxiliary line, box, or editing could help solve the following problem. Give bounding boxes in LOC format."
)
EDIT_PROMPT = (
    "Edit the region where auxiliary line, box, or drawing could help solve the following problem."
)
# '''Generation Instructions: You should first think about the reasoning process in the mind and then provide the user with the answer. 
# The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

INTERLEAVE_PROMPT = (
    "Let's think step-by-step to solve the question."
    "If visual aid is needed, use <vision_start> <vision_end> tags for image editing."
    "Put your final answer in <answer> <\answer> tags."
)

def _build_question_prompt(question):
    return (
    "<|startoftext|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"<image>\n {COT_PROMPT} {question}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

def _build_grounding_prompt(question):
    """Build the grounding prompt for a given question."""
    return f'''<|startoftext|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n {GROUNDING_PROMPT} {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<LOC_BEGIN><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><LOC_END><|eot_id|>'''


def _parse_bbox(raw_bbox):
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, str):
        raw_bbox = raw_bbox.strip()
        if not raw_bbox:
            return None
        try:
            raw_bbox = json.loads(raw_bbox)
        except json.JSONDecodeError:
            return None
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            return [int(coord) for coord in raw_bbox]
        except (TypeError, ValueError):
            return None
    return None


def _decode_dataset_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB") if image_value.mode != "RGB" else image_value
    if isinstance(image_value, str):
        return convert_to_rgb(image_value)
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            image = Image.open(io.BytesIO(image_value["bytes"]))
            return image.convert("RGB") if image.mode != "RGB" else image
        if image_value.get("path"):
            return convert_to_rgb(image_value["path"])
    raise TypeError(f"Unsupported ThinkMorph image value type: {type(image_value)!r}")


def get_thinkmorph_image_editing_questions(
    split: str = "train",
    data_root: str | None = None,
    image_root: str | None = None,
    gen_type: str = "image_gen",
    per_task_sample = None,
) -> Dataset:
    if split != "train":
        raise ValueError(f"Unsupported split '{split}' for ThinkMorph. Use 'train'.")
    if gen_type not in {"text_gen", "grounding", "image_gen"}:
        raise ValueError(f"Unsupported gen_type '{gen_type}'.")

    rows: list[dict] = []
    for config_name in THINKMORPH_HF_CONFIGS:
        data = load_dataset(THINKMORPH_HF_REPO_ID, config_name, split=split)
        for idx, example in enumerate(data):
            gt_bbox = _parse_bbox(example.get("raw_gt_box"))
            if gen_type in {"grounding", "image_gen"} and gt_bbox is None:
                continue

            question = example["question"]
            rows.append(
                {
                    "answer_prompt": _build_question_prompt(question),
                    "edit_prompt": f"{EDIT_PROMPT} {question}",
                    "grounding_prompt": _build_grounding_prompt(question),
                    "answer_gt": example["answer"],
                    "grounding_gt": gt_bbox,  # (x, y, x, y) in original image scale / left top & right bottom coords
                    "image": _decode_dataset_image(example["problem_image_0"]),
                    "image_gt": _decode_dataset_image(example["reasoning_image_0"]),
                    "gen_type": gen_type,
                    "task_type": config_name,
                }
            )
            if per_task_sample is not None and idx >= per_task_sample:
                break

    return Dataset.from_list(rows)
