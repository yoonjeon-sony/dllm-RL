import numpy as np
import re

import multiprocessing
import resource
import os

import time
import random
import string
import shutil


import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
import random
import re


# Reward functions
def boxed_in_answer(prompts, completions, step=None, **kwargs):
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    rewards = []
    for r in responses:
        reward = 0.0
        try:
            r = r.split("<answer>")[1].split("</answer>")[0]
            reward += 1.0
        except:
            reward += 0.0

        reward += 1.0 if "\boxed" in r else 0.5
        rewards.append(reward)
    return rewards


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:
        return s


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def create_few_shot_prompt_math(dataset, num_examples=4):
    """Create few-shot prompt from dataset examples"""
    random.seed(42)
    few_shot_examples = random.sample(range(len(dataset)), num_examples)

    formatted_examples = []
    for example in few_shot_examples:
        input_text = dataset[example]["problem"]
        answer = dataset[example]["solution"]
        formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")

    # prompt = "You are given examples of math questions and answer, and in the end you will be given a new question to solve. Solve it step by step. Wrap the answer in a \\boxed\{\}. \n\n"
    prompt = "You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. \n\n"
    return prompt + "\n\n".join(formatted_examples)


def extract_answer_first_math(generated_text):
    """Extract the first numerical answer following '####' in the generated text."""
    try:
        # Remove the prompt part
        answer_part = generated_text

        # Use regex to find the first occurrence of #### followed by a number
        match = match = re.search(r"####\s*(.*?)\s*<\|EOT\|>", answer_part)

        if match:
            return match.group(1)
        return None
    except:
        return None


def decode(tokenizer, output, skip_special_tokens=False):
    """Decode a batch of output IDs to text."""
    return tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)


def create_prompts(input_texts, tokenizer, few_shot_prompt=""):
    prompts = []
    for input_text in input_texts:
        # Format similar to your chat function
        m = [
            {
                "role": "user",
                "content": f"{few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n",
            }
        ]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        prompts.append(user_input)
    return prompts
    

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"] if isinstance(completion, list) else completion) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores

def boxed_and_answer_tags_format_reward(
    prompts, completions, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, step=step)
    rewards = [b for b in boxed_in_answer_rewards]
    return rewards


def correctness_reward_func(prompts, completions, answer_gt, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    q = prompts[0][-1]["content"] if isinstance(prompts[0], list) else prompts[0]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer_gt[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer_gt)]

def correct_grounding_reward_func(prompts, completions, grounding_gt, step=None, run_name=None, **kwargs) -> list[float]:
    def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise IoU between two tensors of shape (4,) in xyxy format.
        """
        # Intersection
        inter_x1 = torch.max(boxes1[0], boxes2[0])
        inter_y1 = torch.max(boxes1[1], boxes2[1])
        inter_x2 = torch.min(boxes1[2], boxes2[2])
        inter_y2 = torch.min(boxes1[3], boxes2[3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Areas
        area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
        area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        # Union
        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou


    box = [[int(y) for y in re.compile('<LOC_([0-9]+)>').findall(x)] for x in completions]
    rewards = []
    for pred_box, gt_box in zip(box, grounding_gt):
        if len(pred_box) == 4:
            rewards.append(_pairwise_iou(
                torch.tensor(pred_box, dtype=torch.float32),
                torch.tensor(gt_box,   dtype=torch.float32),
            ).item())
        else:
            rewards.append(0.0)
            
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{prompts[0]}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{grounding_gt[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{box[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{rewards[0]}\n",
    )
    return rewards


def perceptual_score_reward_func(prompts, image_completions, image_gt, step=None, run_name=None, **kwargs) -> list[float]:
    """
    Computes perceptual similarity score between generated image and ground truth image.
    Higher is better.
    """
    import lpips
    loss_fn = lpips.LPIPS(net='vgg')  # perceptual similarity

    def _normalize(img: torch.Tensor):
        """
        Normalize image to [-1, 1] for LPIPS or flatten for cosine.
        """
        if img.max() > 1:
            img = img / 255.0
        return img

    def _lpips_score(img1, img2):
        img1 = _normalize(img1).unsqueeze(0)
        img2 = _normalize(img2).unsqueeze(0)
        return 1.0 - loss_fn(img1, img2).item()  # higher is better

    rewards = []
    for pred_img, gt_img in zip(image_completions, image_gt):
        score = _lpips_score(pred_img, gt_img)
        rewards.append(score)

    # ANSI colors
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{prompts[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Perceptual Score:{RESET}\n{rewards[0]}\n",
    )

    return rewards