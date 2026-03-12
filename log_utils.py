import math
import random
from transformers.utils import is_rich_available
from transformers.trainer import is_conversational
import torch.nn as nn


def _align_len(values, target_len, pad_value=None):
    values = list(values)
    if len(values) < target_len:
        values.extend([pad_value] * (target_len - len(values)))
    return values[:target_len]


def _get_reward_func_name(reward_func):
    if isinstance(
        reward_func, nn.Module
    ):  # Module instead of PretrainedModel for compat with compiled models
        return reward_func.config._name_or_path.split("/")[-1]
    return reward_func.__name__

def _build_reward_completions(prompts_for_reward, completion_texts):
    if is_conversational(inputs[0]):
        completions_for_reward = []
        for prompt, completion in zip(prompts_for_reward, completion_texts):
            completion = "" if completion is None else completion
            bootstrap = (
                prompt[-1]["content"]
                if prompt and isinstance(prompt, list) and prompt[-1]["role"] == "assistant"
                else ""
            )
            completions_for_reward.append(
                [{"role": "assistant", "content": bootstrap + completion}]
            )
        return completions_for_reward
    return completion_texts

def _format_image_gen_prompt_log(grounding_prompt, answer_prompt):
    grounding_prompt = "" if grounding_prompt is None else grounding_prompt
    answer_prompt = "" if answer_prompt is None else answer_prompt
    return (
        "[Grounding input]\n"
        f"{grounding_prompt}\n\n"
        "[Text input]\n"
        f"{answer_prompt}"
    )

def _format_image_gen_completion_log(grounding_completion, answer_completion):
    grounding_completion = "" if grounding_completion is None else grounding_completion
    answer_completion = "" if answer_completion is None else answer_completion
    return (
        "[Grounding output]\n"
        f"{grounding_completion}\n\n"
        "[Text output]\n"
        f"{answer_completion}"
    )

def _sample_log_indices(num_rows, sample_ratio, step_seed):
    if num_rows <= 0:
        return []
    sample_ratio = max(0.0, min(1.0, float(sample_ratio)))
    if sample_ratio <= 0.0:
        return []
    sample_size = max(1, int(math.ceil(num_rows * sample_ratio)))
    sample_size = min(num_rows, sample_size)
    rng = random.Random(step_seed)
    return sorted(rng.sample(range(num_rows), sample_size))

def _select_by_indices(values, indices):
    return [values[idx] for idx in indices]

def _log_prompt_completion_samples_rich(
    prompts_to_print,
    completions_to_print,
    rewards_to_print,
    advantages_to_print,
    step,
):
    if not prompts_to_print or not is_rich_available():
        return
    from rich.rule import Rule
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console

    console = Console()

    console.print(Rule(f"[bold magenta]GRPO Samples @ step {step}[/bold magenta]"))

    for row_idx, (prompt, completion, advantage) in enumerate(
        zip(prompts_to_print, completions_to_print, advantages_to_print), start=1
    ):
        prompt_text = "" if prompt is None else str(prompt)
        completion_text = "" if completion is None else str(completion)

        # Prompt block
        console.print(
            Panel(
                prompt_text,
                title=f"[bold cyan]Prompt • Sample {row_idx}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Completion block
        console.print(
            Panel(
                completion_text,
                title="[bold green]Completion[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Rewards block
        reward_lines = []
        for reward_name, reward_values in rewards_to_print.items():
            reward_value = (
                reward_values[row_idx - 1] if row_idx - 1 < len(reward_values) else None
            )
            reward_lines.append(f"[yellow]{reward_name}[/yellow]: {reward_value}")

        reward_lines.append(f"[bold bright_green]advantage[/bold bright_green]: {advantage}")

        console.print(
            Panel(
                "\n".join(reward_lines),
                title="[bold yellow]Rewards[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        if row_idx != len(prompts_to_print):
            console.print(Rule(style="dim"))