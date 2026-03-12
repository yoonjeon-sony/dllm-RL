import os
# Must be set before importing llava modules
os.environ['DEBUG_FIX_PADDING'] = '1'
os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'

import copy
import torch
import wandb
from trl import TrlParser, ModelConfig
import warnings

# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
    coding_reward_func,
    correct_grounding_reward_func,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
    get_code_questions,
    get_thinkmorph_image_editing_questions,
)

from llava.model.builder import load_pretrained_model
from input_processor import MyProcessor


def model_init_fn(path, add_vision_tokens=False):
    vision_kwargs = dict(
        mm_vision_tower="google/siglip-so400m-patch14-384",
        mm_resampler_type=None,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    tok, mdl, img_proc, ctx_len = load_pretrained_model(
        path,
        None,
        "llava_llada",
        attn_implementation="sdpa",
        device_map="cpu",
        torch_dtype="bfloat16",
    )
    if add_vision_tokens:
        num_new_tokens = tok.add_special_tokens(
            {"additional_special_tokens": ["<vision_start>", "<vision_end>"]}
        )
        mdl.config.get_text_config(decoder=True).tie_word_embeddings = False
        mdl.resize_token_embeddings(len(tok))
    
    mdl.tie_weights()
    mdl.to(torch.bfloat16)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl.config.use_cache = False
    return tok, mdl, img_proc, ctx_len


def main(grpo_config, model_config):
    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)
    dataset_name = (grpo_config.dataset or "").replace("-", "_").rstrip(":")

    # Load dataset based on configuration
    if dataset_name == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif dataset_name == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif dataset_name == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif dataset_name == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    elif dataset_name == "code":
        dataset = get_code_questions()
        reward_functions = [xmlcount_reward_func, coding_reward_func]
    
    elif dataset_name == "thinkmorph":
        dataset = get_thinkmorph_image_editing_questions(
            data_root=grpo_config.data_root,
            image_root=grpo_config.image_root,
            gen_type="text_gen",
        )
        reward_functions = [
            correctness_reward_func,
            boxed_and_answer_tags_format_reward,
        ]

    elif dataset_name == "thinkmorph_grounding":
        dataset = get_thinkmorph_image_editing_questions(
            data_root=grpo_config.data_root,
            image_root=grpo_config.image_root,
            gen_type="grounding",
        )
        reward_functions = [
            boxed_and_answer_tags_format_reward,
            correctness_reward_func,
            correct_grounding_reward_func,
        ]
    elif dataset_name == "thinkmorph_edit":
        dataset = get_thinkmorph_image_editing_questions(
            data_root=grpo_config.data_root,
            image_root=grpo_config.image_root,
            gen_type="image_gen",
        )
        reward_functions = [
            boxed_and_answer_tags_format_reward,
            correct_grounding_reward_func,
            correctness_reward_func,
        ]
    else:
        raise ValueError(f"Unsupported dataset '{grpo_config.dataset}'.")

    grpo_config.dataset = dataset_name

        
    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if dataset_name in ["countdown", "sudoku", "thinkmorph"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device (respect LOCAL_RANK for multi-GPU runs)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    
    tokenizer, model, image_processor, _ = model_init_fn(grpo_config.model_path)
    model.to(device)

    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        processing_class=MyProcessor(model, tokenizer, image_processor),
        model_init_fn=model_init_fn,
    )

    if grpo_config.save_steps % grpo_config.num_iterations != 0:
        warnings.warn(
            f"save_steps ({grpo_config.save_steps}) is not divisible by num_iterations ({grpo_config.num_iterations}). If resuming training from a checkpoint, you might need to manually specify the checkpoint where the training step is divisible by {grpo_config.num_iterations}."
        )

    resume = grpo_config.resume_from_checkpoint
    if isinstance(resume, str):
        if resume.lower() == "true":
            resume = True
        elif resume.lower() == "false":
            resume = None
    elif resume is False:
        resume = None
    trainer.train(resume_from_checkpoint=resume)

if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
