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
from diffu_grpo_trainer import DiffuGRPOTrainer # for LaViDa-O
from mmada_grpo_trainer import MMADAGRPOTrainer # for MMADA
from diffu_grpo_config import DiffuGRPOConfig
from transformers import AutoTokenizer
from llava.model.modeling_mmada import MMadaModelLM
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
from input_processor import LavidaOProcessor, MMADAProcessor


def model_init_fn(path, add_vision_tokens=False):
    if "lavida" in path.lower():
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
    elif "mmada" in path.lower(): # /group2/dgm/yoonjeon/MMaDA-8B-Base
        from llava.model.multimodal_encoder.magvit2.magvit2 import MAGVITv2
        tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        vq_model = MAGVITv2
        vq_model = vq_model.from_pretrained("showlab/magvitv2")
        vq_model.requires_grad_(False)
        vq_model.eval()
        img_proc = vq_model
        # mmada config
        # pretrained_model_path: "Gen-Verse/MMaDA-8B-MixCoT"
        # w_clip_vit: False
        # new_vocab_size: 134656
        # llm_vocab_size: 126464
        # codebook_size: 8192
        # num_vq_tokens: 1024
        # num_new_special_tokens: 0
        # tie_word_embeddings: False
        # gradient_checkpointing: True
        mdl = MMadaModelLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        ctx_len = 4096
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
        gen_type = "text_gen"
        dataset = get_thinkmorph_image_editing_questions(
            data_root=grpo_config.data_root,
            image_root=grpo_config.image_root,
            gen_type=gen_type,
        )
        reward_functions = [
            correctness_reward_func,
            boxed_and_answer_tags_format_reward,
        ]

    elif dataset_name == "thinkmorph_edit":
        gen_type = "image_gen"
        dataset = get_thinkmorph_image_editing_questions(
            data_root=grpo_config.data_root,
            image_root=grpo_config.image_root,
            gen_type=gen_type,
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

    if "lavida" in grpo_config.model_path.lower():
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            processing_class=LavidaOProcessor(model, tokenizer, image_processor),
            model_init_fn=model_init_fn,
            gen_type=gen_type,
        )
    elif "mmada" in grpo_config.model_path.lower():
        trainer = MMADAGRPOTrainer(
            args=grpo_config,
            model=model,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            processing_class=MMADAProcessor(model, tokenizer, image_processor, max_seq_len=grpo_config.max_completion_length),
            model_init_fn=model_init_fn,
            gen_type=gen_type,
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
