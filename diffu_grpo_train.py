import os
# Must be set before importing llava modules
os.environ['DEBUG_FIX_PADDING'] = '1'
os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'

import copy
import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, ProcessorMixin
from trl import TrlParser, ModelConfig
from peft import LoraConfig
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
from llava.model.language_model.llada.generate import generate
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import pad_to_square_and_resize, process_images, tokenizer_image_token
from llava.model.utils import pad_along_last_dim

class MyProcessor(ProcessorMixin):
    attributes = []
    optional_attributes = []

    def __init__(self, model, tokenizer, image_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def __call__(
        self,
        texts=None,
        images=None,
        grounding_texts=None,
        edit_texts=None,
        edit_mode=0,
        padding=True,
        return_tensors="pt",
        device="cuda",
        dtype=torch.bfloat16,
        mask_id=126336,
        mode="text_gen",
        image_resolution=1024,
        do_cfg=False,
        **kwargs,
    ):
        from trl.trainer.utils import pad

        reserve_id = 126089
        reserve_id2 = 126090
        reserve_token = "<|reserved_token_5|>"
        reserve_token_2 = "<|reserved_token_6|>"

        texts = texts or []
        grounding_texts = grounding_texts or []
        edit_texts = edit_texts or []

        def _left_pad_2d(tensors, pad_value, dtype_):
            if len(tensors) == 0:
                return torch.empty(0, 0, dtype=dtype_, device=device)
            max_len = max(t.shape[1] for t in tensors)
            out = []
            for t in tensors:
                pad_len = max_len - t.shape[1]
                if pad_len > 0:
                    pad_t = torch.full((t.shape[0], pad_len), pad_value, dtype=dtype_, device=t.device)
                    t = torch.cat([pad_t, t], dim=1)
                out.append(t)
            return torch.cat(out, dim=0)

        def _left_pad_3d(tensors):
            if len(tensors) == 0:
                return torch.empty(0, 0, 0, dtype=dtype, device=device)
            max_len = max(t.shape[1] for t in tensors)
            out = []
            for t in tensors:
                pad_len = max_len - t.shape[1]
                if pad_len > 0:
                    pad_t = torch.zeros((t.shape[0], pad_len, t.shape[2]), dtype=t.dtype, device=t.device)
                    t = torch.cat([pad_t, t], dim=1)
                out.append(t)
            return torch.cat(out, dim=0)

        def _normalize_images(image_groups, expected_len):
            if image_groups is None:
                return None
            norm = []
            for idx in range(expected_len):
                sample_images = image_groups[idx] if idx < len(image_groups) else None
                if isinstance(sample_images, (list, tuple)):
                    sample_group = [img for img in sample_images if img is not None]
                else:
                    sample_group = [sample_images] if sample_images is not None else []
                norm.append(sample_group)
            return norm

        def _extract_primary_images(image_groups):
            primary_images = []
            for idx, sample_images in enumerate(image_groups):
                if len(sample_images) == 0:
                    raise ValueError(f"Missing image for sample index {idx}.")
                primary_images.append(sample_images[0])
            return primary_images

        def _prepare_text_only(text_batch):
            encoded = self.tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            inputs_embeds = self.model.get_model().embed_tokens(input_ids)
            return {
                "input_ids": input_ids,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }

        def _prepare_text_with_images(text_batch, image_groups, include_bbox_mask=False):
            if image_groups is None:
                raise ValueError("`images` are required for multimodal processing.")
            if len(text_batch) != len(image_groups):
                raise ValueError("texts and images must have the same batch size.")

            flat_images = []
            image_sizes = []
            
            for idx, (txt, sample_images) in enumerate(zip(text_batch, image_groups)):
                if len(sample_images) == 0:
                    raise ValueError(f"Found missing image in batch at sample index {idx}.")
                num_image_tokens = txt.count(DEFAULT_IMAGE_TOKEN)
                if num_image_tokens < len(sample_images):
                    missing_tokens = len(sample_images) - num_image_tokens
                    image_prefix = "\n".join([DEFAULT_IMAGE_TOKEN] * missing_tokens)
                    txt = f"{image_prefix}\n{txt}"
                text_batch[idx] = txt

                flat_images.extend(sample_images)
                image_sizes.extend([image.size for image in sample_images])

            image_tensor = process_images(flat_images, self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]
            else:
                image_tensor = [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]

            all_input_ids = [
                tokenizer_image_token(txt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
                for txt in text_batch
            ]
            input_ids = pad(
                [torch.tensor(ids, dtype=torch.long) for ids in all_input_ids],
                padding_value=self.tokenizer.pad_token_id,
                padding_side="left",
            ).to(device)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

            (_, _, attention_mask, _, inputs_embeds, _, raw_input_ids) = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None, image_tensor, ["image"] * len(text_batch), image_sizes=image_sizes, return_inputs=True
            )
            out = {
                "input_ids": input_ids,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            if include_bbox_mask:
                out["bbox_mask"] = raw_input_ids == mask_id
            return out

        def _prepare_edit_generation_embeddings(edit_text_batch, image_batch, do_cfg=False):
            all_input_ids = []
            all_attention_mask = []
            all_inputs_embeds = []
            all_inputs_embeds_cond = []
            all_inputs_embeds_uncond = []
            all_inputs_embeds_uncond_enc = []
            all_is_gen = []
            all_is_gen_enc = []
            all_is_gen_enc_ccc = []
            all_is_prompt = []

            for idx, text in enumerate(edit_text_batch):
                sample_image = image_batch[idx]
                if sample_image is None:
                    raise ValueError(f"Missing image for sample index {idx} in image_gen mode.")

                image_tensor = process_images([sample_image], self.image_processor, self.model.config)
                if isinstance(image_tensor, list):
                    image_tensor = [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]
                else:
                    image_tensor = [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]

                image_for_enc = pad_to_square_and_resize(sample_image.convert("RGB"), image_resolution)
                vq_latents = self.model.model.image_processor_gen.preprocess(image_for_enc).to(device, dtype=dtype)
                init_latents, gen_shape = self.model.encode_image_gen(vq_latents)
                gen_embeddings = self.model.model.call_gen_embedding(init_latents, gen_shape)
                enc_latents, _gen_shape = self.model.encode_image_gen(vq_latents, enc=True)
                enc_embeddings = self.model.model.call_gen_embedding(enc_latents, _gen_shape, enc=True)

                # Build edit prompt with exact token spans for encoder and generation branches.
                n_tokens_txt = gen_embeddings.shape[1]
                conv = copy.deepcopy(conv_templates["llada"])
                conv.append_message(conv.roles[0], f"<image> {reserve_token_2 * enc_embeddings.shape[1]}\n {text} ")
                conv.append_message(conv.roles[1], f"{reserve_token * n_tokens_txt}")
                prompt_question = conv.get_prompt()

                ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                (_, _, attention_mask, _, inputs_embeds, _, raw_input_ids) = self.model.prepare_inputs_labels_for_multimodal(
                    input_ids=ids,
                    position_ids=None,
                    attention_mask=None,
                    past_key_values=None,
                    labels=None,
                    images=image_tensor,
                    modalities=["image"],
                    image_sizes=[sample_image.size],
                    return_inputs=True,
                )
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )

                inputs_embeds = inputs_embeds.to(enc_embeddings.dtype)
                inputs_embeds[raw_input_ids == reserve_id2] = 0
                enc_pad = pad_along_last_dim(enc_embeddings, inputs_embeds.shape[-1])
                inputs_embeds[raw_input_ids == reserve_id2] = enc_pad.flatten(0, 1)

                eot_pos = torch.where(raw_input_ids[0] == 126348)[0]
                if len(eot_pos) >= 2:
                    prompt_cutoff = eot_pos[1].item()
                elif len(eot_pos) == 1:
                    prompt_cutoff = eot_pos[0].item()
                else:
                    gen_pos = torch.where(raw_input_ids[0] == reserve_id)[0]
                    prompt_cutoff = max(0, gen_pos[0].item() - 1) if len(gen_pos) > 0 else raw_input_ids.shape[1] - 1

                is_prompt = torch.zeros_like(raw_input_ids, dtype=torch.bool)
                is_prompt[:, :prompt_cutoff + 1] = True
                is_gen = raw_input_ids == reserve_id
                is_gen_enc = raw_input_ids == reserve_id2

                # Prepare CFG embeddings (mirrors region_edit.py logic)
                inputs_embeds_cond = inputs_embeds
                noise_embed = self.model.get_model().transformer.wte(
                    torch.tensor([mask_id], device=inputs_embeds.device)
                ).to(inputs_embeds.dtype)
                inputs_embeds_uncond = inputs_embeds.clone()
                inputs_embeds_uncond[is_prompt] = noise_embed
                inputs_embeds_uncond_enc = inputs_embeds.clone()
                if edit_mode == 0:
                    inputs_embeds_uncond_enc[~is_gen_enc] = noise_embed
                    is_gen_enc_ccc = is_gen_enc
                elif edit_mode == 1:
                    inputs_embeds_uncond_enc[is_gen_enc] = noise_embed
                    is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
                elif edit_mode == 2:
                    inputs_embeds_uncond_enc[is_gen_enc | (raw_input_ids < 0)] = noise_embed
                    is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
                elif edit_mode == 3:
                    inputs_embeds_uncond_enc[(~is_gen_enc) & (raw_input_ids > 0)] = noise_embed
                    is_gen_enc_ccc = is_gen_enc
                else:
                    raise ValueError(f"Not Supported edit_mode: {edit_mode}")

                all_input_ids.append(ids)
                all_attention_mask.append(attention_mask.to(device))
                all_inputs_embeds.append(inputs_embeds)
                all_inputs_embeds_cond.append(inputs_embeds_cond)
                all_inputs_embeds_uncond.append(inputs_embeds_uncond)
                all_inputs_embeds_uncond_enc.append(inputs_embeds_uncond_enc)
                all_is_gen.append(is_gen)
                all_is_gen_enc.append(is_gen_enc)
                all_is_gen_enc_ccc.append(is_gen_enc_ccc)
                all_is_prompt.append(is_prompt)

            input_ids = _left_pad_2d(all_input_ids, self.tokenizer.pad_token_id, torch.long)
            attention_mask = _left_pad_2d(all_attention_mask, 0, torch.long)
            inputs_embeds = _left_pad_3d(all_inputs_embeds)
            inputs_embeds_cond = _left_pad_3d(all_inputs_embeds_cond)
            inputs_embeds_uncond = _left_pad_3d(all_inputs_embeds_uncond)
            inputs_embeds_uncond_enc = _left_pad_3d(all_inputs_embeds_uncond_enc)
            is_gen = _left_pad_2d(all_is_gen, 0, torch.bool)
            is_gen_enc = _left_pad_2d(all_is_gen_enc, 0, torch.bool)
            is_gen_enc_ccc = _left_pad_2d(all_is_gen_enc_ccc, 0, torch.bool)
            is_prompt = _left_pad_2d(all_is_prompt, 0, torch.bool)
            is_gen_enc_null = torch.zeros_like(is_gen_enc, dtype=torch.bool)

            return {
                "is_gen": is_gen,
                "is_gen_enc": is_gen_enc,
                "is_gen_enc_ccc": is_gen_enc_ccc,
                "is_gen_enc_null": is_gen_enc_null,
                "is_prompt": is_prompt,
                "input_embeds_gen": inputs_embeds,
                "inputs_embeds_cond": inputs_embeds_cond,
                "inputs_embeds_uncond": inputs_embeds_uncond,
                "inputs_embeds_uncond_enc": inputs_embeds_uncond_enc,
                "attention_mask_gen": attention_mask,
                "input_ids_gen": input_ids,
            }

        if mode == "text_gen":
            norm_images = _normalize_images(images, len(texts)) if images is not None else None
            if norm_images is None:
                return _prepare_text_only(texts)
            return _prepare_text_with_images(texts, norm_images, include_bbox_mask=False)

        if mode == "grounding":
            norm_images = _normalize_images(images, len(grounding_texts if grounding_texts else texts))
            grounding_batch = grounding_texts if grounding_texts else texts
            return _prepare_text_with_images(grounding_batch, norm_images, include_bbox_mask=True)

        if mode == "image_gen":
            if images is None:
                raise ValueError("`images` is required for mode='image_gen'")
            if len(texts) == 0 or len(grounding_texts) == 0 or len(edit_texts) == 0:
                raise ValueError("image_gen mode requires texts, grounding_texts, and edit_texts.")
            if not (len(texts) == len(grounding_texts) == len(edit_texts)):
                raise ValueError("texts, grounding_texts, and edit_texts must have the same batch size.")
            norm_images = _normalize_images(images, len(texts))
            primary_image_groups = [[img] for img in _extract_primary_images(norm_images)]
            primary_images = [sample_images[0] for sample_images in primary_image_groups]
            grounding_batch = _prepare_text_with_images(grounding_texts, primary_image_groups, include_bbox_mask=True)
            prompt_batch = _prepare_text_with_images(texts, primary_image_groups, include_bbox_mask=False)
            edit_batch = _prepare_edit_generation_embeddings(edit_texts, primary_images, do_cfg=do_cfg)

            return {
                "input_ids": grounding_batch["input_ids"],
                "input_embeds": grounding_batch["input_embeds"],
                "attention_mask": grounding_batch["attention_mask"],
                "bbox_mask": grounding_batch["bbox_mask"],
                "input_ids_prompt": prompt_batch["input_ids"],
                "input_embeds_prompt": prompt_batch["input_embeds"],
                "attention_mask_prompt": prompt_batch["attention_mask"],
                **edit_batch,
            }

        raise ValueError(f"Unsupported mode: {mode}")


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

    # Single callable for model loading — used here for the initial load and passed to
    # DiffuGRPOTrainer so _load_from_checkpoint uses the identical loading path.
    def model_init_fn(path):
        tok, mdl, img_proc, ctx_len = load_pretrained_model(
            path,
            None,
            "llava_llada",
            attn_implementation="sdpa",
            device_map="cpu",
            torch_dtype="bfloat16",
        )
        mdl.tie_weights()
        mdl.to(torch.bfloat16)
        return tok, mdl, img_proc, ctx_len

    tokenizer, model, image_processor, _ = model_init_fn(grpo_config.model_path)
    model.to(device)

    # tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.use_cache = False

    
    # Configure LoRA for parameter-efficient fine-tuning
    # peft_config = LoraConfig(
    #     r=model_config.lora_r,
    #     lora_alpha=model_config.lora_alpha,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    #     task_type="CAUSAL_LM",
    #     lora_dropout=model_config.lora_dropout,
    # )
    # Initialize and run trainer
    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        # peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        processing_class=MyProcessor(model, tokenizer, image_processor),
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
    trainer.train(resume)


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
