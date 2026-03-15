import collections
import copy
import io
import math
import random
import re
import time
import torch
import torch.distributed as dist
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized, Sequence, Dict
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from torch import nn
from transformers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        set_seed,
        gather,
        gather_object,
        DataLoaderConfiguration,
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        release_memory,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad
import wandb
import os
import logging
import PIL
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from transformers.trainer import TRAINING_ARGS_NAME
from llava.mm_utils import process_images, tokenizer_image_token, pad_to_square_and_resize
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens, cosine_schedule_2, logit_normal_schedule, exp_schedule, wte, get_logits, get_num_transfer_tokens_sch
from llava.model.utils import maybe_truncate_last_dim
from log_utils import (
    _align_len,
    _build_reward_completions,
    _format_image_gen_completion_log,
    _format_image_gen_prompt_log,
    _get_reward_func_name,
    _log_prompt_completion_samples_rich,
    _sample_log_indices,
    _select_by_indices,
)
from interleaved_inferencer import InterleavedInferencer
from reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correct_grounding_reward_func,
)
logger = logging.getLogger(__name__)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class DiffuGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        model_init_fn = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        # Custom attributes not set by GRPOTrainer base class.
        self.max_prompt_length = args.max_prompt_length if args is not None else None
        # Override parent's None; use a dict so item assignment never fails due to list pre-allocation.
        self._buffered_inputs = {}
        self.model_init_fn = model_init_fn
        self.inferencer = InterleavedInferencer(self.model)
        self._current_train_step_time = 0.0

    def _slice_batch(self, batch: dict[str, Union[torch.Tensor, Any]], start: int, end: int):
        batch_size = batch["prompt_ids"].size(0)
        sliced = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() >= 1 and value.size(0) == batch_size:
                sliced[key] = value[start:end]
            elif torch.is_tensor(value) and value.dim() >= 2 and value.size(1) == batch_size:
                sliced[key] = value[:, start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _split_generation_batch(self, batch: dict[str, Union[torch.Tensor, Any]]):
        steps_per_generation = int(getattr(self.args, "steps_per_generation", 1) or 1)
        batch_size = batch["prompt_ids"].size(0)

        if steps_per_generation <= 1:
            return [batch]
        if batch_size % steps_per_generation != 0:
            raise ValueError(
                f"Generated batch size ({batch_size}) must be divisible by steps_per_generation "
                f"({steps_per_generation})."
            )

        chunk_size = batch_size // steps_per_generation
        return [
            self._slice_batch(batch, start, start + chunk_size)
            for start in range(0, batch_size, chunk_size)
        ]

    def _save_checkpoint(self, model, trial):
        original_pc = self.processing_class
        self.processing_class = self.processing_class.tokenizer
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self.processing_class = original_pc

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model + tokenizer in HF format so load_pretrained_model can reload from checkpoint dir."""
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not self.args.should_save:
            return

        # Save model weights + config via PreTrainedModel.save_pretrained
        self.model.save_pretrained(
            output_dir,
            safe_serialization=self.args.save_safetensors,
        )

        # Save tokenizer — handle both MyProcessor (called from save_model directly) and raw tokenizer (called from within _save_checkpoint, which already swapped it).
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        tokenizer.save_pretrained(output_dir)

        # Save training args (matches Trainer._save behaviour)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        """Load checkpoint using load_pretrained_model for proper LLaVA-LLaDA initialisation."""
        if model is None:
            model = self.model

        logger.info(f"Loading model checkpoint from {resume_from_checkpoint}")
        _, loaded_model, _, _ = self.model_init_fn(
            resume_from_checkpoint,
            self.config.add_vision_tokens
        )
        loaded_model.to(torch.bfloat16)

        # Transfer weights into the existing model instance so distributed-training
        # wrappers and device placement set up by the Trainer are preserved.
        load_result = model.load_state_dict(loaded_model.state_dict(), strict=False)

        if load_result.missing_keys:
            logger.warning(f"Checkpoint load — missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.warning(f"Checkpoint load — unexpected keys: {load_result.unexpected_keys}")

        # Keep inferencer bound to the trainer model after checkpoint restore.
        self.inferencer.model = self.model

        del loaded_model
        torch.cuda.empty_cache()
        logger.info(f"Successfully loaded checkpoint from {resume_from_checkpoint}")

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
       
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            clear_device_cache()

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

        output = loss.detach()

        self._step += 1
        time_after = time.perf_counter()
        if self._step % self.args.gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        
        return output

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]
        # Get the current iteration index and corresponding mask seed
        this_itr_idx = (self._step - 1) % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens
        unwrapped_model = self.accelerator.unwrap_model(model).get_model()
        image_completion_ids = inputs.get("image_completion_ids")
        per_token_logps = self._get_current_per_token_logps(
            unwrapped_model,
            input_ids,
            logits_to_keep,
            this_itr_mask_seed,
            image_completion_ids=image_completion_ids,
            image_input_embeds_gen=inputs.get("image_input_embeds_gen"),
            image_is_gen=inputs.get("image_is_gen"),
            image_is_gen_enc=inputs.get("image_is_gen_enc"),
            image_gen_shape=inputs.get("image_gen_shape"),
        )
        # Compute the KL divergence between the model and the reference model
        if self.args.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        # Log the metrics (skipped for non-final PPO mini-batch chunks)
        if not getattr(self, "_skip_loss_metrics", False):
            mode = "eval" if self.control.should_evaluate else "train"

            if self.args.beta != 0.0:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["clip_ratio"].append(
                self.accelerator.gather_for_metrics(clip_ratio).mean().item()
            )

        return loss

    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        if seed is not None and isinstance(seed, torch.Tensor):
            seed = seed.item()
        set_seed(seed)

        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt
        random_matrix = torch.rand((b, l), device=batch.device)
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index
        is_mask = is_mask_prompt | is_mask_completion
        noisy_batch = torch.where(is_mask, mask_id, batch)
        p_mask = torch.where(prompt_index, t_p.unsqueeze(1), torch.ones_like(t_p).unsqueeze(1))
        return noisy_batch, p_mask

    def _get_text_completion_mask(self, completion_ids):
        if completion_ids is None:
            return None
        if completion_ids.size(1) == 0:
            return torch.zeros((completion_ids.size(0), 0), dtype=torch.int, device=completion_ids.device)
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    def _get_image_completion_mask(self, image_completion_ids, edit_region_mask):
        if image_completion_ids is None or edit_region_mask is None:
            return None
        if image_completion_ids.dim() == 2:
            return edit_region_mask.to(dtype=torch.int, device=image_completion_ids.device)
        if image_completion_ids.dim() == 3:
            expanded_mask = edit_region_mask.unsqueeze(1).expand(-1, image_completion_ids.size(1), -1)
            return expanded_mask.reshape(image_completion_ids.size(0), -1).to(
                dtype=torch.int, device=image_completion_ids.device
            )
        raise ValueError(f"Unsupported image completion shape: {tuple(image_completion_ids.shape)}")

    def _infer_image_gen_shape(self, image_completion_ids):
        token_count = image_completion_ids.shape[-1]
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError(f"Cannot infer square image gen shape from {token_count} tokens.")
        return (side, side)

    def _concat_token_logps(self, text_logps, image_logps):
        if text_logps is None:
            return image_logps
        if image_logps is None:
            return text_logps
        return torch.cat([text_logps, image_logps], dim=-1)

    def _get_current_per_token_logps(
        self,
        model,
        input_ids,
        logits_to_keep,
        mask_seed,
        image_completion_ids=None,
        image_input_embeds_gen=None,
        image_is_gen=None,
        image_is_gen_enc=None,
        image_gen_shape=None,
    ):
        if mask_seed is not None and isinstance(mask_seed, torch.Tensor):
            mask_seed = mask_seed.item()

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True

        perturbed_seq, _ = self.forward_process(input_ids, prompt_index, self.args.mask_id, seed=mask_seed)
        text_inputs_embeds = model.transformer.wte(perturbed_seq)
        text_targets = input_ids[:, -logits_to_keep:]

        if image_completion_ids is None:
            text_output = model(
                None,
                input_embeddings=text_inputs_embeds,
                return_last_hidden_state_only=True,
                compute_logits=False,
            )
            completion_hidden_states = text_output.hidden_states[0][:, -logits_to_keep:, :]
            if model.config.weight_tying:
                completion_logits = F.linear(completion_hidden_states, model.transformer.wte.weight, None)
            else:
                completion_logits = model.transformer.ff_out(completion_hidden_states)
            if model.config.scale_logits:
                completion_logits = completion_logits * (1 / math.sqrt(model.config.d_model))
            flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
            flat_targets = text_targets.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            per_token_logps = (-loss.view(batch_size, logits_to_keep)).to(torch.float32)
            del text_output, completion_hidden_states
            return per_token_logps

        if image_gen_shape is None:
            image_gen_shape = self._infer_image_gen_shape(image_completion_ids)

        perturbed_latents = torch.full_like(image_completion_ids, 8193)
        image_inputs_embeds = image_input_embeds_gen.clone()
        image_all_input_embeddings, _ = wte(
            model,
            None,
            True,
            x_gen=perturbed_latents,
            gen_shape=image_gen_shape,
            inputs_embeds_curr=image_inputs_embeds,
            new_token_mask=image_is_gen,
        )
        enc_use_image_branch = getattr(model.config, "enc_use_image_branch", False)
        image_forward_modality_indices = image_is_gen | image_is_gen_enc if enc_use_image_branch else image_is_gen
        text_modality_indices = torch.zeros(text_inputs_embeds.shape[:2], dtype=torch.bool, device=device)
        combined_embeds = torch.cat([text_inputs_embeds, image_all_input_embeddings], dim=1)
        combined_modality_indices = torch.cat([text_modality_indices, image_forward_modality_indices], dim=1)
        combined_output = model(
            None,
            input_embeddings=combined_embeds,
            modality_indices=combined_modality_indices,
            return_last_hidden_state_only=True,
            compute_logits=False,
        )
        combined_hidden_states = combined_output.hidden_states[0]
        completion_hidden_states = combined_hidden_states[:, seq_len - logits_to_keep : seq_len, :]
        if model.config.weight_tying:
            completion_logits = F.linear(completion_hidden_states, model.transformer.wte.weight, None)
        else:
            completion_logits = model.transformer.ff_out(completion_hidden_states)
        if model.config.scale_logits:
            completion_logits = completion_logits * (1 / math.sqrt(model.config.d_model))
        flat_text_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_text_targets = text_targets.reshape(-1)
        text_loss = F.cross_entropy(flat_text_logits, flat_text_targets, reduction="none")
        text_per_token_logps = -text_loss.view(batch_size, logits_to_keep)

        image_hidden_states = combined_hidden_states[:, seq_len:, :]
        gen_hidden_states = image_hidden_states[image_is_gen]
        gen_hidden_states = maybe_truncate_last_dim(gen_hidden_states, model.config.d_model_gen)
        timesteps = torch.ones(batch_size, device=device, dtype=torch.float32)
        image_logits = model.call_gen_predictor(gen_hidden_states, image_gen_shape, timesteps=timesteps)
        seq_len_per_img = int(np.prod(image_gen_shape))
        if len(image_logits.shape) == 2:
            image_logits = image_logits.view(-1, seq_len_per_img, image_logits.shape[-1])
            flat_image_logits = image_logits.reshape(-1, image_logits.size(-1))
        else:
            image_logits = image_logits.view(-1, seq_len_per_img, *image_logits.shape[-2:])
            image_logits = image_logits.permute(0, 2, 1, 3).contiguous()
            flat_image_logits = image_logits.view(-1, image_logits.size(-1))
        flat_image_targets = image_completion_ids.reshape(-1)
        image_loss = F.cross_entropy(flat_image_logits.float(), flat_image_targets, reduction="none")
        image_per_token_logps = -image_loss.view(batch_size, -1)
        per_token_logps = torch.cat([text_per_token_logps, image_per_token_logps], dim=1).to(torch.float32)
        del combined_output, combined_hidden_states, completion_hidden_states, image_hidden_states, gen_hidden_states
        del image_logits, image_all_input_embeddings, combined_embeds
        return per_token_logps

    def _get_text_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Calculate per-token log probabilities.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length: one seed per iteration. compute_loss() calls
        # this helper with a single selected iteration, while rollout/ref-logps
        # paths may pass the full num_iterations batch.
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, _ = self.forward_process(expanded_input, prompt_index, self.args.mask_id, seed=mask_seed)
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]

        # Get model predictions for the combined batch.
        # Use get_logits (non-t2i path): builds zero modality_indices, calls model with embeddings.
        inputs_embeds_curr = model.transformer.wte(perturbed_seq)
        logits = get_logits(model, inputs_embeds_curr, t2i_inference=False)
        # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _get_image_per_token_logps(
        self,
        model,
        image_completion_ids,
        image_input_embeds_gen,
        image_is_gen,
        image_is_gen_enc,
        image_gen_shape,
        mask_seeds,
    ):
        if image_completion_ids is None:
            return None

        num_iterations = len(mask_seeds)
        batch_size = image_input_embeds_gen.size(0)
        device = image_input_embeds_gen.device
        img_mask_id = 8193
        enc_use_image_branch = getattr(model.config, "enc_use_image_branch", False)
        if image_gen_shape is None:
            image_gen_shape = self._infer_image_gen_shape(image_completion_ids)

        per_token_logps = []
        checkpoint_strategy = getattr(model, "activation_checkpointing_strategy", None)
        disable_checkpointing = torch.is_grad_enabled() and hasattr(model, "set_activation_checkpointing")
        if disable_checkpointing:
            model.set_activation_checkpointing(None)
        try:
            for mask_seed in mask_seeds:
                if mask_seed is not None and isinstance(mask_seed, torch.Tensor):
                    mask_seed = mask_seed.item()
                set_seed(mask_seed)
                perturbed_latents = torch.full_like(image_completion_ids, img_mask_id)
                inputs_embeds_curr = image_input_embeds_gen.clone()
                all_input_embeddings, new_token_mask = wte(
                    model,
                    None,
                    True,
                    x_gen=perturbed_latents,
                    gen_shape=image_gen_shape,
                    inputs_embeds_curr=inputs_embeds_curr,
                    new_token_mask=image_is_gen,
                )
                forward_modality_indices = image_is_gen | image_is_gen_enc if enc_use_image_branch else image_is_gen
                timesteps = torch.ones(batch_size, device=device, dtype=torch.float32)
                logits = get_logits(
                    model,
                    all_input_embeddings,
                    image_is_gen,
                    True,
                    gen_shape=image_gen_shape,
                    timesteps=timesteps,
                    input_modality_indices=forward_modality_indices,
                )

                if image_completion_ids.dim() == 3:
                    logits = logits.permute(0, 2, 1, 3).contiguous()
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_targets = image_completion_ids.reshape(-1)
                elif image_completion_ids.dim() == 2:
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_targets = image_completion_ids.reshape(-1)
                else:
                    raise ValueError(f"Unsupported image completion shape: {tuple(image_completion_ids.shape)}")

                loss = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
                per_token_logps.append(-loss.view(batch_size, -1))
        finally:
            if disable_checkpointing:
                model.set_activation_checkpointing(checkpoint_strategy)

        return torch.stack(per_token_logps, dim=0).to(torch.float32)

    def _compute_multimodal_rewards(
        self,
        gen_type,
        prompts_text,
        grounding_prompts,
        answer_completions_text,
        grounding_completions_text,
        inputs,
        device,
    ):
        reward_kwargs = {
            key: [example[key] for example in inputs]
            for key in inputs[0]
            if key not in ["prompt", "completion", "answer", "gt_bbox"]
        }
        answer_values = [example.get("answer") for example in inputs]
        bbox_values = [example.get("gt_bbox", example.get("answer")) for example in inputs]

        reward_names = []
        reward_columns = []
        answer_correctness_scores = None

        if gen_type != "grounding":
            answer_completions = _build_reward_completions(prompts_text, answer_completions_text)
            answer_correctness_scores = correctness_reward_func(
                prompts=prompts_text,
                completions=answer_completions,
                answer=answer_values,
                step=self._step,
                run_name=self.args.output_dir,
                **reward_kwargs,
            )
            answer_format_scores = boxed_and_answer_tags_format_reward(
                prompts=prompts_text,
                completions=answer_completions,
                answer=answer_values,
                step=self._step,
                run_name=self.args.output_dir,
                **reward_kwargs,
            )
            reward_names.extend(["answer_correctness", "answer_format"])
            reward_columns.extend([answer_correctness_scores, answer_format_scores])

        if gen_type in {"grounding", "image_gen"}:
            grounding_iou_scores = correct_grounding_reward_func(
                prompts=grounding_prompts,
                completions=grounding_completions_text,
                answer=bbox_values,
                step=self._step,
                run_name=self.args.output_dir,
                **reward_kwargs,
            )
            reward_names.append("grounding_iou")
            reward_columns.append(grounding_iou_scores)

        if gen_type == "image_gen":
            reward_names.append("image_correctness_proxy")
            reward_columns.append(list(answer_correctness_scores))

        rewards_per_func = torch.stack(
            [torch.tensor(column, dtype=torch.float32, device=device) for column in reward_columns],
            dim=1,
        )
        return rewards_per_func, reward_names

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_debug_artifacts: bool = False,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts_text = [x["prompt"] for x in inputs]
        grounding_prompts = [x.get("grounding_prompt", x["prompt"]) for x in inputs]
        edit_prompts = [x.get("edit_prompt", x["prompt"]) for x in inputs]
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        original_images_all = []
        for sample_images in images or [None] * len(inputs):
            if isinstance(sample_images, (list, tuple)):
                original_images_all.append(sample_images[0] if len(sample_images) > 0 else None)
            else:
                original_images_all.append(sample_images)
        gen_type = inputs[0].get("gen_type", "text_gen")
        assert gen_type in ["text_gen", "image_gen", "grounding"], f"Invalid generation type: {gen_type}"

        if gen_type == "image_gen":
            batch_inputs = self.processing_class(
                texts=prompts_text,
                grounding_texts=grounding_prompts,
                edit_texts=edit_prompts,
                images=images,
                edit_mode=self.args.edit_mode,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                device=device,
                dtype=torch.bfloat16,
                mode="image_gen",
                do_cfg= self.args.guidance_scale > 0 ,
            )
            generation_prompts = edit_prompts
            input_embeds = batch_inputs["input_embeds"]
            attention_mask = batch_inputs["attention_mask"]
        elif gen_type == "grounding":
            batch_inputs = self.processing_class(
                texts=grounding_prompts,
                grounding_texts=grounding_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                device=device,
                dtype=torch.bfloat16,
                mode="grounding",
            )
            generation_prompts = grounding_prompts
            input_embeds = batch_inputs["input_embeds"]
            attention_mask = batch_inputs["attention_mask"]
        else:
            batch_inputs = self.processing_class(
                texts=prompts_text,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                device=device,
                dtype=torch.bfloat16,
                mode="text_gen",
            )
            generation_prompts = prompts_text
            input_embeds = batch_inputs["input_embeds"]
            attention_mask = batch_inputs["attention_mask"]
        
        bbox_mask = batch_inputs.get("bbox_mask", None)
        pred_bboxes_all = [None] * len(inputs)
        edited_images_all = [None] * len(inputs)
        image_gen_debug_all = [None] * len(inputs)
        grounding_bbox_completion_text_all = [None] * len(inputs)

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        if self.accelerator.is_main_process:
            logger.info(
                f"[Step {self.state.global_step}] Generate  | start | "
                f"bs={input_embeds.size(0)}, gen_len={gen_length}, steps={steps}"
            )
        _t_gen = time.perf_counter()
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            # Generation should run on the currently unwrapped model instance.
            self.inferencer.model = unwrapped_model
            generation_batch_size = self.args.generation_batch_size
            init_images = None
            image_sizes = None
            bbox_ids = None
            image_gen_kwargs = None
            if gen_type == "image_gen":
                init_images = []
                target_resolution = 1024
                for sample_images in images:
                    if isinstance(sample_images, (list, tuple)):
                        sample_image = sample_images[0] if len(sample_images) > 0 else None
                    else:
                        sample_image = sample_images
                    if sample_image is None:
                        raise ValueError("image_gen sample is missing init image.")
                    sample_image = sample_image.convert("RGB")
                    if sample_image.size != (target_resolution, target_resolution):
                        sample_image = pad_to_square_and_resize(sample_image, target_resolution)
                    init_images.append(sample_image)
                image_sizes = [(target_resolution, target_resolution)] * len(init_images)

                image_gen_kwargs = dict(
                    guidance_scale=self.args.guidance_scale,
                    guidance_scale_image=self.args.guidance_scale_image,
                    edit_mode=self.args.edit_mode,
                    confidence_policy="stratified",
                    enable_stratified=True,
                    image_resolution=1024,
                    n_tokens=4096,
                    n_steps=64,
                    shift=5,
                    schedule="shift",
                    alg_temp=5,
                    dynamic_temperature=True,
                    temperature=0.8,
                    schedule_temp="cosine2",
                    min_temperature=0.5,
                )
                if return_debug_artifacts:
                    image_gen_kwargs["debug_label"] = (
                        f"step_{self.state.global_step}_batch_0_{input_embeds.size(0)}"
                    )

            gen_result = self.inferencer._generate_mode(
                gen_type=gen_type,
                tokenizer=self.processing_class.tokenizer,
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                bbox_mask=bbox_mask if gen_type in {"grounding", "image_gen"} else None,
                input_embeds_gen=batch_inputs.get("input_embeds_gen") if gen_type == "image_gen" else None,
                inputs_embeds_cond=batch_inputs.get("inputs_embeds_cond") if gen_type == "image_gen" else None,
                inputs_embeds_uncond=batch_inputs.get("inputs_embeds_uncond") if gen_type == "image_gen" else None,
                inputs_embeds_uncond_enc=batch_inputs.get("inputs_embeds_uncond_enc") if gen_type == "image_gen" else None,
                attention_mask_gen=batch_inputs.get("attention_mask_gen") if gen_type == "image_gen" else None,
                is_gen=batch_inputs.get("is_gen") if gen_type == "image_gen" else None,
                is_gen_enc=batch_inputs.get("is_gen_enc") if gen_type == "image_gen" else None,
                is_gen_enc_null=batch_inputs.get("is_gen_enc_null") if gen_type == "image_gen" else None,
                is_gen_enc_ccc=batch_inputs.get("is_gen_enc_ccc") if gen_type == "image_gen" else None,
                is_prompt=batch_inputs.get("is_prompt") if gen_type == "image_gen" else None,
                init_images=init_images,
                image_sizes=image_sizes,
                generation_prompts=generation_prompts if gen_type == "image_gen" else None,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
                generation_batch_size=generation_batch_size,
                image_gen_kwargs=image_gen_kwargs,
                return_debug=return_debug_artifacts and gen_type == "image_gen",
                processing_class=self.processing_class,
                max_prompt_length=self.max_prompt_length,
                device=device,
            )
            text_completion_ids_raw = gen_result.get("completion_ids")
            if gen_type == "image_gen":
                image_completion_ids = gen_result["edit_completion_ids"]
                edit_region_mask = gen_result["edit_region_mask"]
                image_input_embeds_gen = gen_result["image_input_embeds_gen"]
                image_is_gen = gen_result["image_is_gen"]
                image_is_gen_enc = gen_result["image_is_gen_enc"]
                image_gen_shape = self._infer_image_gen_shape(image_completion_ids)
                edited_images_all = gen_result["edited_images"]
            else:
                image_completion_ids = None
                edit_region_mask = None
                image_input_embeds_gen = None
                image_is_gen = None
                image_is_gen_enc = None
                image_gen_shape = None

            if gen_type in {"grounding", "image_gen"}:
                ground_completion_ids = gen_result.get("ground_completion_ids")
                pred_bboxes_all = gen_result["pred_bboxes"]
                bbox_texts = gen_result["bbox_texts"]
                grounding_bbox_completion_text_all = bbox_texts
            else:
                ground_completion_ids = None
            
        self.inferencer.model = self.model
        if self.accelerator.is_main_process:
            logger.info(
                f"[Step {self.state.global_step}] Generate  | done  | {time.perf_counter() - _t_gen:.1f}s"
            )

        batch_size = len(inputs)
        prompt_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        prompt_mask = torch.empty((batch_size, 0), dtype=torch.int, device=device)

        if gen_type == "grounding":
            completion_ids = ground_completion_ids
            text_completion_mask = torch.ones_like(ground_completion_ids, dtype=torch.int)
        elif gen_type == "image_gen":
            grounding_completion_mask = torch.ones_like(ground_completion_ids, dtype=torch.int)
            answer_completion_mask = self._get_text_completion_mask(text_completion_ids_raw)
            completion_ids = torch.cat([ground_completion_ids, text_completion_ids_raw], dim=1)
            text_completion_mask = torch.cat([grounding_completion_mask, answer_completion_mask], dim=1)
        else:
            completion_ids = text_completion_ids_raw
            text_completion_mask = self._get_text_completion_mask(text_completion_ids_raw)

        image_completion_mask = self._get_image_completion_mask(image_completion_ids, edit_region_mask)
        completion_mask = text_completion_mask
        if image_completion_mask is not None:
            completion_mask = torch.cat([completion_mask, image_completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        if self.args.random_masking:
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        # Expand once; used by both old-logps and ref-logps branches.
        prompt_completion_ids_expanded = completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)
        with torch.no_grad():
            if self.num_iterations > 1:
                if self.accelerator.is_main_process:
                    logger.info(
                        f"[Step {self.state.global_step}] Old logps | start | "
                        f"n_iter={self.num_iterations}, logits_to_keep={logits_to_keep}"
                    )
                _t_old = time.perf_counter()
                old_text_per_token_logps = self._get_text_per_token_logps(
                    self.accelerator.unwrap_model(self.model).get_model(),
                    prompt_completion_ids_expanded,
                    logits_to_keep,
                    mask_seeds,
                )
                old_image_per_token_logps = self._get_image_per_token_logps(
                    self.accelerator.unwrap_model(self.model).get_model(),
                    image_completion_ids,
                    image_input_embeds_gen,
                    image_is_gen,
                    image_is_gen_enc,
                    image_gen_shape,
                    mask_seeds,
                )
                all_old_per_token_logps = self._concat_token_logps(
                    old_text_per_token_logps, old_image_per_token_logps
                )
                if self.accelerator.is_main_process:
                    logger.info(
                        f"[Step {self.state.global_step}] Old logps | done  | {time.perf_counter() - _t_old:.1f}s"
                    )

            if self.args.beta == 0.0:
                ref_per_token_logps = None
            else:
                if self.accelerator.is_main_process:
                    logger.info(
                        f"[Step {self.state.global_step}] Ref logps | start | "
                        f"n_iter={self.num_iterations}, logits_to_keep={logits_to_keep}"
                    )
                _t_ref = time.perf_counter()
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_text_per_token_logps = self._get_text_per_token_logps(
                        self.accelerator.unwrap_model(self.model).get_model(),
                        prompt_completion_ids_expanded,
                        logits_to_keep,
                        mask_seeds,
                    )
                    ref_image_per_token_logps = self._get_image_per_token_logps(
                        self.accelerator.unwrap_model(self.model).get_model(),
                        image_completion_ids,
                        image_input_embeds_gen,
                        image_is_gen,
                        image_is_gen_enc,
                        image_gen_shape,
                        mask_seeds,
                    )
                    all_ref_per_token_logps = self._concat_token_logps(
                        ref_text_per_token_logps, ref_image_per_token_logps
                    )
                if self.accelerator.is_main_process:
                    logger.info(
                        f"[Step {self.state.global_step}] Ref logps | done  | {time.perf_counter() - _t_ref:.1f}s"
                    )

        answer_completions_text = [""] * batch_size
        if text_completion_ids_raw is not None:
            answer_completions_text = self.processing_class.tokenizer.batch_decode(
                text_completion_ids_raw, skip_special_tokens=True
            )

        grounding_completions_text = grounding_bbox_completion_text_all
        if ground_completion_ids is not None and not any(x is not None for x in grounding_completions_text):
            grounding_completions_text = self.processing_class.tokenizer.batch_decode(
                ground_completion_ids, skip_special_tokens=False
            )

        completions_text = grounding_completions_text if gen_type == "grounding" else answer_completions_text
        log_prompts = grounding_prompts if gen_type == "grounding" else prompts_text
        log_completions_text = completions_text
        if gen_type == "image_gen":
            log_prompts = [
                _format_image_gen_prompt_log(grounding_prompt, answer_prompt)
                for grounding_prompt, answer_prompt in zip(grounding_prompts, prompts_text)
            ]
            log_completions_text = [
                _format_image_gen_completion_log(grounding_completion, answer_completion)
                for grounding_completion, answer_completion in zip(
                    grounding_completions_text, answer_completions_text
                )
            ]
        is_multimodal_reward_routing = gen_type in {"grounding", "image_gen"} or any(
            image is not None for image in original_images_all
        )
        if is_multimodal_reward_routing:
            rewards_per_func, reward_names = self._compute_multimodal_rewards(
                gen_type,
                prompts_text,
                grounding_prompts,
                answer_completions_text,
                grounding_completions_text,
                inputs,
                device,
            )
            rewards_per_func = gather(rewards_per_func)
            rewards = rewards_per_func.sum(dim=1)
        else:
            reward_prompts = prompts_text
            completions = _build_reward_completions(reward_prompts, completions_text)
            rewards_per_func = torch.zeros(len(prompts_text), len(self.reward_funcs), device=device)
            last_reward_prompts = reward_prompts
            last_reward_completions = completions
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                reward_func_name = _get_reward_func_name(reward_func)
                with profiling_context(self, reward_func_name):
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    if reward_func_name == "coding_reward_func":
                        reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                    last_reward_prompts = reward_prompts
                    last_reward_completions = completions
                    output_reward_func = reward_func(
                        prompts=reward_prompts,
                        completions=completions,
                        step=self._step,
                        run_name=self.args.output_dir,
                        **reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan for reward in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            if torch.isnan(rewards_per_func).all(dim=1).any():
                nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
                row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
                row_reward_kwargs["prompt"] = last_reward_prompts[nan_row_idx]
                row_reward_kwargs["completion"] = last_reward_completions[nan_row_idx]
                warnings.warn(
                    f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                    "Please ensure that at least one reward function returns a valid reward."
                )

            rewards_per_func = gather(rewards_per_func)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            reward_names = [_get_reward_func_name(reward_func) for reward_func in self.reward_funcs]

        reward_columns_to_log = {"reward": rewards.tolist()}
        for i, reward_name in enumerate(reward_names):
            reward_columns_to_log[reward_name] = rewards_per_func[:, i].tolist()

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func_name in enumerate(reward_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(log_prompts)
            completions_to_log = gather_object(log_completions_text)
            rewards_to_log = reward_columns_to_log["reward"]
            advantages_to_log = gather_object(advantages.tolist())
            gen_types_to_log = gather_object([gen_type] * len(log_prompts))
            pred_bboxes_to_log = gather_object(pred_bboxes_all)
            gt_bboxes_to_log = gather_object([example.get("gt_bbox") for example in inputs])
            original_images_to_log = gather_object(original_images_all)
            edited_images_to_log = gather_object(edited_images_all)

            

            if self.accelerator.is_main_process:
                num_rows = len(rewards_to_log)
                prompts_to_log = _align_len(prompts_to_log, num_rows)
                completions_to_log = _align_len(completions_to_log, num_rows)
                advantages_to_log = _align_len(advantages_to_log, num_rows)
                gen_types_to_log = _align_len(gen_types_to_log, num_rows)
                pred_bboxes_to_log = _align_len(pred_bboxes_to_log, num_rows)
                gt_bboxes_to_log = _align_len(gt_bboxes_to_log, num_rows)
                original_images_to_log = _align_len(original_images_to_log, num_rows)
                edited_images_to_log = _align_len(edited_images_to_log, num_rows)
                reward_columns_to_log = {
                    key: _align_len(values, num_rows)
                    for key, values in reward_columns_to_log.items()
                }

                completion_log_sample_ratio = getattr(self.args, "completion_log_sample_ratio", 0.1)
                sampled_indices = _sample_log_indices(
                    num_rows,
                    completion_log_sample_ratio,
                    step_seed=self.state.global_step,
                )
                if sampled_indices:
                    prompts_to_log = _select_by_indices(prompts_to_log, sampled_indices)
                    completions_to_log = _select_by_indices(completions_to_log, sampled_indices)
                    advantages_to_log = _select_by_indices(advantages_to_log, sampled_indices)
                    gen_types_to_log = _select_by_indices(gen_types_to_log, sampled_indices)
                    pred_bboxes_to_log = _select_by_indices(pred_bboxes_to_log, sampled_indices)
                    gt_bboxes_to_log = _select_by_indices(gt_bboxes_to_log, sampled_indices)
                    original_images_to_log = _select_by_indices(original_images_to_log, sampled_indices)
                    edited_images_to_log = _select_by_indices(edited_images_to_log, sampled_indices)
                    reward_columns_to_log = {
                        key: _select_by_indices(values, sampled_indices)
                        for key, values in reward_columns_to_log.items()
                    }
                else:
                    prompts_to_log = []
                    completions_to_log = []
                    advantages_to_log = []
                    gen_types_to_log = []
                    pred_bboxes_to_log = []
                    gt_bboxes_to_log = []
                    original_images_to_log = []
                    edited_images_to_log = []
                    reward_columns_to_log = {key: [] for key in reward_columns_to_log}

                _log_prompt_completion_samples_rich(
                    prompts_to_log,
                    completions_to_log,
                    reward_columns_to_log,
                    advantages_to_log,
                    self.state.global_step,
                )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    num_rows = len(prompts_to_log)
                    reward_columns_for_table = reward_columns_to_log
                    should_log_images = (
                        gen_type == "image_gen"
                        and getattr(self.args, "image_log_steps", 50) > 0
                        and self.state.global_step % getattr(self.args, "image_log_steps", 50) == 0
                    )
                    original_image_artifacts = [None] * num_rows
                    edited_image_artifacts = [None] * num_rows
                    if should_log_images and num_rows > 0:
                        image_sample_ratio = getattr(self.args, "image_log_sample_ratio", 0.1)
                        image_sample_indices = set(
                            _sample_log_indices(
                                num_rows,
                                image_sample_ratio,
                                step_seed=self.state.global_step + 17,
                            )
                        )
                        for idx, orig_img in enumerate(original_images_to_log):
                            if idx in image_sample_indices and isinstance(orig_img, Image.Image):
                                original_image_artifacts[idx] = wandb.Image(orig_img)
                        for idx, edited_img in enumerate(edited_images_to_log):
                            if idx in image_sample_indices and isinstance(edited_img, Image.Image):
                                edited_image_artifacts[idx] = wandb.Image(edited_img)

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * num_rows,
                        "gen_type": gen_types_to_log,
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        **reward_columns_for_table,
                        "pred_bbox": [str(x) if x is not None else None for x in pred_bboxes_to_log],
                        "gt_bbox": [str(x) if x is not None else None for x in gt_bboxes_to_log],
                    }
                    if should_log_images:
                        table["original_image"] = original_image_artifacts
                        table["edited_image"] = edited_image_artifacts
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,  # Store all mask seeds for consistent mask patterns
        }
        if text_completion_ids_raw is not None:
            result["text_completion_ids_raw"] = text_completion_ids_raw
            result["answer_completions_text"] = answer_completions_text
        if ground_completion_ids is not None:
            result["ground_completion_ids"] = ground_completion_ids
            result["grounding_completions_text"] = grounding_completions_text
        if image_completion_ids is not None:
            result["image_completion_ids"] = image_completion_ids
            result["image_completion_mask"] = image_completion_mask
            result["edit_region_mask"] = edit_region_mask
            result["image_input_embeds_gen"] = image_input_embeds_gen
            result["image_is_gen"] = image_is_gen
            result["image_is_gen_enc"] = image_is_gen_enc
            result["image_gen_shape"] = image_gen_shape
        if return_debug_artifacts and gen_type == "image_gen":
            result["pred_bboxes"] = pred_bboxes_all
            result["edited_images"] = edited_images_all
            result["image_gen_debug"] = image_gen_debug_all
            result["grounding_bbox_completion_text"] = grounding_bbox_completion_text_all
        return result
