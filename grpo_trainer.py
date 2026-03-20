import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional, Union
from collections import defaultdict
import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME

from accelerate.utils import DistributedType, gather, gather_object

from trl.extras.profiling import profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer, nanstd

from llava.mm_utils import pad_to_square_and_resize
from llava.model.language_model.llada.generate import get_logits, wte as llada_wte

from interleaved_inferencer import InterleavedInferencer
from reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correct_grounding_reward_func,
    perceptual_score_reward_func,
)

logger = logging.getLogger(__name__)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
Batch = dict[str, Union[torch.Tensor, Any]]


def _clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DiffuGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer for diffusion-style multimodal language models.

    High-level flow, kept close to TRL GRPOTrainer:
        1) process inputs
        2) generate modality-specific completions
        3) compute rewards and advantages
        4) cache old / reference per-token log-probs
        5) pack all modality rows into one flat batch
        6) compute current per-token log-probs in compute_loss()
        7) apply GRPO importance weighting + clipping

    Supported modalities:
        - grounding text
        - answer text
        - image tokens
    """

    MODALITY_GROUND = 0
    MODALITY_ANSWER = 1
    MODALITY_IMAGE = 2

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
        optimizers: tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
        model_init_fn=None,
        peft_config: Optional[Any] = None,
        gen_type: str = "text_gen",
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

        self.max_prompt_length = args.max_prompt_length if args is not None else None
        self._buffered_inputs: Optional[list[Batch]] = {}
        self.model_init_fn = model_init_fn
        self.inferencer = InterleavedInferencer(self.model)
        self._current_train_step_time = 0.0
        self.gen_type = gen_type

        self.resolution = 1024
        self.image_gen_kwargs = dict(
            guidance_scale=args.guidance_scale,
            guidance_scale_image=args.guidance_scale_image,
            edit_mode=args.edit_mode,
            confidence_policy="stratified",
            enable_stratified=True,
            image_resolution=self.resolution,
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

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    def _offload_cached_logps(self, batch: Batch) -> Batch:
        """
        Keep cached rollout log-probs off GPU between generation and compute_loss().
        These tensors are detached baselines/reference values, so bf16 CPU storage is
        enough and lowers the ZeRO-2 backward peak.
        """
        for key in ("old_per_token_logps", "ref_per_token_logps"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.device.type != "cpu":
                batch[key] = value.to(device="cpu", dtype=torch.bfloat16)
        return batch

    def _save_checkpoint(self, model, trial):
        original_processing_class = self.processing_class
        self.processing_class = self.processing_class.tokenizer
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self.processing_class = original_processing_class

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model + tokenizer in HF format so model_init_fn can reload from checkpoint dir.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not self.args.should_save:
            return

        # unwrapped_model: PreTrainedModel
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            output_dir,
            safe_serialization=self.args.save_safetensors,
        )

        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        """
        Load checkpoint via model_init_fn, then copy into the already-prepared trainer model.
        """
        if model is None:
            model = self.model

        current_model = self.accelerator.unwrap_model(self.model)
        current_config = getattr(current_model, "config", None)
        add_vision_tokens = getattr(current_config, "add_vision_tokens", False)

        logger.info("Loading model checkpoint from %s", resume_from_checkpoint)
        _, loaded_model, _, _ = self.model_init_fn(resume_from_checkpoint, add_vision_tokens)
        loaded_model.to(torch.bfloat16)

        load_result = model.load_state_dict(loaded_model.state_dict(), strict=False)
        if load_result.missing_keys:
            logger.warning("Checkpoint load missing keys: %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logger.warning("Checkpoint load unexpected keys: %s", load_result.unexpected_keys)

        self.inferencer.model = self.model

        del loaded_model
        _clear_device_cache()
        logger.info("Successfully loaded checkpoint from %s", resume_from_checkpoint)

    # -------------------------------------------------------------------------
    # Generic tensor packer
    # -------------------------------------------------------------------------

    def _pad_and_concat(
        self,
        tensors: list[torch.Tensor],
        pad_dim: int,
        cat_dim: int,
        pad_value: float,
        padding_side: str = "right",
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """
        Pad all tensors along `pad_dim` up to the maximum size, then concatenate along `cat_dim`.
        Typical usage:
            ids / masks:
                [B_i, L_i] -> pad on L -> [B_i, L_max] -> concat on B -> [sum(B_i), L_max]

            prompt embeds:
                [B_i, L_i, D] -> pad on L -> [B_i, L_max, D] -> concat on B -> [sum(B_i), L_max, D]

            old / ref log-probs:
                [N_iter, B_i, T_i] -> pad on T -> [N_iter, B_i, T_max] -> concat on B -> [N_iter, sum(B_i), T_max]
        """
        if not tensors:
            return None

        rank = tensors[0].dim()
        pad_dim = pad_dim % rank
        cat_dim = cat_dim % rank

        max_pad_len = max(t.size(pad_dim) for t in tensors)

        padded_tensors: list[torch.Tensor] = []
        for tensor in tensors:
            tensor = tensor.to(dtype or tensor.dtype)
            pad_len = max_pad_len - tensor.size(pad_dim)

            if pad_len > 0:
                pad_shape = list(tensor.shape)
                pad_shape[pad_dim] = pad_len

                pad_tensor = torch.full(
                    pad_shape,
                    pad_value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )

                if padding_side == "left":
                    tensor = torch.cat([pad_tensor, tensor], dim=pad_dim)
                else:
                    tensor = torch.cat([tensor, pad_tensor], dim=pad_dim)

            padded_tensors.append(tensor)

        return torch.cat(padded_tensors, dim=cat_dim)


    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        all_per_token_logps = []
        all_old_per_token_logps = []
        all_completion_masks = []
        all_advantages = []
        this_itr_idx = self._step % self.args.num_iterations
        for modality in ["answer", "ground", "image"]:
            
            if f"{modality}_prompt_ids" in inputs:
                prompt_ids, prompt_mask = inputs[f"{modality}_prompt_ids"], inputs[f"{modality}_prompt_mask"]
                completion_ids, completion_mask = inputs[f"{modality}_completion_ids"], inputs[f"{modality}_completion_mask"]
                all_completion_masks.append(completion_mask)
                all_old_per_token_logps.append(inputs[f"{modality}_old_per_token_logps"])
                all_advantages.append(inputs[f"{modality}_advantages"])
                mask_seeds = inputs[f"{modality}_mask_seeds"]
                this_itr_mask_seed = mask_seeds[this_itr_idx]
                
                input_embeds = inputs[f"{modality}_prompt_input_embeds"] # 1, bsz, prompt_seq_len, hidden_dim
                logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens
                if modality in ["answer", "ground"]:
                    per_token_logps = self._get_text_per_token_logps(
                        model = model.get_model(),
                        input_ids = completion_ids.unsqueeze(0),
                        prompt_input_embeds = input_embeds.unsqueeze(0),
                        logits_to_keep=logits_to_keep,
                        mask_seeds=[this_itr_mask_seed],
                    )  # (num_iter, bsz, answer_completion_seq_len)
                elif modality in ["image"]:
                    per_token_logps = self._get_image_per_token_logps(
                        model = model.get_model(),
                        image_completion_ids = completion_ids.unsqueeze(0),
                        image_input_embeds_gen = input_embeds.unsqueeze(0),
                        image_is_gen = inputs["is_gen"].unsqueeze(0),
                        image_is_gen_enc = inputs["is_gen_enc"].unsqueeze(0),
                        image_gen_shape = self._infer_image_gen_shape(completion_ids),
                        mask_seeds=[this_itr_mask_seed],
                        logits_to_keep=logits_to_keep,
                    )  # (num_iter, bsz, flat_image_completion_len)
                all_per_token_logps.append(per_token_logps)


                # Compute the KL divergence between the model and the reference model
                if self.beta != 0.0:
                    # ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
                    # per_token_kl = (
                    #     torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                    # )
                    raise NotImplementedError("KL divergence not implemented for multi-modal training")
                
        # pad and concat  
        per_token_logps = self._pad_and_concat(all_per_token_logps, pad_dim=2, cat_dim=1, pad_value=0.0)[0] # (bsz, completion_seq_len)
        old_per_token_logps = self._pad_and_concat(all_old_per_token_logps, pad_dim=2, cat_dim=1, pad_value=0.0)[this_itr_idx] # (bsz, completion_seq_len)
        advantages = torch.cat(all_advantages, dim=0) # bsz
        completion_mask = self._pad_and_concat(all_completion_masks, pad_dim=1, cat_dim=0, pad_value=0.0) # bsz, completion_seq_len
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon_low, 1 + self.args.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            raise NotImplementedError("KL divergence not implemented for multi-modal training")

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    # -------------------------------------------------------------------------
    # Masking / completion masks
    # -------------------------------------------------------------------------

    def forward_process(
        self,
        batch: torch.Tensor,
        mask_id: int,
        seed: Optional[Union[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupt completion tokens for diffusion log-prob evaluation.

        batch:        (bsz, seq_len)
        prompt_index: (bsz, seq_len) bool; True only on prompt positions

        returns:
            noisy_batch: (bsz, seq_len)
        """
        assert batch.dim() == 2, f"Batch must be 2D, got {batch.dim()}D but got {batch.size()}"
        if isinstance(seed, torch.Tensor):
            seed = seed.item()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=batch.device)
            generator.manual_seed(int(seed))

        bsz, seq_len = batch.shape
        t_p = torch.ones(bsz, device=batch.device) * self.args.p_mask_prompt  # (bsz,)
        random_matrix = torch.rand((bsz, seq_len), device=batch.device, generator=generator)  # (bsz, seq_len)

        is_mask = (random_matrix < t_p.unsqueeze(1))  # (bsz, seq_len)
        noisy_batch = torch.where(is_mask, mask_id, batch)  # (bsz, seq_len)

        return noisy_batch

    def _get_text_completion_mask(self, completion_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        completion_ids: (bsz, completion_seq_len)

        returns:
            completion_mask: (bsz, completion_seq_len)
        """
        if completion_ids is None:
            return None

        if completion_ids.size(1) == 0:
            return torch.zeros(
                (completion_ids.size(0), 0),
                dtype=torch.int,
                device=completion_ids.device,
            )

        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id  # (bsz, completion_seq_len)
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )  # (bsz,)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

        sequence_indices = torch.arange(
            is_eos.size(1),
            device=completion_ids.device,
        ).expand(is_eos.size(0), -1)  # (bsz, completion_seq_len)

        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        return completion_mask, completion_ids_list, completion_lengths

    def _get_image_completion_mask(
        self,
        image_completion_ids: Optional[torch.Tensor],
        edit_region_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        image_completion_ids:
            non-unitok: (bsz_image, image_token_len)
            unitok:     (bsz_image, codebook_dim, image_token_len)

        edit_region_mask:
            (bsz_image, image_token_len)

        returns:
            image_completion_mask:
                non-unitok: (bsz_image, image_token_len)
                unitok:     (bsz_image, codebook_dim * image_token_len)
        """
        if image_completion_ids is None:
            return None

        if image_completion_ids.dim() == 2:
            if edit_region_mask is not None:
                return edit_region_mask.to(dtype=torch.int, device=image_completion_ids.device)
            return torch.ones_like(image_completion_ids, dtype=torch.int, device=image_completion_ids.device)

        if image_completion_ids.dim() == 3:
            if edit_region_mask is not None:
                expanded_mask = edit_region_mask.unsqueeze(1).expand(
                    -1, image_completion_ids.size(1), -1
                )  # (bsz_image, codebook_dim, image_token_len)

                return expanded_mask.reshape(image_completion_ids.size(0), -1).to(
                    dtype=torch.int,
                    device=image_completion_ids.device,
                )  # (bsz_image, codebook_dim * image_token_len)

            return torch.ones(
                (image_completion_ids.size(0), image_completion_ids.size(1) * image_completion_ids.size(2)),
                dtype=torch.int,
                device=image_completion_ids.device,
            )

        raise ValueError(f"Unsupported image completion shape: {tuple(image_completion_ids.shape)}")

    def _infer_image_gen_shape(self, image_completion_ids: torch.Tensor) -> tuple[int, int]:
        """
        Infer spatial image token grid from the last dimension.

        image_completion_ids:
            non-unitok: (bsz_image, image_token_len)
            unitok:     (bsz_image, codebook_dim, image_token_len)

        returns:
            image_gen_shape: (grid_h, grid_w)
        """
        token_count = image_completion_ids.shape[-1]
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError(f"Cannot infer square image gen shape from {token_count} tokens.")
        return (side, side)

    # -------------------------------------------------------------------------
    # Log-prob computation
    # -------------------------------------------------------------------------

    def _get_text_per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,
        prompt_input_embeds: torch.Tensor,
        logits_to_keep: int,
        mask_seeds: list[Union[int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute text per-token log-probs following the model's native generate_text-style forward path.

        input_ids:           (num_iter, bsz, completion_tokens) = [completion tokens]
        prompt_input_embeds: (num_iter, bsz, prompt_seq_len, hidden_dim)
        logits_to_keep:      scalar = completion_seq_len

        returns:
            per_token_logps: (num_iter, bsz, completion_seq_len)
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        
        assert len(mask_seeds) == num_iterations, f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"
        core_model = model.get_model() if hasattr(model, "get_model") else model

        past_key_values = None
        
        with torch.no_grad():
            prompt_input_embeds_flat = prompt_input_embeds.reshape(
                num_iterations * batch_size,
                prompt_input_embeds.size(-2),
                prompt_input_embeds.size(-1),
            )
            past_key_values = core_model(
                None,
                input_embeddings=prompt_input_embeds_flat,
                use_cache=True,
                modality_indices=None,
            ).attn_key_values
            
        
        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            x = input_ids[iter_idx].clone()  # (bsz, full_seq_len)
            all_expanded_inputs.append(x)

            perturbed_seq = self.forward_process(
                batch=x,
                mask_id=self.args.mask_id,
                seed=mask_seed,
            )  # -> (bsz, completion_tokens)

            all_perturbed_seqs.append(perturbed_seq)

        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # (num_iterations * bsz, full_seq_len)
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # (num_iterations * bsz, full_seq_len)

        completion_seq = perturbed_seq  # (num_iterations * bsz, completion_seq_len)

        inputs_embeds_curr, new_token_mask = llada_wte(
            core_model,
            completion_seq,
            t2i_inference=False,
        )
        logits = get_logits(
            core_model,
            inputs_embeds_curr,
            new_token_mask,
            t2i_inference=False,
            past_key_values=past_key_values,
        )  # (num_iterations * bsz, completion_seq_len, vocab_size)

        completion_logits = logits[:, -logits_to_keep:, :]  # (num_iterations * bsz, completion_seq_len, vocab_size)
        completion_targets = expanded_input[:, -logits_to_keep:]  # (num_iterations * bsz, completion_seq_len)

        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))  # (num_iterations * bsz * completion_seq_len, vocab_size)
        flat_targets = completion_targets.reshape(-1)  # (num_iterations * bsz * completion_seq_len,)

        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        logps = -loss.view(num_iterations * batch_size, logits_to_keep)  # (bsz, completion_seq_len)
        per_token_logps = logps.view(num_iterations, batch_size, logits_to_keep)

        # has_ckpt_toggle = hasattr(core_model, "set_activation_checkpointing")
        # checkpoint_strategy = getattr(core_model, "activation_checkpointing_strategy", None)

        # try:
        #     if has_ckpt_toggle:
        #         core_model.set_activation_checkpointing(None)

        #     with self._gather_deepspeed_zero3_params(core_model):
                
        # finally:
        #     if has_ckpt_toggle:
        #         core_model.set_activation_checkpointing(checkpoint_strategy)
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _get_image_per_token_logps(
        self,
        model,
        image_completion_ids: torch.Tensor,
        image_input_embeds_gen: torch.Tensor,
        image_is_gen: torch.Tensor,
        image_is_gen_enc: Optional[torch.Tensor],
        image_gen_shape: Optional[tuple[int, int]],
        logits_to_keep,
        mask_seeds: list[Union[int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute image token log-probs following the model's native generate_image-style forward path.

        image_completion_ids:
            non-unitok: (num_iter, bsz_image, image_token_len)
            unitok:     (num_iter, bsz_image, codebook_dim, image_token_len)

        image_input_embeds_gen:
            (num_iter, bsz_image, seq_len_model, hidden_dim)

        image_is_gen:
            (num_iter, bsz_image, seq_len_model)

        image_is_gen_enc:
            (num_iter, bsz_image, seq_len_model)

        returns:
            per_token_logps:
                non-unitok: (num_iter, bsz_image, image_token_len)
                unitok:     (num_iter, bsz_image, codebook_dim * image_token_len)
        """
        img_mask_id = 8193
        num_iterations, batch_size, seq_len = image_completion_ids.size()

        image_is_gen = image_is_gen.to(device=image_completion_ids.device, dtype=torch.bool)
        image_is_gen_enc = image_is_gen_enc.to(device=image_completion_ids.device, dtype=torch.bool)

        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=image_completion_ids.device)  # (bsz, full_seq_len)

        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            x = image_completion_ids[iter_idx].clone()  # (bsz_image, image_token_len)
            perturbed_seq = self.forward_process(
                batch=x,
                mask_id=img_mask_id,
                seed=mask_seed,
            )  # -> (num_iter, bsz, image_token_len)
            all_expanded_inputs.append(x)
            all_perturbed_seqs.append(perturbed_seq)
        
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # (num_iterations * bsz_image, image_token_len)
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # (num_iterations * bsz_image, image_token_len)
        mask_idx = perturbed_seq == img_mask_id
        n_mask_per_sample = mask_idx.sum(dim=1)
        timesteps = n_mask_per_sample.float() / mask_idx.shape[1]
        image_input_embeds_gen_flat = image_input_embeds_gen.reshape(
            num_iterations * batch_size,
            image_input_embeds_gen.size(-2),
            image_input_embeds_gen.size(-1),
        )
        image_is_gen_flat = image_is_gen.reshape(-1, image_is_gen.size(-1))
        all_input_embeddings, new_token_mask = llada_wte(
            model,
            None,
            True,
            x_gen=perturbed_seq,
            gen_shape=image_gen_shape,
            inputs_embeds_curr=image_input_embeds_gen_flat,
            new_token_mask=image_is_gen_flat,
        )
        enc_use_image_branch = getattr(model.config, "enc_use_image_branch", False)
        if enc_use_image_branch:
            modality_indices = image_is_gen | image_is_gen_enc
        else:
            modality_indices = image_is_gen
        # modality_indices: (bsz_image, seq_len_model)

        completion_logits = get_logits(
            model,
            all_input_embeddings,
            new_token_mask,
            True,
            gen_shape=image_gen_shape,
            input_modality_indices=modality_indices.reshape(-1, modality_indices.size(-1)),
            timesteps=timesteps,
        ) # num_iterations * bsz_image, image_token_len, vocab_size
        completion_targets = expanded_input[:, -logits_to_keep:]  # (num_iterations * bsz_image, completion_seq_len)
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))  # (num_iterations * bsz_image * completion_seq_len, vocab_size)
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
        logps = -loss.view(num_iterations * batch_size, logits_to_keep)  # (bsz_image, flat_image_completion_len)
        per_token_logps = logps.view(num_iterations, batch_size, logits_to_keep)
        del perturbed_seq, completion_logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps
        # try:
        #     if has_ckpt_toggle:
        #         core_model.set_activation_checkpointing(None)

        #     with self._gather_deepspeed_zero3_params(core_model):

        # finally:
        #     if has_ckpt_toggle:
        #         core_model.set_activation_checkpointing(checkpoint_strategy)

    # -------------------------------------------------------------------------
    # Input preparation
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Generation + reward scoring
    # -------------------------------------------------------------------------

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Any]],
        return_debug_artifacts: bool = False,
    ) -> Batch:
        """
        End-to-end rollout:
            - process inputs
            - generate modality-specific completions
            - compute rewards / advantages
            - compute old / ref log-probs
            - pack all modality rows into one flat batch
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        local_source_batch_size = len(inputs)

        # Per-source-example python lists, length == local_source_batch_size.
        answer_prompts = [x["answer_prompt"] for x in inputs]
        grounding_prompts = (
            [x["grounding_prompt"] for x in inputs]
            if inputs[0].get("grounding_prompt") is not None
            else None
        )
        image_prompts = (
            [x["edit_prompt"] for x in inputs]
            if inputs[0].get("edit_prompt") is not None
            else None
        )

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None

        init_images = []
        if images is not None:
            for sample_images in images:
                sample_image = sample_images[0] if isinstance(sample_images, (list, tuple)) else sample_images
                if sample_image is None:
                    raise ValueError("image_gen sample is missing init image.")
                sample_image = sample_image.convert("RGB")
                if sample_image.size != (self.resolution, self.resolution):
                    sample_image = pad_to_square_and_resize(sample_image, self.resolution)
                init_images.append(sample_image)

            image_sizes = [(self.resolution, self.resolution)] * len(init_images)
        else:
            init_images = None
            image_sizes = None

        prompt_inputs = self.processing_class(
            texts=answer_prompts,
            grounding_texts=grounding_prompts,
            edit_texts=image_prompts,
            images=images,
            edit_mode=self.args.edit_mode,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            device=device,
            dtype=torch.bfloat16,
            mode=self.gen_type,
            do_cfg=self.args.guidance_scale > 0,
        )
        # prompt_inputs tensor shapes:
        #   input_ids:                (bsz_answer, answer_prompt_seq_len)
        #   attention_mask:           (bsz_answer, answer_prompt_seq_len)
        #   input_embeds:             (bsz_answer, answer_prompt_seq_len, hidden_dim)
        #
        #   input_ids_grounding:      (bsz_ground, ground_prompt_seq_len)
        #   attention_mask_grounding: (bsz_ground, ground_prompt_seq_len)
        #   input_embeds_grounding:   (bsz_ground, ground_prompt_seq_len, hidden_dim)
        #
        #   input_ids_gen:            (bsz_image, image_prompt_seq_len)
        #   attention_mask_gen:       (bsz_image, image_prompt_seq_len)
        #   input_embeds_gen:         (bsz_image, seq_len_model, hidden_dim)
        #   is_gen:                   (bsz_image, seq_len_model)
        #   is_gen_enc:               (bsz_image, seq_len_model)

        generation_batch_size = self.args.generation_batch_size
        with (
            unwrap_model_for_generation(self.model_wrapped, self.accelerator),
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            self.inferencer.model = self.model
            gen_result = self.inferencer._generate_mode(
                gen_type=self.gen_type,
                tokenizer=self.processing_class.tokenizer,
                init_images=init_images,
                image_sizes=image_sizes,
                steps=self.args.diffusion_steps,
                gen_length=self.args.max_completion_length,
                block_length=self.args.block_length,
                temperature=self.args.temperature,
                cfg_scale=self.args.cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
                generation_batch_size=generation_batch_size,
                image_gen_kwargs=self.image_gen_kwargs,
                return_debug=return_debug_artifacts and self.gen_type == "image_gen",
                processing_class=self.processing_class,
                max_prompt_length=self.max_prompt_length,
                device=device,
                answer_prompts=answer_prompts,
                **prompt_inputs,
            )

        tokenizer = self.processing_class.tokenizer

        # Generated tensors:
        ground_prompt_ids = prompt_inputs.get("input_ids_grounding")  # (bsz_ground, ground_prompt_seq_len)
        ground_prompt_mask = prompt_inputs.get("attention_mask_grounding")  # (bsz_ground, ground_prompt_seq_len)
        ground_prompt_input_embeds = prompt_inputs.get("input_embeds_grounding")  # (bsz_ground, ground_prompt_seq_len, hidden_dim)
        ground_completion_ids = gen_result.get("ground_completion_ids")  # (bsz_ground, ground_completion_seq_len)

        answer_prompt_ids = prompt_inputs["input_ids"]  # (bsz_answer, answer_prompt_seq_len)
        answer_prompt_mask = prompt_inputs["attention_mask"]  # (bsz_answer, answer_prompt_seq_len)
        answer_prompt_input_embeds = prompt_inputs["input_embeds"]  # (bsz_answer, answer_prompt_seq_len, hidden_dim)
        answer_completion_ids = gen_result["completion_ids"]  # (bsz_answer, answer_completion_seq_len)

        image_prompt_ids = prompt_inputs.get("input_ids_gen")  # (bsz_image, image_prompt_seq_len)
        image_prompt_mask = prompt_inputs.get("attention_mask_gen")  # (bsz_image, image_prompt_seq_len)
        image_prompt_input_embeds = prompt_inputs.get("input_embeds_gen")  # (bsz_image, seq_len_model, hidden_dim)
        image_is_gen = prompt_inputs.get("is_gen")  # (bsz_image, seq_len_model)
        image_is_gen_enc = prompt_inputs.get("is_gen_enc")  # (bsz_image, seq_len_model)
        image_completion_ids = gen_result.get("edit_completion_ids")  # (bsz_image, image_token_len) or (bsz_image, codebook_dim, image_token_len)
        edit_region_mask = gen_result.get("edit_region_mask")  # (bsz_image, image_token_len)
        image_completions = gen_result.get("edited_images")  # (bsz_image, image_token_len) or (bsz_image, codebook_dim, image_token_len)
        
        answer_completion_mask, answer_completion_ids_list, answer_completion_lengths = self._get_text_completion_mask(answer_completion_ids)  # (bsz_answer, answer_completion_seq_len)
        ground_completion_mask, ground_completion_ids_list, ground_completion_lengths = (
            self._get_text_completion_mask(ground_completion_ids)
            if ground_completion_ids is not None
            else (None, None, None)
        ) # (bsz_ground, ground_completion_seq_len)
        image_completion_mask = (
            self._get_image_completion_mask(image_completion_ids, edit_region_mask)
            if image_completion_ids is not None
            else (None, None, None)
        )  # (bsz_image, flat_image_completion_len)

        mask_seeds = (
            torch.randint(0, 2**12, (self.num_iterations,), device=device)
            if self.args.random_masking
            else torch.full((self.num_iterations,), 42, device=device, dtype=torch.long)
        )  # (num_iter,)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        

        ground_old_text_per_token_logps = None
        answer_old_text_per_token_logps = None
        old_image_per_token_logps = None

        with torch.no_grad():
            answer_completion_ids_expanded = answer_completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)
            answer_prompt_input_embeds_expanded = answer_prompt_input_embeds.unsqueeze(0).expand(self.num_iterations, -1, -1, -1)
            # TODO: Add attention mask to per_token_logps
            answer_old_text_per_token_logps = self._get_text_per_token_logps(
                model=unwrapped_model.get_model(),
                input_ids=answer_completion_ids_expanded,
                prompt_input_embeds=answer_prompt_input_embeds_expanded,
                logits_to_keep=answer_completion_ids.size(1),
                mask_seeds=mask_seeds.tolist(),
            )  # (num_iter, bsz, answer_completion_seq_len)

            if ground_completion_ids is not None:
                ground_completion_ids_expanded = ground_completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)
                ground_prompt_input_embeds_expanded = ground_prompt_input_embeds.unsqueeze(0).expand(self.num_iterations, -1, -1, -1)
                ground_old_text_per_token_logps = self._get_text_per_token_logps(
                    model=unwrapped_model.get_model(),
                    input_ids=ground_completion_ids_expanded,
                    prompt_input_embeds=ground_prompt_input_embeds_expanded,
                    logits_to_keep=ground_completion_ids.size(1),
                    mask_seeds=mask_seeds.tolist(),
                )  # (num_iter, bsz, ground_completion_seq_len)
                assert ground_old_text_per_token_logps.shape[1] == answer_old_text_per_token_logps.shape[1], f"Expected ground old per-token logps batch size to be {answer_old_text_per_token_logps.shape[1]}, got {ground_old_text_per_token_logps.shape[1]}"
            else:
                ground_old_text_per_token_logps = None
            
            if image_completion_ids is not None:
                image_completion_ids_expanded = image_completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)
                input_embeds_gen_expanded = prompt_inputs["input_embeds_gen"].unsqueeze(0).expand(self.num_iterations, -1, -1, -1)
                is_gen_expanded = prompt_inputs["is_gen"].unsqueeze(0).expand(self.num_iterations, -1, -1)
                is_gen_enc_expanded = prompt_inputs["is_gen_enc"].unsqueeze(0).expand(self.num_iterations, -1, -1)
                old_image_per_token_logps = self._get_image_per_token_logps(
                    model=unwrapped_model.get_model(),
                    image_completion_ids=image_completion_ids_expanded,
                    image_input_embeds_gen=input_embeds_gen_expanded,
                    image_is_gen=is_gen_expanded,
                    image_is_gen_enc=is_gen_enc_expanded,
                    image_gen_shape=self._infer_image_gen_shape(image_completion_ids),
                    logits_to_keep=image_completion_ids.size(1),
                    mask_seeds=mask_seeds.tolist(),
                )  # (num_iter, bsz, flat_image_completion_len)
                assert old_image_per_token_logps.shape[1] == answer_old_text_per_token_logps.shape[1], f"Expected image old per-token logps batch size to be {answer_old_text_per_token_logps.shape[1]}, got {old_image_per_token_logps.shape[1]}"

            else:
                old_image_per_token_logps = None
            
            if self.beta == 0.0: # No KL divergence
                ref_ground_text_per_token_logps = None
                ref_answer_text_per_token_logps = None
                ref_image_per_token_logps = None
            else:
                raise NotImplementedError("KL divergence term is not supported")
                

        grounding_completions = (
            tokenizer.batch_decode(ground_completion_ids, skip_special_tokens=True)
            if ground_completion_ids is not None
            else None
        )
        answer_completions = tokenizer.batch_decode(answer_completion_ids, skip_special_tokens=True)

        all_outputs = {}
        modalities = ["answer", "ground", "image"]
        reward_weights_list = [[2.0, 1.0], [1.0], [1.0]]
        reward_fncs = [
            {"correctness": correctness_reward_func, "format": boxed_and_answer_tags_format_reward},
            {"grounding": correct_grounding_reward_func}, 
            {"perceptual": perceptual_score_reward_func}
        ]
        
        modality_dict_list = [
            {"prompts": answer_prompts, "completions": answer_completions, "completion_ids": answer_completion_ids, "prompt_ids": answer_prompt_ids, "prompt_mask": answer_prompt_mask, "prompt_input_embeds": answer_prompt_input_embeds, "completion_mask": answer_completion_mask, "mask_seeds": mask_seeds, "old_per_token_logps": answer_old_text_per_token_logps, "ref_per_token_logps": None, "completion_ids_list": answer_completion_ids_list, "completion_lengths": answer_completion_lengths},
            {"prompts": grounding_prompts, "completions": grounding_completions, "completion_ids": ground_completion_ids, prompt_ids": ground_prompt_ids, "prompt_mask": ground_prompt_mask, "prompt_input_embeds": ground_prompt_input_embeds, "completion_ids": ground_completion_ids, "completion_mask": ground_completion_mask, "mask_seeds": mask_seeds, "old_per_token_logps": ground_old_text_per_token_logps, "ref_per_token_logps": None, "completion_ids_list": ground_completion_ids_list, "completion_lengths": ground_completion_lengths},
            {"prompts": image_prompts, "completions": image_completions, "completion_ids": image_completion_ids, "prompt_ids": image_prompt_ids, "prompt_mask": image_prompt_mask, "prompt_input_embeds": image_prompt_input_embeds, "is_gen": image_is_gen, "is_gen_enc": image_is_gen_enc, "completion_ids": image_completion_ids, "completion_mask": image_completion_mask, "mask_seeds": mask_seeds, "old_per_token_logps": old_image_per_token_logps, "ref_per_token_logps": None},
        ]
        for modality, reward_fnc, reward_weight, modality_dict in zip(modalities, reward_fncs, reward_weights_list, modality_dict_list):
            if completion_ids is not None:
                self.reward_funcs = list(reward_fnc.values())
                self.reward_func_names = list(reward_fnc.keys())
                keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
                assert f"{modality}_gt" in keys, f"Expected {modality}_gt to be in keys, got {keys} only"
                prompts = modality_dict_list["prompts"]
                completions = modality_dict_list["completions"]
                completion_ids = modality_dict_list["completion_ids"]
                rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids)
                # Apply weights to each reward function's output and sum
                rewards = (rewards_per_func * torch.tensor(reward_weight).to(device).unsqueeze(0)).nansum(dim=1)
                # Compute grouped-wise rewards
                mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
                std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
                is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

                # Normalize the rewards to compute the advantages
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                advantages = rewards - mean_grouped_rewards
                if self.scale_rewards:
                    advantages = advantages / (std_grouped_rewards + 1e-4)

                # Slice to keep only the local part of the data
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
                advantages = advantages[process_slice]
        
                # Log the metrics
                if mode == "train":
                    self.state.num_input_tokens_seen += self.accelerator.gather(modality_dict["prompt_mask"].sum()).sum().item()
                self._metrics[mode][f"{modality}_num_tokens"] = [self.state.num_input_tokens_seen]

                # Log completion lengths, mean, min, max
                agg_completion_lengths = self.accelerator.gather(modality_dict.get("completion_lengths", 0))
                self._metrics[mode][f"{modality}_completions/mean_length"].append(agg_completion_lengths.float().mean().item())
                self._metrics[mode][f"{modality}_completions/min_length"].append(agg_completion_lengths.float().min().item())
                self._metrics[mode][f"{modality}_completions/max_length"].append(agg_completion_lengths.float().max().item())

                # Identify sequences that terminated with EOS and log their lengths
                if modality in ["answer", "ground"]:
                    is_eos = modality_dict["completion_ids"] == self.processing_class.tokenizer.eos_token_id  # (bsz, completion_seq_len)
                    agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
                    term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
                    clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
                    self._metrics[mode][f"{modality}_completions/clipped_ratio"].append(clipped_completions_ratio)
                    if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
                        term_completion_lengths = torch.zeros(1, device=device)
                    self._metrics[mode][f"{modality}_completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
                    self._metrics[mode][f"{modality}_completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
                    self._metrics[mode][f"{modality}_completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

                # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
                for i, reward_func_name in enumerate(self.reward_func_names):
                    mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
                    std_rewards = nanstd(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
                self._metrics[mode][f"{modality}_reward"].append(mean_grouped_rewards.mean().item())
                self._metrics[mode][f"{modality}_reward_std"].append(std_grouped_rewards.mean().item())
                self._metrics[mode][f"{modality}_frac_reward_zero_std"].append(is_std_zero.float().mean().item())
                # self._logs[f"{modality}_prompt"].extend(gather_object(prompts))
                # self._logs[f"{modality}_completion"].extend(gather_object(completions))
                for i, name in enumerate(self.reward_func_names):
                    self._logs[f"{modality}_rewards"][name].extend(rewards_per_func[:, i].tolist())
                self._logs[f"{modality}_advantages"].extend(all_process_advantages.tolist())


                all_outputs[f"{modality}_prompt_ids"] = modality_dict["prompt_ids"] # bsz, prompt_seq_len
                all_outputs[f"{modality}_prompt_mask"] = modality_dict["prompt_mask"] # bsz, prompt_seq_len
                all_outputs[f"{modality}_completion_ids"] = modality_dict["completion_ids"] # bsz, completion_seq_len
                all_outputs[f"{modality}_completion_mask"] = modality_dict["completion_mask"] # bsz, completion_seq_len
                all_outputs[f"{modality}_prompt_input_embeds"] = modality_dict["prompt_input_embeds"] # bsz, prompt_seq_len, hidden_dim
                
                if modality in ["image"]:
                    all_outputs["is_gen"] = modality_dict["is_gen"] # bsz, seq_len_model
                    all_outputs["is_gen_enc"] = modality_dict["is_gen_enc"] # bsz, seq_len_model

                all_outputs[f"{modality}_advantages"] = advantages # bsz
                all_outputs[f"{modality}_mask_seeds"] = modality_dict["mask_seeds"] # num_iter
                if modality_dict["old_per_token_logps"] is not None:
                    all_outputs[f"{modality}_old_per_token_logps"] = modality_dict["old_per_token_logps"] # (num_iter, bsz, completion_seq_len)
                if modality_dict["ref_per_token_logps"] is not None:
                    all_outputs[f"{modality}_ref_per_token_logps"] = modality_dict["ref_per_token_logps"] # (num_iter, bsz, completion_seq_len)
                        
            else:
                continue
        return all_outputs
