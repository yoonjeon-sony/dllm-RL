import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional, Union

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
from trl.trainer.grpo_trainer import GRPOTrainer

from llava.mm_utils import pad_to_square_and_resize
from llava.model.language_model.llada.generate import get_logits, wte as llada_wte

from interleaved_inferencer import InterleavedInferencer
from reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correct_grounding_reward_func,
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
        self._buffered_inputs: Optional[list[Batch]] = None
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

        This is the single replacement for the old:
            - _pad_and_concat_2d
            - _pad_and_concat_3d
            - _pad_and_concat_prompt_embeds

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
    # Batch slicing / splitting
    # -------------------------------------------------------------------------

    def _slice_batch(self, batch: Batch, start: int, end: int) -> Batch:
        """
        Slice a packed batch on the row / batch dimension.

        batch["prompt_ids"]: (num_rows, prompt_seq_len)
        """
        batch_size = batch["prompt_ids"].size(0)
        sliced: Batch = {}
        metadata_keys = {"mask_seeds"}

        for key, value in batch.items():
            if key in metadata_keys:
                sliced[key] = value
                continue
            if torch.is_tensor(value) and value.dim() >= 1 and value.size(0) == batch_size:
                # value: (num_rows, ...)
                sliced[key] = value[start:end]
            elif torch.is_tensor(value) and value.dim() >= 2 and value.size(1) == batch_size:
                # value: (num_iter, num_rows, ...)
                sliced[key] = value[:, start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value

        return sliced

    def _split_generation_batch(self, batch: Batch) -> list[Batch]:
        """
        Split one packed generation batch into `steps_per_generation` chunks.

        Packed batch layout:
            - row-major tensors have first dimension == num_rows
            - cached old/ref log-probs have shape (num_iter, num_rows, seq_len)
            - image-only tensors have first dimension == num_image_rows
        """
        steps_per_generation = int(getattr(self.args, "steps_per_generation", 1) or 1)
        if steps_per_generation <= 1:
            return [self._offload_cached_logps(batch)]

        if "row_source_index" not in batch or "source_batch_size" not in batch:
            batch_size = batch["prompt_ids"].size(0)
            if batch_size % steps_per_generation != 0:
                raise ValueError(
                    f"Generated batch size ({batch_size}) must be divisible by "
                    f"steps_per_generation ({steps_per_generation})."
                )

            chunk_size = batch_size // steps_per_generation
            return [
                self._slice_batch(batch, start, start + chunk_size)
                for start in range(0, batch_size, chunk_size)
            ]

        source_batch_size = int(batch["source_batch_size"])
        if source_batch_size % steps_per_generation != 0:
            raise ValueError(
                f"Expanded source batch size ({source_batch_size}) must be divisible by "
                f"steps_per_generation ({steps_per_generation})."
            )

        chunk_source_size = source_batch_size // steps_per_generation
        flat_batch_size = batch["prompt_ids"].size(0)

        image_source_batch_size = None
        if "image_completion_ids" in batch and torch.is_tensor(batch["image_completion_ids"]):
            # image_completion_ids:
            #   non-unitok: (num_image_rows, image_token_len)
            #   unitok:     (num_image_rows, codebook_dim, image_token_len)
            image_source_batch_size = batch["image_completion_ids"].size(0)

        chunks: list[Batch] = []
        for start in range(0, source_batch_size, chunk_source_size):
            end = start + chunk_source_size
            row_mask = (batch["row_source_index"] >= start) & (batch["row_source_index"] < end)  # (num_rows,)

            chunk: Batch = {}
            metadata_keys = {"mask_seeds"}
            for key, value in batch.items():
                if key in metadata_keys:
                    chunk[key] = value
                    continue
                if torch.is_tensor(value):
                    if value.dim() >= 1 and value.size(0) == flat_batch_size:
                        # value: (num_rows, ...)
                        sliced = value[row_mask]
                        if key == "row_source_index":
                            sliced = sliced - start
                        chunk[key] = sliced
                    elif value.dim() >= 2 and value.size(1) == flat_batch_size:
                        # value: (num_iter, num_rows, ...)
                        chunk[key] = value[:, row_mask]
                    elif (
                        image_source_batch_size is not None
                        and value.dim() >= 1
                        and value.size(0) == image_source_batch_size
                    ):
                        # image-only tensor: (num_image_rows, ...)
                        chunk[key] = value[start:end]
                    else:
                        chunk[key] = value
                elif isinstance(value, list):
                    if len(value) == flat_batch_size:
                        indices = row_mask.nonzero(as_tuple=True)[0].tolist()
                        chunk[key] = [value[idx] for idx in indices]
                    elif len(value) == source_batch_size:
                        chunk[key] = value[start:end]
                    else:
                        chunk[key] = value
                else:
                    chunk[key] = value
            chunk = self._offload_cached_logps(chunk)
            chunk["source_batch_size"] = chunk_source_size
            chunks.append(chunk)

        return chunks

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Multimodal GRPO loss, aligned with the original GRPOTrainer clipping / importance weighting.

        inputs["completion_mask"]:     (num_rows, completion_seq_len)
        inputs["advantages"]:          (num_rows,)
        inputs["old_per_token_logps"]: (num_iter, num_rows, completion_seq_len) or None
        inputs["ref_per_token_logps"]: (num_iter, num_rows, completion_seq_len) or None
        """
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        if getattr(self, "use_liger_loss", False):
            raise NotImplementedError("Expanded modality batches are not supported with liger loss.")

        completion_mask = inputs["completion_mask"].to(torch.float32)  # (num_rows, completion_seq_len)

        # Current rollout-iteration index inside the buffered generation window.
        this_itr_idx = ((self._step - 1) // self.args.steps_per_generation) % self.args.num_iterations
        mask_seed = inputs["mask_seeds"][this_itr_idx]

        policy_model = self.accelerator.unwrap_model(model)
        policy_core_model = policy_model.get_model() if hasattr(policy_model, "get_model") else policy_model

        # per_token_logps: (num_rows, completion_seq_len)
        per_token_logps = self._get_current_flat_per_token_logps(policy_core_model, inputs, mask_seed)

        if self.args.beta != 0.0:
            ref_per_token_logps = (
                inputs["ref_per_token_logps"][this_itr_idx]
                .to(device=per_token_logps.device, dtype=per_token_logps.dtype, non_blocking=True)
                .detach()
            )  # (num_rows, completion_seq_len)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
            )
        else:
            per_token_kl = None

        cached_old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if cached_old_per_token_logps is None
            else (
                cached_old_per_token_logps[this_itr_idx]
                .to(device=per_token_logps.device, dtype=per_token_logps.dtype, non_blocking=True)
                .detach()
            )
        )  # (num_rows, completion_seq_len)

        advantages = inputs["advantages"]  # (num_rows,)

        log_ratio = per_token_logps - old_per_token_logps  # (num_rows, completion_seq_len)
        del per_token_logps
        del old_per_token_logps
        if self.args.beta != 0.0:
            del ref_per_token_logps

        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio  # (num_rows, completion_seq_len)
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (
                (log_ratio * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)  # (num_rows, 1)
        else:
            raise ValueError(f"Unknown importance_sampling_level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.args.epsilon_low,
            1 + self.args.epsilon_high,
        )

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)  # (num_rows, completion_seq_len) or (num_rows, 1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)  # (num_rows, completion_seq_len) or (num_rows, 1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl

        if self.args.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.args.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.args.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.args.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.args.loss_type}")

        if not getattr(self, "_skip_loss_metrics", False):
            mode = "train" if self.model.training else "eval"
            completion_token_count = completion_mask.sum().clamp(min=1.0)

            def masked_mean(x: torch.Tensor) -> torch.Tensor:
                if x.shape[1] == 1:
                    return x.mean()
                return (x * completion_mask).sum() / completion_token_count

            if per_token_kl is not None:
                mean_kl = masked_mean(per_token_kl)
                self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

            is_low_clipped = (coef_1 < 1 - self.args.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.args.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_mean(is_low_clipped.float())
            high_clip = masked_mean(is_high_clipped.float())
            region_clip = masked_mean(is_region_clipped.float())

            self._metrics[mode]["clip_ratio/low"].append(
                self.accelerator.gather(low_clip).nanmean().item()
            )
            self._metrics[mode]["clip_ratio/high"].append(
                self.accelerator.gather(high_clip).nanmean().item()
            )
            self._metrics[mode]["clip_ratio/region"].append(
                self.accelerator.gather(region_clip).nanmean().item()
            )

        return loss

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
            _clear_device_cache()

        kwargs = {}
        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
                _clear_device_cache()

            self.accelerator.backward(loss, **kwargs)

        output = loss.detach()

        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before

        if self._step % self.args.gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0

        return output

    # -------------------------------------------------------------------------
    # Masking / completion masks
    # -------------------------------------------------------------------------

    def forward_process(
        self,
        batch: torch.Tensor,
        prompt_index: torch.Tensor,
        mask_id: int,
        seed: Optional[Union[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupt prompt / completion tokens for diffusion log-prob evaluation.

        batch:        (bsz, seq_len)
        prompt_index: (bsz, seq_len) bool; True only on prompt positions

        returns:
            noisy_batch: (bsz, seq_len)
            p_mask:      (bsz, seq_len)
        """
        if isinstance(seed, torch.Tensor):
            seed = seed.item()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=batch.device)
            generator.manual_seed(int(seed))

        bsz, seq_len = batch.shape
        t_p = torch.ones(bsz, device=batch.device) * self.args.p_mask_prompt  # (bsz,)
        random_matrix = torch.rand((bsz, seq_len), device=batch.device, generator=generator)  # (bsz, seq_len)

        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))  # (bsz, seq_len)
        is_mask_completion = ~prompt_index  # (bsz, seq_len)
        is_mask = is_mask_prompt | is_mask_completion  # (bsz, seq_len)

        noisy_batch = torch.where(is_mask, mask_id, batch)  # (bsz, seq_len)
        p_mask = torch.where(prompt_index, t_p.unsqueeze(1), torch.ones_like(t_p).unsqueeze(1))  # (bsz, seq_len)

        return noisy_batch, p_mask

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

        return (sequence_indices <= eos_idx.unsqueeze(1)).int()  # (bsz, completion_seq_len)

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

    def _flatten_image_completion_ids(self, image_completion_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        image_completion_ids:
            non-unitok: (bsz_image, image_token_len)
            unitok:     (bsz_image, codebook_dim, image_token_len)

        returns:
            flat_image_completion_ids:
                non-unitok: (bsz_image, image_token_len)
                unitok:     (bsz_image, codebook_dim * image_token_len)
        """
        if image_completion_ids is None:
            return None

        if image_completion_ids.dim() == 2:
            return image_completion_ids

        if image_completion_ids.dim() == 3:
            return image_completion_ids.reshape(image_completion_ids.size(0), -1)

        raise ValueError(f"Unsupported image completion shape: {tuple(image_completion_ids.shape)}")

    def _masked_ids_to_list(self, token_ids: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """
        token_ids: (bsz, seq_len)
        mask:      (bsz, seq_len)
        """
        return [
            [token.item() for token, keep in zip(row, mask_row) if int(keep) != 0]
            for row, mask_row in zip(token_ids, mask)
        ]

    # -------------------------------------------------------------------------
    # Reward normalization
    # -------------------------------------------------------------------------

    def _normalize_rewards(
        self, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        rewards: (global_rollout_bsz,) where rows are ordered in contiguous groups of size self.num_generations

        returns:
            advantages:            (global_rollout_bsz,)
            mean_grouped_rewards:  (global_rollout_bsz,)
            std_grouped_rewards:   (global_rollout_bsz,)
            is_std_zero:           (num_groups,)
        """
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  # (num_groups,)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)  # (num_groups,)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # (global_rollout_bsz,)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # (global_rollout_bsz,)

        advantages = rewards - mean_grouped_rewards  # (global_rollout_bsz,)
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        return advantages, mean_grouped_rewards, std_grouped_rewards, is_std_zero

    def _slice_global_tensor_for_current_rank(self, global_tensor: torch.Tensor, local_count: int) -> torch.Tensor:
        local_count_tensor = torch.tensor([local_count], device=global_tensor.device, dtype=torch.long)
        all_counts = self.accelerator.gather(local_count_tensor).cpu().tolist()
        start = sum(all_counts[: self.accelerator.process_index])
        end = start + all_counts[self.accelerator.process_index]
        return global_tensor[start:end]

    def _repeat_list_to_length(self, values: Optional[list[Any]], target_len: int) -> Optional[list[Any]]:
        """
        Repeat a per-source-example python list so it matches rollout row count.

        Example:
            len(values)=B, target_len=B*G -> repeat each item G times.
        """
        if values is None:
            return None
        if len(values) == target_len:
            return values
        if len(values) == 0:
            return values
        if target_len % len(values) != 0:
            raise ValueError(
                f"Cannot repeat list of length {len(values)} to target length {target_len}."
            )
        repeat_factor = target_len // len(values)
        return [item for item in values for _ in range(repeat_factor)]

    # -------------------------------------------------------------------------
    # Packing / assertions
    # -------------------------------------------------------------------------

    def _pack_modality_rows(self, modality_batches: list[Batch]) -> Batch:
        """
        Pack modality-specific batches into one row-major batch.

        Text modality input tensors:
            prompt_ids:          (bsz_mod, prompt_seq_len)
            prompt_mask:         (bsz_mod, prompt_seq_len)
            prompt_input_embeds: (bsz_mod, prompt_seq_len, hidden_dim)
            completion_ids:      (bsz_mod, completion_seq_len)
            completion_mask:     (bsz_mod, completion_seq_len)

        Image modality input tensors:
            prompt_ids:          (bsz_mod, prompt_seq_len)
            prompt_mask:         (bsz_mod, prompt_seq_len)
            prompt_input_embeds: (bsz_mod, prompt_context_seq_len, hidden_dim)
            completion_ids:      (bsz_mod, flat_image_completion_len)
            completion_mask:     (bsz_mod, flat_image_completion_len)

        Packed outputs:
            prompt_ids:          (num_rows_total, max_prompt_seq_len)
            prompt_mask:         (num_rows_total, max_prompt_seq_len)
            prompt_input_embeds: (num_rows_total, max_prompt_context_seq_len, hidden_dim)
            completion_ids:      (num_rows_total, max_completion_seq_len)
            completion_mask:     (num_rows_total, max_completion_seq_len)
        """
        ordered = [batch for batch in modality_batches if batch is not None]
        if not ordered:
            raise ValueError("At least one modality batch is required.")

        pad_token_id = self.processing_class.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        prompt_ids = self._pad_and_concat(
            [batch["prompt_ids"] for batch in ordered],  # each: (bsz_i, prompt_seq_len_i)
            pad_dim=1,
            cat_dim=0,
            pad_value=pad_token_id,
            padding_side="left",
            dtype=torch.long,
        )  # -> (num_rows_total, max_prompt_seq_len)

        prompt_mask = self._pad_and_concat(
            [batch["prompt_mask"] for batch in ordered],  # each: (bsz_i, prompt_seq_len_i)
            pad_dim=1,
            cat_dim=0,
            pad_value=0,
            padding_side="left",
            dtype=torch.long,
        )  # -> (num_rows_total, max_prompt_seq_len)

        prompt_input_embeds = self._pad_and_concat(
            [batch["prompt_input_embeds"] for batch in ordered],  # each: (bsz_i, prompt_context_seq_len_i, hidden_dim)
            pad_dim=1,
            cat_dim=0,
            pad_value=0.0,
            padding_side="left",
            dtype=None,
        )  # -> (num_rows_total, max_prompt_context_seq_len, hidden_dim)

        completion_ids = self._pad_and_concat(
            [batch["completion_ids"] for batch in ordered],  # each: (bsz_i, completion_seq_len_i)
            pad_dim=1,
            cat_dim=0,
            pad_value=self.processing_class.tokenizer.eos_token_id,
            padding_side="right",
            dtype=torch.long,
        )  # -> (num_rows_total, max_completion_seq_len)

        completion_mask = self._pad_and_concat(
            [batch["completion_mask"] for batch in ordered],  # each: (bsz_i, completion_seq_len_i)
            pad_dim=1,
            cat_dim=0,
            pad_value=0,
            padding_side="right",
            dtype=torch.long,
        )  # -> (num_rows_total, max_completion_seq_len)

        advantages = torch.cat(
            [batch["advantages"] for batch in ordered], dim=0
        ).to(torch.float32)  # -> (num_rows_total,)

        row_modality = torch.cat(
            [
                torch.full(
                    (batch["prompt_ids"].size(0),),
                    batch["modality_id"],
                    dtype=torch.long,
                    device=batch["prompt_ids"].device,
                )
                for batch in ordered
            ],
            dim=0,
        )  # -> (num_rows_total,)

        row_source_index = torch.cat(
            [
                torch.arange(
                    batch["source_batch_size"],
                    device=batch["prompt_ids"].device,
                    dtype=torch.long,
                )
                for batch in ordered
            ],
            dim=0,
        )  # -> (num_rows_total,)

        row_completion_width = torch.cat(
            [
                torch.full(
                    (batch["completion_ids"].size(0),),
                    batch["completion_width"],
                    dtype=torch.long,
                    device=batch["completion_ids"].device,
                )
                for batch in ordered
            ],
            dim=0,
        )  # -> (num_rows_total,)

        output: Batch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "prompt_input_embeds": prompt_input_embeds,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "row_modality": row_modality,
            "row_source_index": row_source_index,
            "row_completion_width": row_completion_width,
            "source_batch_size": ordered[0]["source_batch_size"],
        }

        old_chunks = [
            batch["old_per_token_logps"]
            for batch in ordered
            if batch.get("old_per_token_logps") is not None
        ]
        if len(old_chunks) == len(ordered):
            output["old_per_token_logps"] = self._pad_and_concat(
                old_chunks,  # each: (num_iter, bsz_i, completion_seq_len_i)
                pad_dim=-1,
                cat_dim=1,
                pad_value=0.0,
                padding_side="right",
                dtype=torch.float32,
            )  # -> (num_iter, num_rows_total, max_completion_seq_len)

        ref_chunks = [
            batch["ref_per_token_logps"]
            for batch in ordered
            if batch.get("ref_per_token_logps") is not None
        ]
        if len(ref_chunks) == len(ordered):
            output["ref_per_token_logps"] = self._pad_and_concat(
                ref_chunks,  # each: (num_iter, bsz_i, completion_seq_len_i)
                pad_dim=-1,
                cat_dim=1,
                pad_value=0.0,
                padding_side="right",
                dtype=torch.float32,
            )  # -> (num_iter, num_rows_total, max_completion_seq_len)

        image_batch = next((batch for batch in ordered if batch["modality_id"] == self.MODALITY_IMAGE), None)
        if image_batch is not None:
            output["image_completion_ids"] = image_batch["image_completion_ids"]
            output["image_input_embeds_gen"] = image_batch["image_input_embeds_gen"]
            output["image_is_gen"] = image_batch["image_is_gen"]
            output["image_is_gen_enc"] = image_batch["image_is_gen_enc"]
            output["image_gen_shape"] = image_batch["image_gen_shape"]

        self._assert_packed_batch_shapes(output)
        return output

    def _assert_packed_batch_shapes(self, batch: Batch) -> None:
        """
        Validate row-aligned packed tensors.
        """
        num_rows = batch["prompt_ids"].size(0)

        if batch["prompt_mask"].size(0) != num_rows:
            raise ValueError("prompt_mask batch dimension does not match prompt_ids.")
        if batch["prompt_input_embeds"].size(0) != num_rows:
            raise ValueError("prompt_input_embeds batch dimension does not match prompt_ids.")
        if batch["completion_ids"].size(0) != num_rows:
            raise ValueError("completion_ids batch dimension does not match prompt_ids.")
        if batch["completion_mask"].size(0) != num_rows:
            raise ValueError("completion_mask batch dimension does not match prompt_ids.")
        if batch["row_modality"].size(0) != num_rows:
            raise ValueError("row_modality batch dimension does not match prompt_ids.")
        if batch["row_completion_width"].size(0) != num_rows:
            raise ValueError("row_completion_width batch dimension does not match prompt_ids.")

        image_rows = batch["row_modality"] == self.MODALITY_IMAGE  # (num_rows,)
        if image_rows.any() and "image_completion_ids" in batch:
            num_image_rows = int(image_rows.sum().item())

            if batch["image_completion_ids"].size(0) != num_image_rows:
                raise ValueError("image_completion_ids batch dimension does not match number of image rows.")
            if batch["image_input_embeds_gen"].size(0) != num_image_rows:
                raise ValueError("image_input_embeds_gen batch dimension does not match number of image rows.")
            if batch["image_is_gen"].size(0) != num_image_rows:
                raise ValueError("image_is_gen batch dimension does not match number of image rows.")
            if batch["image_is_gen_enc"].size(0) != num_image_rows:
                raise ValueError("image_is_gen_enc batch dimension does not match number of image rows.")

            flat_image_completion_ids = self._flatten_image_completion_ids(batch["image_completion_ids"])
            expected_image_width = flat_image_completion_ids.size(1)

            image_widths = batch["row_completion_width"][image_rows].unique()
            if image_widths.numel() != 1 or int(image_widths.item()) != expected_image_width:
                raise ValueError(
                    f"Packed image width mismatch: row_completion_width={image_widths.tolist()}, "
                    f"flat_image_width={expected_image_width}."
                )

            if batch["completion_mask"][image_rows, expected_image_width:].sum().item() != 0:
                raise ValueError("Image completion mask has non-zero padded positions beyond flat image width.")

    # -------------------------------------------------------------------------
    # DeepSpeed helper
    # -------------------------------------------------------------------------

    @contextmanager
    def _gather_deepspeed_zero3_params(self, model):
        if (
            not self.is_deepspeed_enabled
            or self.accelerator.state.deepspeed_plugin is None
            or self.accelerator.state.deepspeed_plugin.zero_stage != 3
        ):
            yield
            return

        import deepspeed

        with deepspeed.zero.GatheredParameters(model.parameters()):
            yield

    # -------------------------------------------------------------------------
    # Log-prob computation
    # -------------------------------------------------------------------------

    def _get_current_flat_per_token_logps(
        self,
        model,
        inputs: Batch,
        mask_seed: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute current policy per-token log-probs for the packed multimodal batch.

        inputs["prompt_ids"]:          (num_rows, prompt_seq_len)
        inputs["prompt_mask"]:         (num_rows, prompt_seq_len)
        inputs["prompt_input_embeds"]: (num_rows, prompt_context_seq_len, hidden_dim)
        inputs["completion_ids"]:      (num_rows, completion_seq_len)
        inputs["completion_mask"]:     (num_rows, completion_seq_len)
        inputs["row_modality"]:        (num_rows,)
        inputs["row_completion_width"]:(num_rows,)

        returns:
            per_token_logps:           (num_rows, completion_seq_len)
        """
        completion_ids = inputs["completion_ids"]  # (num_rows, completion_seq_len)
        completion_mask = inputs["completion_mask"]  # (num_rows, completion_seq_len)
        prompt_ids = inputs["prompt_ids"]  # (num_rows, prompt_seq_len)
        prompt_input_embeds = inputs["prompt_input_embeds"]  # (num_rows, prompt_context_seq_len, hidden_dim)
        prompt_mask = inputs["prompt_mask"]  # (num_rows, prompt_seq_len)
        row_modality = inputs["row_modality"]  # (num_rows,)
        row_completion_width = inputs["row_completion_width"]  # (num_rows,)

        per_token_logps = torch.zeros(
            completion_ids.size(0),
            completion_ids.size(1),
            dtype=torch.float32,
            device=completion_ids.device,
        )  # (num_rows, completion_seq_len)

        for modality_id in (self.MODALITY_GROUND, self.MODALITY_ANSWER):
            text_mask = row_modality == modality_id  # (num_rows,)
            if not text_mask.any():
                continue

            completion_widths = row_completion_width[text_mask].unique()
            if completion_widths.numel() != 1:
                raise ValueError(
                    f"Expected uniform completion widths within modality {modality_id}, got "
                    f"{completion_widths.tolist()}."
                )

            completion_width = int(completion_widths.item())

            text_logps = self._get_padded_text_per_token_logps(
                model=model,
                prompt_ids=prompt_ids[text_mask],  # (bsz_text, prompt_seq_len_text)
                prompt_input_embeds=prompt_input_embeds[text_mask],  # (bsz_text, prompt_context_seq_len_text, hidden_dim)
                prompt_mask=prompt_mask[text_mask],  # (bsz_text, prompt_seq_len_text)
                completion_ids=completion_ids[text_mask][:, :completion_width],  # (bsz_text, completion_width)
                mask_seeds=[mask_seed],
            )[0]  # -> (bsz_text, completion_width)

            per_token_logps[text_mask, :completion_width] = text_logps

        image_mask = row_modality == self.MODALITY_IMAGE  # (num_rows,)
        if image_mask.any():
            image_logps = self._get_image_per_token_logps(
                model=model,
                image_completion_ids=inputs["image_completion_ids"],
                image_input_embeds_gen=inputs["image_input_embeds_gen"],
                image_is_gen=inputs["image_is_gen"],
                image_is_gen_enc=inputs["image_is_gen_enc"],
                image_gen_shape=inputs.get("image_gen_shape"),
                mask_seeds=[mask_seed],
            )[0]  # -> (bsz_image, flat_image_completion_len)

            image_width = image_logps.size(1)
            per_token_logps[image_mask, :image_width] = image_logps

            if completion_mask[image_mask, image_width:].sum().item() != 0:
                raise ValueError(
                    "Image completion mask has non-zero padded positions beyond computed image log-probs."
                )

        return per_token_logps

    def _get_padded_text_per_token_logps(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_input_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        mask_seeds: list[Union[int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Group text rows by true prompt length so left padding does not contaminate scoring.

        prompt_ids:          (bsz_text, prompt_seq_len_padded)
        prompt_input_embeds: (bsz_text, prompt_context_seq_len_padded, hidden_dim)
        prompt_mask:         (bsz_text, prompt_seq_len_padded)
        completion_ids:      (bsz_text, completion_seq_len)

        returns:
            per_token_logps: (num_iter, bsz_text, completion_seq_len)
        """
        batch_size = prompt_ids.size(0)
        completion_width = completion_ids.size(1)
        prompt_lengths = prompt_mask.sum(dim=1).to(torch.long)  # (bsz_text,)

        per_token_logps = torch.zeros(
            len(mask_seeds),
            batch_size,
            completion_width,
            dtype=torch.float32,
            device=prompt_ids.device,
        )  # (num_iter, bsz_text, completion_seq_len)

        for prompt_length in prompt_lengths.unique(sorted=True).tolist():
            row_mask = prompt_lengths == prompt_length  # (bsz_text,)

            compact_prompt_ids = (
                prompt_ids[row_mask][:, -prompt_length:]
                if prompt_length > 0
                else prompt_ids[row_mask][:, :0]
            )  # -> (bsz_group, prompt_length)

            compact_prompt_input_embeds = (
                prompt_input_embeds[row_mask][:, -prompt_length:]
                if prompt_length > 0
                else prompt_input_embeds[row_mask][:, :0]
            )  # -> (bsz_group, prompt_length, hidden_dim)

            compact_input_ids = torch.cat(
                [compact_prompt_ids, completion_ids[row_mask]], dim=1
            )  # -> (bsz_group, prompt_length + completion_seq_len)

            group_logps = self._get_text_per_token_logps(
                model=model,
                input_ids=compact_input_ids,
                prompt_input_embeds=compact_prompt_input_embeds,
                logits_to_keep=completion_width,
                mask_seeds=mask_seeds,
            )  # -> (num_iter, bsz_group, completion_seq_len)

            per_token_logps[:, row_mask, :] = group_logps

        return per_token_logps

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

        input_ids:           (bsz, full_seq_len) = [prompt tokens | completion tokens]
        prompt_input_embeds: (bsz, prompt_seq_len, hidden_dim)
        logits_to_keep:      scalar = completion_seq_len

        returns:
            per_token_logps: (num_iter, bsz, completion_seq_len)
        """
        num_iterations = len(mask_seeds)
        batch_size = input_ids.size(0)
        prompt_length = prompt_input_embeds.size(1)

        per_token_logps = torch.zeros(
            num_iterations,
            batch_size,
            logits_to_keep,
            dtype=torch.float32,
            device=input_ids.device,
        )  # (num_iter, bsz, completion_seq_len)

        core_model = model.get_model() if hasattr(model, "get_model") else model
        checkpoint_strategy = getattr(core_model, "activation_checkpointing_strategy", None)
        disable_checkpointing = torch.is_grad_enabled() and hasattr(core_model, "set_activation_checkpointing")

        try:
            if disable_checkpointing:
                core_model.set_activation_checkpointing(None)

                with self._gather_deepspeed_zero3_params(core_model):
                    past_key_values = None
                    if prompt_length > 0:
                        # Match native prefix_lm path: cache prompt context first, then score completion only.
                        with torch.no_grad():
                            past_key_values = core_model(
                                None,
                                input_embeddings=prompt_input_embeds,
                                use_cache=True,
                                modality_indices=None,
                            ).attn_key_values

                    for iter_idx, mask_seed in enumerate(mask_seeds):
                        x = input_ids.clone()  # (bsz, full_seq_len)

                        # prompt_index marks only prompt positions.
                        prompt_index = torch.zeros_like(x, dtype=torch.bool)  # (bsz, full_seq_len)
                        if prompt_length > 0:
                            prompt_index[:, :prompt_length] = True

                        perturbed_seq, _ = self.forward_process(
                            batch=x,
                            prompt_index=prompt_index,
                            mask_id=self.args.mask_id,
                            seed=mask_seed,
                        )  # -> (bsz, full_seq_len)

                        completion_seq = perturbed_seq[:, prompt_length:]  # (bsz, completion_seq_len)

                        inputs_embeds_curr, new_token_mask = llada_wte(
                            core_model,
                            completion_seq,
                            t2i_inference=False,
                        )
                        # inputs_embeds_curr: (bsz, completion_seq_len, hidden_dim)
                        # new_token_mask:     (bsz, completion_seq_len) or model-specific routing mask

                        logits = get_logits(
                            core_model,
                            inputs_embeds_curr,
                            new_token_mask,
                            t2i_inference=False,
                            past_key_values=past_key_values,
                        )  # (bsz, completion_seq_len, vocab_size)

                        completion_logits = logits[:, -logits_to_keep:, :]  # (bsz, completion_seq_len, vocab_size)
                        completion_targets = x[:, -logits_to_keep:]  # (bsz, completion_seq_len)

                        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))  # (bsz * completion_seq_len, vocab_size)
                        flat_targets = completion_targets.reshape(-1)  # (bsz * completion_seq_len,)

                        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                        logps = -loss.view(batch_size, logits_to_keep)  # (bsz, completion_seq_len)

                        if not torch.isfinite(logps).all():
                            raise ValueError("NaN/Inf detected in text per-token log-probs.")

                        per_token_logps[iter_idx] = logps.to(torch.float32)

        finally:
            if disable_checkpointing:
                core_model.set_activation_checkpointing(checkpoint_strategy)

        return per_token_logps

    def _get_image_per_token_logps(
        self,
        model,
        image_completion_ids: torch.Tensor,
        image_input_embeds_gen: torch.Tensor,
        image_is_gen: torch.Tensor,
        image_is_gen_enc: Optional[torch.Tensor],
        image_gen_shape: Optional[tuple[int, int]],
        mask_seeds: list[Union[int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute image token log-probs following the model's native generate_image-style forward path.

        image_completion_ids:
            non-unitok: (bsz_image, image_token_len)
            unitok:     (bsz_image, codebook_dim, image_token_len)

        image_input_embeds_gen:
            (bsz_image, seq_len_model, hidden_dim)

        image_is_gen:
            (bsz_image, seq_len_model)

        image_is_gen_enc:
            (bsz_image, seq_len_model)

        returns:
            per_token_logps:
                non-unitok: (num_iter, bsz_image, image_token_len)
                unitok:     (num_iter, bsz_image, codebook_dim * image_token_len)
        """
        if image_completion_ids is None:
            return None

        core_model = model.get_model() if hasattr(model, "get_model") else model
        img_mask_id = 8193

        image_is_gen = image_is_gen.to(device=image_input_embeds_gen.device, dtype=torch.bool)
        image_is_gen_enc = (
            image_is_gen_enc.to(device=image_input_embeds_gen.device, dtype=torch.bool)
            if image_is_gen_enc is not None
            else torch.zeros_like(image_is_gen)
        )

        if image_gen_shape is None:
            image_gen_shape = self._infer_image_gen_shape(image_completion_ids)

        flat_targets = self._flatten_image_completion_ids(image_completion_ids)  # (bsz_image, flat_image_completion_len)
        batch_size = flat_targets.size(0)
        flat_seq_len = flat_targets.size(1)

        per_token_logps: list[torch.Tensor] = []

        checkpoint_strategy = getattr(core_model, "activation_checkpointing_strategy", None)
        disable_checkpointing = torch.is_grad_enabled() and hasattr(core_model, "set_activation_checkpointing")

        try:
            if disable_checkpointing:
                core_model.set_activation_checkpointing(None)

                with self._gather_deepspeed_zero3_params(core_model):
                    enc_use_image_branch = getattr(core_model.config, "enc_use_image_branch", False)

                    for _ in mask_seeds:
                        xt = torch.full_like(image_completion_ids, img_mask_id)
                        # xt:
                        #   non-unitok: (bsz_image, image_token_len)
                        #   unitok:     (bsz_image, codebook_dim, image_token_len)

                        inputs_embeds_curr = image_input_embeds_gen.clone()  # (bsz_image, seq_len_model, hidden_dim)

                        all_input_embeddings, new_token_mask = llada_wte(
                            core_model,
                            None,
                            True,
                            x_gen=xt,
                            gen_shape=image_gen_shape,
                            inputs_embeds_curr=inputs_embeds_curr,
                            new_token_mask=image_is_gen,
                        )
                        # all_input_embeddings: (bsz_image, seq_len_model, hidden_dim)
                        # new_token_mask:       (bsz_image, seq_len_model)

                        if not torch.isfinite(all_input_embeddings).all():
                            raise ValueError("NaN/Inf detected in image input embeddings.")

                        if enc_use_image_branch:
                            modality_indices = image_is_gen | image_is_gen_enc
                        else:
                            modality_indices = image_is_gen
                        # modality_indices: (bsz_image, seq_len_model)

                        n_mask = image_is_gen.sum(dim=1)  # (bsz_image,)
                        timesteps = n_mask.float() / image_is_gen.shape[1]  # (bsz_image,)

                        logits = get_logits(
                            core_model,
                            all_input_embeddings,
                            new_token_mask,
                            True,
                            gen_shape=image_gen_shape,
                            input_modality_indices=modality_indices,
                            timesteps=timesteps,
                        )

                        if image_completion_ids.dim() == 3:
                            logits = logits.permute(0, 2, 1, 3).contiguous()
                            # logits: (bsz_image, codebook_dim, image_token_len, vocab_size)

                            flat_logits = logits.view(-1, logits.size(-1))  # (bsz_image * codebook_dim * image_token_len, vocab_size)
                            flat_targets_iter = image_completion_ids.reshape(-1)  # (bsz_image * codebook_dim * image_token_len,)
                        else:
                            # logits: (bsz_image, image_token_len, vocab_size)
                            flat_logits = logits.reshape(-1, logits.size(-1))  # (bsz_image * image_token_len, vocab_size)
                            flat_targets_iter = flat_targets.reshape(-1)  # (bsz_image * image_token_len,)

                        loss = F.cross_entropy(flat_logits.float(), flat_targets_iter, reduction="none")
                        logps = -loss.view(batch_size, -1)  # (bsz_image, flat_image_completion_len)

                        if not torch.isfinite(logps).all():
                            raise ValueError("NaN/Inf detected in image per-token log-probs.")
                        if logps.size(1) != flat_seq_len:
                            raise ValueError(
                                f"Expected image log-prob width {flat_seq_len}, got {logps.size(1)}."
                            )

                        per_token_logps.append(logps)

        finally:
            if disable_checkpointing:
                core_model.set_activation_checkpointing(checkpoint_strategy)

        return torch.stack(per_token_logps, dim=0).to(torch.float32)  # (num_iter, bsz_image, flat_image_completion_len)

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------

    def _calculate_rewards(
        self,
        inputs: list[dict[str, Any]],
        answer_prompts: list[str],
        answer_completions: list[str],
        ground_prompts: Optional[list[str]] = None,
        ground_completions: Optional[list[str]] = None,
        image_prompts: Optional[list[str]] = None,
        image_completion_ids_list: Optional[list[list[int]]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute gathered reward tensors for each modality.

        Assumption, same as original GRPOTrainer:
            after gather(), rollout rows remain contiguous in groups of size self.num_generations.
        """
        device = self.accelerator.device

        base_reward_kwargs = {
            key: [example[key] for example in inputs]
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        }

        answer_gts = [example.get("answer_gt") for example in inputs]
        bbox_gts = [example.get("grounding_gt") for example in inputs]

        def to_reward_tensor(values) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=device)

        def reward_kwargs_for(target_len: int) -> dict[str, Any]:
            out = {
                key: self._repeat_list_to_length(values, target_len)
                for key, values in base_reward_kwargs.items()
            }
            out["trainer_state"] = self.state
            return out

        # Answer branch
        answer_target_len = len(answer_completions)
        answer_prompts_expanded = self._repeat_list_to_length(answer_prompts, answer_target_len)
        answer_gts_expanded = self._repeat_list_to_length(answer_gts, answer_target_len)

        if answer_target_len > 0:
            answer_correctness_scores = to_reward_tensor(
                correctness_reward_func(
                    prompts=answer_prompts_expanded,
                    completions=answer_completions,
                    answer=answer_gts_expanded,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs_for(answer_target_len),
                )
            )  # (local_answer_rollout_bsz,)

            answer_format_scores = to_reward_tensor(
                boxed_and_answer_tags_format_reward(
                    prompts=answer_prompts_expanded,
                    completions=answer_completions,
                    answer=answer_gts_expanded,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs_for(answer_target_len),
                )
            )  # (local_answer_rollout_bsz,)
        else:
            answer_correctness_scores = torch.zeros(0, dtype=torch.float32, device=device)
            answer_format_scores = torch.zeros(0, dtype=torch.float32, device=device)

        # Ground branch
        if ground_prompts is not None and ground_completions is not None:
            ground_target_len = len(ground_completions)
            ground_prompts_expanded = self._repeat_list_to_length(ground_prompts, ground_target_len)
            bbox_gts_expanded = self._repeat_list_to_length(bbox_gts, ground_target_len)

            grounding_iou_scores = to_reward_tensor(
                correct_grounding_reward_func(
                    prompts=ground_prompts_expanded,
                    completions=ground_completions,
                    answer=bbox_gts_expanded,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs_for(ground_target_len),
                )
            )  # (local_ground_rollout_bsz,)
        else:
            grounding_iou_scores = torch.zeros(0, dtype=torch.float32, device=device)

        rewards = {
            "answer_correctness": gather(answer_correctness_scores),  # (global_answer_rollout_bsz,)
            "answer_format": gather(answer_format_scores),  # (global_answer_rollout_bsz,)
            "grounding_iou": gather(grounding_iou_scores),  # (global_ground_rollout_bsz,)
        }

        rewards["ground_reward"] = rewards["grounding_iou"]
        rewards["answer_reward"] = (
            2.0 * rewards["answer_correctness"]
            + rewards["answer_format"]
            )

        if image_prompts is not None or image_completion_ids_list is not None:
            # Current design: image reward reuses answer correctness.
            rewards["image_reward"] = rewards["answer_correctness"]

        return rewards

    # -------------------------------------------------------------------------
    # Input preparation
    # -------------------------------------------------------------------------

    def _prepare_inputs(self, generation_batch: Batch) -> Batch:
        mode = "train" if self.model.training else "eval"

        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or not self._buffered_inputs:
                expanded_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = self._split_generation_batch(expanded_batch)

            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
            return inputs

        return self._generate_and_score_completions(generation_batch)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_flat_batch_metrics(
        self,
        mode: str,
        packed_output: Batch,
        modality_metric_payloads: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """
        packed_output["prompt_mask"]:     (num_rows, prompt_seq_len)
        packed_output["completion_mask"]: (num_rows, completion_seq_len)
        """
        attention_mask = torch.cat(
            [packed_output["prompt_mask"], packed_output["completion_mask"]],
            dim=1,
        )  # (num_rows, prompt_seq_len + completion_seq_len)

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()

        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["rows/total"].append(packed_output["prompt_ids"].size(0))

        # ------------------------------------------------------------------
        # Scalar metrics
        # ------------------------------------------------------------------
        for modality_name, payload in modality_metric_payloads.items():
            completion_lengths = payload["completion_mask"].sum(1)  # local modality rows
            agg_completion_lengths = self.accelerator.gather(completion_lengths)  # global modality rows

            if agg_completion_lengths.numel() > 0:
                self._metrics[mode][f"completions/{modality_name}/mean_length"].append(
                    agg_completion_lengths.float().mean().item()
                )
                self._metrics[mode][f"completions/{modality_name}/min_length"].append(
                    agg_completion_lengths.float().min().item()
                )
                self._metrics[mode][f"completions/{modality_name}/max_length"].append(
                    agg_completion_lengths.float().max().item()
                )
                self._metrics[mode][f"rows/{modality_name}"].append(float(agg_completion_lengths.numel()))

            rewards_global = payload.get("global_reward")
            if rewards_global is not None and rewards_global.numel() > 0:
                self._metrics[mode][f"rewards/{modality_name}/mean"].append(
                    rewards_global.float().mean().item()
                )
                self._metrics[mode][f"rewards/{modality_name}/std"].append(
                    rewards_global.float().std().item()
                )

        # ------------------------------------------------------------------
        # Gather row-aligned logs for this batch
        # ------------------------------------------------------------------
        gathered_prompts = gather_object(packed_output.get("prompt_logs", []))
        gathered_completions = gather_object(packed_output.get("completion_logs", []))
        gathered_advantages = gather_object(packed_output["advantages"].detach().cpu().tolist())
        gathered_row_modality = gather_object(packed_output["row_modality"].detach().cpu().tolist())

        rows_in_batch = len(gathered_row_modality)

        if len(gathered_prompts) != rows_in_batch:
            raise ValueError(
                f"Prompt log length mismatch: got {len(gathered_prompts)} prompts for {rows_in_batch} gathered rows."
            )
        if len(gathered_completions) != rows_in_batch:
            raise ValueError(
                f"Completion log length mismatch: got {len(gathered_completions)} completions for {rows_in_batch} gathered rows."
            )
        if len(gathered_advantages) != rows_in_batch:
            raise ValueError(
                f"Advantage log length mismatch: got {len(gathered_advantages)} advantages for {rows_in_batch} gathered rows."
            )

        self._logs["prompt"].extend(gathered_prompts)
        self._logs["completion"].extend(gathered_completions)
        self._logs["advantages"].extend(gathered_advantages)

        # # ------------------------------------------------------------------
        # # Align image logs to packed rows, then gather globally
        # # ------------------------------------------------------------------
        # image_logs = packed_output.get("image_logs")
        # images_aligned_local = []
        # image_idx = 0

        # for m in packed_output["row_modality"].tolist():
        #     if m == self.MODALITY_IMAGE:
        #         if image_logs is not None and image_idx < len(image_logs):
        #             images_aligned_local.append(image_logs[image_idx])
        #             image_idx += 1
        #         else:
        #             images_aligned_local.append(None)
        #     else:
        #         images_aligned_local.append(None)

        # gathered_images = gather_object(images_aligned_local)
        # if len(gathered_images) != rows_in_batch:
        #     raise ValueError(
        #         f"Image log length mismatch: got {len(gathered_images)} image entries for {rows_in_batch} gathered rows."
        #     )
        # self._logs["image"].extend(gathered_images)

        # ------------------------------------------------------------------
        # Reward logs: keep every reward column aligned to global gathered rows
        # ------------------------------------------------------------------
        if "rewards" not in self._logs:
            self._logs["rewards"] = {}

        nan_value = float("nan")
        prev_total_rows = len(self._logs["prompt"]) - rows_in_batch

        modality_to_id = {
            "ground": self.MODALITY_GROUND,
            "answer": self.MODALITY_ANSWER,
            "image": self.MODALITY_IMAGE,
        }

        # Always maintain all known reward columns at the same total length.
        known_reward_keys = {"reward_ground", "reward_answer", "reward_image"}
        known_reward_keys.update(self._logs["rewards"].keys())

        reward_key_to_values = {
            key: [nan_value] * rows_in_batch
            for key in known_reward_keys
        }

        for modality_name, payload in modality_metric_payloads.items():
            rewards_global = payload.get("global_reward")
            if rewards_global is None:
                continue

            key = f"reward_{modality_name}"
            modality_id = modality_to_id.get(modality_name)
            if modality_id is None:
                continue

            # IMPORTANT:
            # rewards_global is already gathered across ranks.
            # Do NOT gather_object() it again.
            reward_values = rewards_global.detach().cpu().tolist()

            aligned_rewards = [nan_value] * rows_in_batch
            reward_idx = 0

            for row_idx, row_modality_id in enumerate(gathered_row_modality):
                if row_modality_id == modality_id:
                    if reward_idx < len(reward_values):
                        aligned_rewards[row_idx] = float(reward_values[reward_idx])
                    reward_idx += 1

            if reward_idx != len(reward_values):
                logger.warning(
                    "Reward alignment mismatch for %s: consumed %d values for %d global rewards.",
                    modality_name,
                    reward_idx,
                    len(reward_values),
                )

            reward_key_to_values[key] = aligned_rewards

        for key in known_reward_keys:
            if key not in self._logs["rewards"]:
                self._logs["rewards"][key] = [nan_value] * prev_total_rows
            self._logs["rewards"][key].extend(reward_key_to_values[key])

        # ------------------------------------------------------------------
        # Final consistency check
        # ------------------------------------------------------------------
        total_rows = len(self._logs["prompt"])
        if len(self._logs["completion"]) != total_rows:
            raise ValueError(
                f"Accumulated completion log length mismatch: prompt={total_rows}, completion={len(self._logs['completion'])}"
            )
        if len(self._logs["advantages"]) != total_rows:
            raise ValueError(
                f"Accumulated advantage log length mismatch: prompt={total_rows}, advantages={len(self._logs['advantages'])}"
            )
        if len(self._logs["image"]) != total_rows:
            raise ValueError(
                f"Accumulated image log length mismatch: prompt={total_rows}, image={len(self._logs['image'])}"
            )
        for key, values in self._logs["rewards"].items():
            if len(values) != total_rows:
                raise ValueError(
                    f"Accumulated reward log length mismatch for {key}: prompt={total_rows}, reward={len(values)}"
                )

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
        image_completion_ids = gen_result.get("edit_completion_ids")  # (bsz_image, image_token_len) or (bsz_image, codebook_dim, image_token_len)
        edit_region_mask = gen_result.get("edit_region_mask")  # (bsz_image, image_token_len)

        ground_completion_mask = (
            self._get_text_completion_mask(ground_completion_ids)
            if ground_completion_ids is not None
            else None
        )  # (bsz_ground, ground_completion_seq_len)

        answer_completion_mask = self._get_text_completion_mask(answer_completion_ids)  # (bsz_answer, answer_completion_seq_len)

        image_completion_ids_flat = (
            self._flatten_image_completion_ids(image_completion_ids)
            if image_completion_ids is not None
            else None
        )  # (bsz_image, flat_image_completion_len)

        image_completion_mask = (
            self._get_image_completion_mask(image_completion_ids, edit_region_mask)
            if image_completion_ids is not None
            else None
        )  # (bsz_image, flat_image_completion_len)

        ground_completion_ids_list = (
            self._masked_ids_to_list(ground_completion_ids, ground_completion_mask)
            if ground_completion_ids is not None
            else None
        )
        answer_completion_ids_list = self._masked_ids_to_list(answer_completion_ids, answer_completion_mask)
        image_completion_ids_list = (
            self._masked_ids_to_list(image_completion_ids_flat, image_completion_mask)
            if image_completion_ids_flat is not None
            else None
        )

        mask_seeds = (
            torch.randint(0, 2**12, (self.num_iterations,), device=device)
            if self.args.random_masking
            else torch.full((self.num_iterations,), 42, device=device, dtype=torch.long)
        )  # (num_iter,)

        need_old_logps = (
            self.args.gradient_accumulation_steps
            % (self.args.steps_per_generation * self.num_iterations)
            != 0
        )

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_core_model = unwrapped_model.get_model() if hasattr(unwrapped_model, "get_model") else unwrapped_model

        ground_old_text_per_token_logps = None
        answer_old_text_per_token_logps = None
        old_image_per_token_logps = None

        if need_old_logps:
            with torch.inference_mode():
                if ground_completion_ids is not None:
                    ground_old_text_per_token_logps = self._get_padded_text_per_token_logps(
                        model=unwrapped_core_model,
                        prompt_ids=ground_prompt_ids,
                        prompt_input_embeds=ground_prompt_input_embeds,
                        prompt_mask=ground_prompt_mask,
                        completion_ids=ground_completion_ids,
                        mask_seeds=mask_seeds.tolist(),
                    )  # (num_iter, bsz_ground, ground_completion_seq_len)

                answer_old_text_per_token_logps = self._get_padded_text_per_token_logps(
                    model=unwrapped_core_model,
                    prompt_ids=answer_prompt_ids,
                    prompt_input_embeds=answer_prompt_input_embeds,
                    prompt_mask=answer_prompt_mask,
                    completion_ids=answer_completion_ids,
                    mask_seeds=mask_seeds.tolist(),
                )  # (num_iter, bsz_answer, answer_completion_seq_len)

                if image_completion_ids is not None:
                    old_image_per_token_logps = self._get_image_per_token_logps(
                        model=unwrapped_core_model,
                        image_completion_ids=image_completion_ids,
                        image_input_embeds_gen=prompt_inputs["input_embeds_gen"],
                        image_is_gen=prompt_inputs["is_gen"],
                        image_is_gen_enc=prompt_inputs["is_gen_enc"],
                        image_gen_shape=self._infer_image_gen_shape(image_completion_ids),
                        mask_seeds=mask_seeds.tolist(),
                    )  # (num_iter, bsz_image, flat_image_completion_len)

        ref_ground_text_per_token_logps = None
        ref_answer_text_per_token_logps = None
        ref_image_per_token_logps = None

        if self.args.beta != 0.0:
            ref_model = self.ref_model if self.ref_model is not None else unwrapped_model
            ref_core_model = ref_model.get_model() if hasattr(ref_model, "get_model") else ref_model

            with torch.inference_mode():
                if ground_completion_ids is not None:
                    ref_ground_text_per_token_logps = self._get_padded_text_per_token_logps(
                        model=ref_core_model,
                        prompt_ids=ground_prompt_ids,
                        prompt_input_embeds=ground_prompt_input_embeds,
                        prompt_mask=ground_prompt_mask,
                        completion_ids=ground_completion_ids,
                        mask_seeds=mask_seeds.tolist(),
                    )  # (num_iter, bsz_ground, ground_completion_seq_len)

                ref_answer_text_per_token_logps = self._get_padded_text_per_token_logps(
                    model=ref_core_model,
                    prompt_ids=answer_prompt_ids,
                    prompt_input_embeds=answer_prompt_input_embeds,
                    prompt_mask=answer_prompt_mask,
                    completion_ids=answer_completion_ids,
                    mask_seeds=mask_seeds.tolist(),
                )  # (num_iter, bsz_answer, answer_completion_seq_len)

                if image_completion_ids is not None:
                    ref_image_per_token_logps = self._get_image_per_token_logps(
                        model=ref_core_model,
                        image_completion_ids=image_completion_ids,
                        image_input_embeds_gen=prompt_inputs["input_embeds_gen"],
                        image_is_gen=prompt_inputs["is_gen"],
                        image_is_gen_enc=prompt_inputs["is_gen_enc"],
                        image_gen_shape=self._infer_image_gen_shape(image_completion_ids),
                        mask_seeds=mask_seeds.tolist(),
                    )  # (num_iter, bsz_image, flat_image_completion_len)

        grounding_completions = (
            tokenizer.batch_decode(ground_completion_ids, skip_special_tokens=True)
            if ground_completion_ids is not None
            else None
        )
        answer_completions = tokenizer.batch_decode(answer_completion_ids, skip_special_tokens=True)

        rewards = self._calculate_rewards(
            inputs=inputs,
            answer_prompts=answer_prompts,
            answer_completions=answer_completions,
            ground_prompts=grounding_prompts,
            ground_completions=grounding_completions,
            image_prompts=image_prompts,
            image_completion_ids_list=image_completion_ids_list,
        )

        modality_batches: list[Batch] = []
        modality_metric_payloads: dict[str, dict[str, torch.Tensor]] = {}
        prompt_logs: list[Any] = []
        completion_logs: list[Any] = []

        # ------------------------------------------------------------------
        # Ground modality
        # ------------------------------------------------------------------
        if ground_completion_ids is not None and ground_prompt_ids is not None:
            local_ground_bsz = ground_completion_ids.size(0)
            ground_reward_global = rewards["ground_reward"]
            ground_reward_local = self._slice_global_tensor_for_current_rank(ground_reward_global, local_ground_bsz)

            ground_advantages_global, _, _, _ = self._normalize_rewards(ground_reward_global)
            ground_advantages = self._slice_global_tensor_for_current_rank(ground_advantages_global, local_ground_bsz)

            ground_prompt_logs = self._repeat_list_to_length(grounding_prompts, local_ground_bsz)
            ground_completion_logs = grounding_completions

            modality_batches.append(
                {
                    "modality_id": self.MODALITY_GROUND,
                    "source_batch_size": local_ground_bsz,
                    "prompt_ids": ground_prompt_ids,
                    "prompt_mask": ground_prompt_mask,
                    "prompt_input_embeds": ground_prompt_input_embeds,
                    "completion_ids": ground_completion_ids,
                    "completion_mask": ground_completion_mask,
                    "completion_width": ground_completion_ids.size(1),
                    "advantages": ground_advantages,
                    "old_per_token_logps": ground_old_text_per_token_logps,
                    "ref_per_token_logps": ref_ground_text_per_token_logps,
                }
            )

            modality_metric_payloads["ground"] = {
                "completion_mask": ground_completion_mask,
                "global_reward": rewards["ground_reward"],
            }

            prompt_logs.extend([f"[GROUND] {p}" for p in (ground_prompt_logs or [])])
            completion_logs.extend(ground_completion_logs or [])

        # ------------------------------------------------------------------
        # Answer modality
        # ------------------------------------------------------------------
        local_answer_bsz = answer_completion_ids.size(0)
        answer_reward_global = rewards["answer_reward"]
        answer_reward_local = self._slice_global_tensor_for_current_rank(answer_reward_global, local_answer_bsz)

        answer_advantages_global, _, _, _ = self._normalize_rewards(answer_reward_global)
        answer_advantages = self._slice_global_tensor_for_current_rank(answer_advantages_global, local_answer_bsz)

        answer_prompt_logs = self._repeat_list_to_length(answer_prompts, local_answer_bsz)
        answer_completion_logs = answer_completions

        modality_batches.append(
            {
                "modality_id": self.MODALITY_ANSWER,
                "source_batch_size": local_answer_bsz,
                "prompt_ids": answer_prompt_ids,
                "prompt_mask": answer_prompt_mask,
                "prompt_input_embeds": answer_prompt_input_embeds,
                "completion_ids": answer_completion_ids,
                "completion_mask": answer_completion_mask,
                "completion_width": answer_completion_ids.size(1),
                "advantages": answer_advantages,
                "old_per_token_logps": answer_old_text_per_token_logps,
                "ref_per_token_logps": ref_answer_text_per_token_logps,
            }
        )

        modality_metric_payloads["answer"] = {
            "completion_mask": answer_completion_mask,
            "global_reward": rewards["answer_reward"],
        }

        prompt_logs.extend([f"[ANSWER] {p}" for p in (answer_prompt_logs or [])])
        completion_logs.extend(answer_completion_logs or [])

        # ------------------------------------------------------------------
        # Image modality
        # ------------------------------------------------------------------
        if image_completion_ids is not None and image_prompt_ids is not None:
            local_image_bsz = image_completion_ids_flat.size(0)
            image_reward_global = rewards["image_reward"]
            image_reward_local = self._slice_global_tensor_for_current_rank(image_reward_global, local_image_bsz)

            image_advantages_global, _, _, _ = self._normalize_rewards(image_reward_global)
            image_advantages = self._slice_global_tensor_for_current_rank(image_advantages_global, local_image_bsz)

            image_gen_shape = self._infer_image_gen_shape(image_completion_ids)
            image_prompt_logs = self._repeat_list_to_length(image_prompts, local_image_bsz)
            image_completion_logs = ["<image_completion>"] * local_image_bsz

            modality_batches.append(
                {
                    "modality_id": self.MODALITY_IMAGE,
                    "source_batch_size": local_image_bsz,
                    "prompt_ids": image_prompt_ids,
                    "prompt_mask": image_prompt_mask,
                    "prompt_input_embeds": image_prompt_input_embeds,
                    "completion_ids": image_completion_ids_flat,
                    "completion_mask": image_completion_mask,
                    "completion_width": image_completion_ids_flat.size(1),
                    "advantages": image_advantages,
                    "old_per_token_logps": old_image_per_token_logps,
                    "ref_per_token_logps": ref_image_per_token_logps,
                    "image_completion_ids": image_completion_ids,
                    "image_input_embeds_gen": prompt_inputs["input_embeds_gen"],
                    "image_is_gen": prompt_inputs["is_gen"],
                    "image_is_gen_enc": prompt_inputs["is_gen_enc"],
                    "image_gen_shape": image_gen_shape,
                }
            )

            modality_metric_payloads["image"] = {
                "completion_mask": image_completion_mask,
                "global_reward": rewards["image_reward"],
            }

            prompt_logs.extend([f"[IMAGE] {p}" for p in (image_prompt_logs or [])])
            completion_logs.extend(image_completion_logs or [])

        packed_output = self._pack_modality_rows(modality_batches)
        packed_output["mask_seeds"] = mask_seeds
        n = packed_output["prompt_ids"].size(0)
        row_modality = packed_output["row_modality"]

        reward_logs = {
            "reward_ground": [float("nan")] * n,
            "reward_answer": [float("nan")] * n,
            "reward_image": [float("nan")] * n,
        }

        def fill_reward_log(key, modality_id, local_rewards):
            if local_rewards is None:
                return
            row_indices = (row_modality == modality_id).nonzero(as_tuple=True)[0].tolist()
            values = local_rewards.detach().cpu().tolist()
            if len(row_indices) != len(values):
                raise ValueError(
                    f"{key}: row count {len(row_indices)} != local reward count {len(values)}"
                )
            for idx, val in zip(row_indices, values):
                reward_logs[key][idx] = float(val)

        fill_reward_log("reward_ground", self.MODALITY_GROUND, ground_reward_local if ground_completion_ids is not None else None)
        fill_reward_log("reward_answer", self.MODALITY_ANSWER, answer_reward_local)
        fill_reward_log("reward_image", self.MODALITY_IMAGE, image_reward_local if image_completion_ids is not None else None)

        packed_output["reward_logs"] = reward_logs
        # self._log_flat_batch_metrics(mode, packed_output, modality_metric_payloads)
        return packed_output
