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
from transformers.utils import is_peft_available
from torch import nn
from transformers.utils import is_rich_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)
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
logger = logging.getLogger(__name__)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

try:
    from lmms_eval.tasks import TaskManager, get_task_dict
    _lmms_task_manager = TaskManager()
    _LMMS_EVAL_AVAILABLE = True
except Exception:
    _LMMS_EVAL_AVAILABLE = False

EVAL_CONV_TEMPLATE = os.environ.get("EVAL_CONV_TEMPLATE", "llada")

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GrpoLMMsEvalDataset(TorchDataset):
    """
    Reads a pre-cached parquet file (from prepare_eval_datasets.py) and turns each
    row into a tokenized prompt ready for diffusion generation.

    Each item:
      input_ids    : 1-D LongTensor (not batched yet)
      modalities   : list[str]
      images       : list[Tensor] or empty list
      image_sizes  : list[tuple]
      index        : int (row_idx in parquet, 0-based)
      doc_index    : int (same as index — used by _process_results_grpo)
      prompt       : str
    """

    def __init__(
        self,
        parquet_path: str,
        task_obj,
        model_config,
        image_processor,
        conv_template: str,
        tokenizer,
        image_col_names: list,
    ):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.task_obj = task_obj
        self.model_config = model_config
        self.image_processor = image_processor
        self.conv_template = conv_template
        self.tokenizer = tokenizer
        self.image_col_names = image_col_names

    # ------------------------------------------------------------------
    def _row_to_doc(self, row: dict) -> dict:
        """Reconstruct a lmms-eval doc dict from a parquet row."""
        doc = {}
        for k, v in row.items():
            if k.endswith("_bytes"):
                col = k[: -len("_bytes")]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    doc[col] = None
                else:
                    doc[col] = Image.open(io.BytesIO(bytes(v)))
            else:
                doc[k] = v
        return doc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index].to_dict()
        doc = self._row_to_doc(row)

        # Get visuals and text using lmms-eval task methods
        visual = self.task_obj.doc_to_visual(doc)
        context = self.task_obj.doc_to_text(doc)

        if visual is None or visual == []:
            visual = []
            task_type = "text"
            image_tensor = None
            image_sizes = []
        else:
            visual = [v for v in visual if v is not None and isinstance(v, PIL.Image.Image)]
            if len(visual) == 0:
                task_type = "text"
                image_tensor = None
                image_sizes = []
            else:
                image_tensor = process_images(visual, self.image_processor, self.model_config)
                if not isinstance(image_tensor, list):
                    image_tensor = [image_tensor]
                task_type = "image"
                image_sizes = [v.size for v in visual]

        if image_tensor is not None and len(image_tensor) > 0 and DEFAULT_IMAGE_TOKEN not in context:
            placeholder_count = len(visual)
            image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * placeholder_count)
            prompts_input = image_tokens + "\n" + context
        else:
            prompts_input = context

        if "llama_3" in self.conv_template or "llada" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        conv.append_message(conv.roles[0], prompts_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        return {
            "input_ids": input_ids,
            "modalities": ["image"] if task_type == "image" else ["text"],
            "images": image_tensor if image_tensor is not None else [],
            "image_sizes": image_sizes,
            "index": index,
            "doc_index": int(row["row_idx"]),
            "prompt": prompt,
        }


@dataclass
class GrpoDataCollatorForEval:
    """Left-pad input_ids to uniform length; pass other fields through unchanged."""

    tokenizer: PreTrainedTokenizerBase

    def _pad_sequence(self, input_ids_list):
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if self.tokenizer.padding_side == "left":
            input_ids_list = [torch.flip(x, [0]) for x in input_ids_list]
        padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        if self.tokenizer.padding_side == "left":
            padded = torch.flip(padded, [1])
        return padded

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        input_ids = [inst["input_ids"] for inst in instances]
        input_ids = self._pad_sequence(input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        attention_mask = input_ids.ne(pad_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "modalities": [inst["modalities"] for inst in instances],
            "images": [inst["images"] for inst in instances],
            "image_sizes": [inst["image_sizes"] for inst in instances],
            "index": [inst["index"] for inst in instances],
            "doc_index": torch.tensor([inst["doc_index"] for inst in instances], dtype=torch.long),
            "prompt": [inst["prompt"] for inst in instances],
        }


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

    # Image column names per lmms-eval task (used by _TASK_IMAGE_COL_NAMES)
    _TASK_IMAGE_COL_NAMES = {
        "chartqa": ["image"],
        "blink_jigsaw": ["image_1", "image_2", "image_3", "image_4"],
        "cv_bench": ["image"],
        "vstar_bench": ["image"],
        "VisualPuzzles_direct": ["image"],
    }

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
        # lmms-eval mid-training evaluation state (lazily initialised on first evaluate())
        self._eval_task_dict = {}
        self._eval_df_dict = {}
        self._eval_initialized = False
        self._eval_model_config = None
        self._eval_image_processor = None

    def _move_cached_tensors(self, value, device: Union[str, torch.device]):
        if torch.is_tensor(value):
            return value.detach().to(device=device, non_blocking=(str(device) != "cpu"))
        if isinstance(value, dict):
            return {k: self._move_cached_tensors(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [self._move_cached_tensors(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_cached_tensors(v, device) for v in value)
        return value

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
        _, loaded_model, _, _ = load_pretrained_model(
            resume_from_checkpoint,
            None,
            "llava_llada",
            attn_implementation="sdpa",
            device_map="cpu",
            torch_dtype="bfloat16",
        )
        loaded_model.to(torch.bfloat16)

        # Transfer weights into the existing model instance so distributed-training
        # wrappers and device placement set up by the Trainer are preserved.
        load_result = model.load_state_dict(loaded_model.state_dict(), strict=False)

        if load_result.missing_keys:
            logger.warning(f"Checkpoint load — missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.warning(f"Checkpoint load — unexpected keys: {load_result.unexpected_keys}")

        del loaded_model
        torch.cuda.empty_cache()
        logger.info(f"Successfully loaded checkpoint from {resume_from_checkpoint}")

    # ------------------------------------------------------------------
    # Mid-training lmms-eval evaluation
    # ------------------------------------------------------------------

    def _init_eval_infrastructure(self):
        """Lazily initialise lmms-eval tasks and parquet DataFrames on first evaluate() call."""
        if self._eval_initialized:
            return

        args = self.args
        eval_tasks_str = getattr(args, "eval_tasks", None)
        eval_parquet_dir = getattr(args, "eval_parquet_dir", None)

        if not _LMMS_EVAL_AVAILABLE:
            logger.warning("lmms_eval not available — skipping mid-training eval.")
            self._eval_initialized = True
            return

        if not eval_tasks_str or not eval_parquet_dir:
            logger.info("eval_tasks or eval_parquet_dir not set — skipping mid-training eval.")
            self._eval_initialized = True
            return

        task_names = [t.strip() for t in eval_tasks_str.split(",") if t.strip()]
        try:
            self._eval_task_dict = get_task_dict(task_names, _lmms_task_manager)
        except Exception as e:
            logger.warning(f"Failed to load lmms-eval task dict: {e}")
            self._eval_initialized = True
            return

        for task_name in task_names:
            parquet_path = os.path.join(eval_parquet_dir, f"{task_name}_n100_seed42.parquet")
            if not os.path.exists(parquet_path):
                logger.warning(f"Eval parquet not found: {parquet_path} — skipping {task_name}.")
                continue
            try:
                self._eval_df_dict[task_name] = pd.read_parquet(parquet_path)
            except Exception as e:
                logger.warning(f"Failed to read parquet for {task_name}: {e}")

        # Grab model config and image processor
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            self._eval_model_config = unwrapped.config
            self._eval_image_processor = unwrapped.get_vision_tower().image_processor
        except Exception as e:
            logger.warning(f"Could not get model config / image processor for eval: {e}")

        self._eval_initialized = True

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        Complete override of Trainer.evaluate().  Runs lmms-eval benchmarks on
        pre-cached parquet splits and logs results to W&B.
        We do NOT call super().evaluate() because no eval_dataset is passed to
        the trainer (it would raise ValueError).
        """
        self._init_eval_infrastructure()

        if not self._eval_task_dict:
            return {}

        self.model.eval()
        log_dict = {}

        try:
            tokenizer = self.processing_class.tokenizer
            collator = GrpoDataCollatorForEval(tokenizer)
            args = self.args

            for task_name, task_obj in self._eval_task_dict.items():
                if task_name not in self._eval_df_dict:
                    continue

                parquet_path = os.path.join(
                    args.eval_parquet_dir, f"{task_name}_n100_seed42.parquet"
                )
                image_col_names = self._TASK_IMAGE_COL_NAMES.get(task_name, ["image"])

                eval_ds = GrpoLMMsEvalDataset(
                    parquet_path=parquet_path,
                    task_obj=task_obj,
                    model_config=self._eval_model_config,
                    image_processor=self._eval_image_processor,
                    conv_template=EVAL_CONV_TEMPLATE,
                    tokenizer=tokenizer,
                    image_col_names=image_col_names,
                )

                dataloader = self.accelerator.prepare_data_loader(
                    DataLoader(eval_ds, batch_size=1, collate_fn=collator, shuffle=False)
                )

                logger.info(
                    f"[eval] Starting {task_name} ({len(eval_ds)} samples, "
                    f"step={self.state.global_step})"
                )
                resps, doc_indices = self.generate_until_loop_grpo(
                    dataloader, task_name, task_obj
                )

                processed = self._process_results_grpo(resps, doc_indices, task_name, task_obj)

                # Gather results from all ranks
                world_size = self.accelerator.num_processes
                all_processed = [None] * world_size
                dist.all_gather_object(all_processed, processed)

                merged = collections.defaultdict(list)
                for rank_result in all_processed:
                    for metric, data in rank_result.items():
                        merged[metric].extend(data)

                agg_fns = task_obj.aggregation()

                if self.accelerator.is_main_process:
                    for metric, data in merged.items():
                        # Skip metrics not in metric_list and the "submission" sentinel
                        if metric not in agg_fns or metric == "submission":
                            continue
                        try:
                            score = agg_fns[metric](data)
                            log_dict[f"eval/{task_name}/{metric}"] = score
                            logger.info(f"[eval] {task_name}/{metric} = {score:.4f}")
                        except Exception as e:
                            logger.warning(f"[eval] aggregation failed for {task_name}/{metric}: {e}")

                self.accelerator.wait_for_everyone()
                torch.cuda.empty_cache()

            if log_dict and self.accelerator.is_main_process:
                if wandb.run is not None:
                    wandb.log({**log_dict, "train/global_step": self.state.global_step})

        finally:
            self.model.train()

        return log_dict

    def generate_until_loop_grpo(self, dataloader, task_name: str, task_obj):
        """
        Run diffusion generation over the eval dataloader.
        Returns (resps, doc_indices) — lists local to this rank.
        """
        args = self.args
        device = self.accelerator.device
        tokenizer = self.processing_class.tokenizer

        # Determine generation length
        task_gkwargs = task_obj.config.generation_kwargs or {}
        task_max_tokens = task_gkwargs.get("max_new_tokens", 16)
        cap = getattr(args, "eval_max_new_tokens_override", 64)
        gen_length = min(task_max_tokens, cap) if cap is not None else task_max_tokens

        block_length = min(getattr(args, "block_length", 64), gen_length)
        if block_length == 0:
            block_length = gen_length
        # Ensure gen_length is a multiple of block_length
        gen_length = math.ceil(gen_length / block_length) * block_length
        steps = gen_length  # same convention as training rollout

        resps = []
        doc_indices = []

        pbar = tqdm(
            total=len(dataloader),
            desc=task_name,
            disable=not self.accelerator.is_local_main_process,
        )

        with torch.inference_mode():
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                inner_model = unwrapped_model.get_model()

                for batch in dataloader:
                    modalities = batch.pop("modalities")[0]
                    image_sizes = batch.pop("image_sizes")[0]
                    images = batch["images"][0]
                    input_ids = batch["input_ids"]
                    doc_idx = batch["doc_index"][0].item()

                    # Build prompt embeddings (vision-aware)
                    if images is not None and len(images) > 0:
                        images_dev = [img.to(dtype=torch.bfloat16, device=device) for img in images]
                        (_, _, attn_mask, _, inputs_embeds, _) = (
                            unwrapped_model.prepare_inputs_labels_for_multimodal(
                                input_ids,
                                None,
                                batch["attention_mask"],
                                None,
                                None,
                                images_dev,
                                modalities,
                                image_sizes=image_sizes,
                            )
                        )
                    else:
                        inputs_embeds = unwrapped_model.get_model().embed_tokens(
                            input_ids.to(device)
                        )
                        attn_mask = batch["attention_mask"]

                    # Truncate prompt to max_prompt_length if needed
                    if args.max_prompt_length is not None:
                        inputs_embeds = inputs_embeds[:, -args.max_prompt_length:]
                        attn_mask = attn_mask[:, -args.max_prompt_length:]

                    # Diffusion generation
                    completion_ids = self.generate(
                        model=inner_model,
                        prompt=None,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        position_ids=None,
                        tokenizer=tokenizer,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking=args.remasking,
                        mask_id=args.mask_id,
                        t2i_inference=False,
                        do_sample=False,
                        prefix_lm=True,
                    )

                    text = tokenizer.decode(
                        completion_ids[0], skip_special_tokens=True
                    ).strip()
                    resps.append(text)
                    doc_indices.append(doc_idx)
                    pbar.update(1)

        pbar.close()
        return resps, doc_indices

    def _process_results_grpo(
        self,
        resps,
        doc_indices,
        task_name: str,
        task_obj,
    ):
        """
        Run task_obj.process_results() for each (resp, doc_idx) pair.
        Returns a dict {metric_name: [result, ...]} local to this rank.
        """
        df = self._eval_df_dict[task_name]
        processed_results = collections.defaultdict(list)

        for resp, doc_idx in zip(resps, doc_indices):
            rows = df[df["row_idx"] == doc_idx]
            if len(rows) == 0:
                logger.warning(f"[eval] row_idx={doc_idx} not found in {task_name} parquet")
                continue
            row = rows.iloc[0].to_dict()
            # Reconstruct meta doc (strip _bytes columns, no image decoding needed for process_results)
            meta_doc = {k: v for k, v in row.items() if not k.endswith("_bytes")}
            try:
                result = task_obj.process_results(meta_doc, [resp])
                for metric, data in result.items():
                    processed_results[metric].append(data)
            except Exception as e:
                logger.warning(f"[eval] process_results failed for {task_name} doc {doc_idx}: {e}")

        return processed_results

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # GRPO-specific: this generates rollouts and prepares trajectory-level batch
        inputs = self._prepare_inputs(inputs)

        # -----------------------------------------------------------
        # PPO MINI-BATCH SPLITTING
        # -----------------------------------------------------------
        ppo_tasks = getattr(self.args, "ppo_mini_batch_size", None)

        if ppo_tasks is None:
            # Fallback to original single backward behavior
            if self.accelerator.is_main_process:
                logger.info(f"[Step {self.state.global_step}] Update    | start | single batch, B={inputs['prompt_ids'].size(0)}")
            _t_upd = time.perf_counter()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                loss = loss / self.current_gradient_accumulation_steps

            self.accelerator.backward(loss)
            if self.accelerator.is_main_process:
                logger.info(
                    f"[Step {self.state.global_step}] Update    | done  | {time.perf_counter() - _t_upd:.1f}s, "
                    f"loss={loss.item():.4f}"
                )

            self._step += 1
            time_after = time.perf_counter()
            self._current_train_step_time += time_after - time_before
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics["train"]["step_time"].append(self._current_train_step_time)
                self._current_train_step_time = 0.0

            return loss.detach()

        # Convert task-level mini-batch size → trajectory-level
        rollout_n = self.num_generations
        ppo_mb_global = ppo_tasks * rollout_n

        world = self.accelerator.num_processes
        if ppo_mb_global % world != 0:
            raise ValueError(
                f"ppo_mini_batch_size (task-level={ppo_tasks}) "
                f"* rollout.n ({rollout_n}) must be divisible by num_processes ({world})"
            )

        ppo_mb_local = ppo_mb_global // world
        B = inputs["prompt_ids"].size(0)

        if ppo_mb_local <= 0:
            raise ValueError(f"Local PPO mini-batch size <= 0: {ppo_mb_local}")

        need_ga_divide = (
            (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
            and self.compute_loss_func is None
        )
        ga = float(self.current_gradient_accumulation_steps) if need_ga_divide else 1.0

        total_loss = 0.0
        n_chunks = (B + ppo_mb_local - 1) // ppo_mb_local
        if self.accelerator.is_main_process:
            logger.info(
                f"[Step {self.state.global_step}] Update    | start | PPO mini-batch, "
                f"B={B}, mb_local={ppo_mb_local}, n_chunks={n_chunks}"
            )
        _t_upd = time.perf_counter()

        # Avoid metric duplication during chunked loss computation
        old_skip_flag = getattr(self, "_skip_loss_metrics", False)

        for start in range(0, B, ppo_mb_local):
            end = min(start + ppo_mb_local, B)
            chunk_bs = end - start
            chunk_idx = start // ppo_mb_local

            # Slice batch-aligned tensors
            chunk = {}
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == B:
                    chunk[k] = v[start:end]
                elif torch.is_tensor(v) and v.dim() >= 2 and v.size(1) == B:
                    chunk[k] = v[:, start:end]
                else:
                    chunk[k] = v

            # Scale num_items_in_batch proportionally if used
            chunk_num_items = None
            if num_items_in_batch is not None:
                if torch.is_tensor(num_items_in_batch):
                    chunk_num_items = num_items_in_batch * (chunk_bs / B)
                else:
                    chunk_num_items = num_items_in_batch * (chunk_bs / B)

            # Skip metric logging except for final chunk
            self._skip_loss_metrics = (end != B)

            _t_chunk = time.perf_counter()
            with self.compute_loss_context_manager():
                chunk_loss = self.compute_loss(model, chunk, num_items_in_batch=chunk_num_items)

            if self.args.n_gpu > 1:
                chunk_loss = chunk_loss.mean()

            # Weight chunk to match full-batch mean gradient
            chunk_loss = chunk_loss * (chunk_bs / B)

            # Apply gradient accumulation scaling
            chunk_loss = chunk_loss / ga

            self.accelerator.backward(chunk_loss)
            if self.accelerator.is_main_process:
                logger.info(
                    f"[Step {self.state.global_step}] Update    | chunk {chunk_idx + 1}/{n_chunks} | "
                    f"{time.perf_counter() - _t_chunk:.1f}s, loss={chunk_loss.item():.4f}"
                )

            total_loss += chunk_loss.detach()

        self._skip_loss_metrics = old_skip_flag
        if self.accelerator.is_main_process:
            logger.info(
                f"[Step {self.state.global_step}] Update    | done  | {time.perf_counter() - _t_upd:.1f}s, "
                f"total_loss={total_loss.item():.4f}"
            )

        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0

        return total_loss


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]
        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens
        input_ids = input_ids.unsqueeze(0)
        unwrapped_model = self.accelerator.unwrap_model(model).get_model()
        per_token_logps = self._get_per_token_logps(
            unwrapped_model, input_ids, logits_to_keep, [this_itr_mask_seed]
        ).squeeze(0)
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

    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds):
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
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            buf_key = self._step % self.args.gradient_accumulation_steps
            if self.state.global_step % self.num_iterations == 0 or buf_key not in self._buffered_inputs:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[buf_key] = self._move_cached_tensors(inputs, "cpu")
            else:
                inputs = self._move_cached_tensors(self._buffered_inputs[buf_key], self.accelerator.device)
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
                mask_id=self.args.mask_id,
                mode="image_gen",
                do_cfg= self.args.guidance_scale > 0 ,
            )
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
                mask_id=self.args.mask_id,
                mode="grounding",
            )
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
                mask_id=self.args.mask_id,
                mode="text_gen",
            )

        generation_prompts = grounding_prompts if gen_type == "grounding" else prompts_text
        input_embeds = batch_inputs["input_embeds"]
        attention_mask = batch_inputs["attention_mask"]
        bbox_mask = batch_inputs.get("bbox_mask", None)
        if gen_type in {"grounding", "image_gen"} and bbox_mask is None:
            raise ValueError("bbox_mask is required for grounding and image_gen modes.")
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
            generation_batch_size = self.args.generation_batch_size
            init_images = None
            image_sizes = None
            bbox_ids = None
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

            def _bbox_postprocess_fn(bbox_ids, _batch_indices):
                decoded = self.processing_class.tokenizer.batch_decode(
                    bbox_ids, skip_special_tokens=False
                )
                pred_bboxes = []
                for decoded_bbox in decoded:
                    loc_vals = [int(v) for v in re.findall(r"<LOC_([0-9]+)>", decoded_bbox)]
                    pred_bboxes.append(loc_vals[:4] if len(loc_vals) >= 4 else [])
                return pred_bboxes, decoded

            # TODO: Re-encode -> Use the latent output embedding during generate_image()
            def _reencode_fn(batch_generation_prompts, batch_image_groups, _batch_start, _batch_end): 
                re_batch_inputs = self.processing_class(
                    texts=batch_generation_prompts,
                    images=batch_image_groups,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                    device=device,
                    dtype=torch.bfloat16,
                    mask_id=self.args.mask_id,
                    mode="text_gen",
                )
                batch_input_embeds = re_batch_inputs["input_embeds"]
                batch_attention_mask = re_batch_inputs["attention_mask"]
                if self.args.max_prompt_length is not None:
                    batch_input_embeds = batch_input_embeds[:, -self.args.max_prompt_length :]
                    batch_attention_mask = batch_attention_mask[:, -self.args.max_prompt_length :]
                return batch_input_embeds, batch_attention_mask

            image_gen_kwargs = None
            if gen_type == "image_gen":
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

            gen_result = unwrapped_model._generate_mode(
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
                bbox_postprocess_fn=_bbox_postprocess_fn if gen_type in {"grounding", "image_gen"} else None,
                reencode_fn=_reencode_fn if gen_type == "image_gen" else None,
                image_gen_kwargs=image_gen_kwargs,
                return_debug=return_debug_artifacts and gen_type == "image_gen",
            )
            completion_ids = gen_result["completion_ids"]
            prompt_mask = gen_result["prompt_mask"]
            bbox_ids = gen_result.get("bbox_ids")
            if gen_type in {"grounding", "image_gen"}:
                pred_bboxes_all = gen_result.get("pred_bboxes", pred_bboxes_all)
                bbox_texts = gen_result.get("bbox_texts")
                if bbox_texts is None and bbox_ids is not None:
                    bbox_texts = self.processing_class.tokenizer.batch_decode(
                        bbox_ids, skip_special_tokens=False
                    )
                if bbox_texts is not None:
                    grounding_bbox_completion_text_all = bbox_texts
            if gen_type == "image_gen":
                edited_images_all = gen_result.get("edited_images", edited_images_all)
                if return_debug_artifacts:
                    image_gen_debug_all = gen_result.get("image_gen_debug", image_gen_debug_all)
        if self.accelerator.is_main_process:
            logger.info(
                f"[Step {self.state.global_step}] Generate  | done  | {time.perf_counter() - _t_gen:.1f}s"
            )

        # With prefix_lm=True, generate() returns only the gen_length completion tokens.
        # The prompt was handled via embeddings, so prompt_ids is empty.
        prompt_ids = completion_ids[:, :0]      # empty, shape [bs, 0]

        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
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
                old_per_token_logps = self._get_per_token_logps(
                    self.accelerator.unwrap_model(self.model).get_model(),
                    prompt_completion_ids_expanded,
                    logits_to_keep,
                    mask_seeds,
                )
                all_old_per_token_logps = old_per_token_logps
                if self.accelerator.is_main_process:
                    logger.info(
                        f"[Step {self.state.global_step}] Old logps | done  | {time.perf_counter() - _t_old:.1f}s"
                    )
            else:
                old_per_token_logps = None

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
                    ref_per_token_logps = self._get_per_token_logps(
                        self.accelerator.unwrap_model(self.model).get_model(),
                        prompt_completion_ids_expanded,
                        logits_to_keep,
                        mask_seeds,
                    )
                    all_ref_per_token_logps = ref_per_token_logps
                if self.accelerator.is_main_process:
                    logger.info(
                        f"[Step {self.state.global_step}] Ref logps | done  | {time.perf_counter() - _t_ref:.1f}s"
                    )

        decode_skip_special_tokens = gen_type != "grounding"
        completions_text = self.processing_class.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=decode_skip_special_tokens
        )
        reward_prompts = grounding_prompts if gen_type == "grounding" else prompts_text
        completions = _build_reward_completions(reward_prompts, completions_text)
        answer_reward_prompts = prompts_text
        answer_completions_text = completions_text
        answer_completions = _build_reward_completions(answer_reward_prompts, answer_completions_text)
        grounding_reward_prompts = grounding_prompts
        grounding_completions_text = grounding_bbox_completion_text_all
        if gen_type == "grounding":
            grounding_completions_text = completions_text
        elif bbox_ids is not None and not any(x is not None for x in grounding_completions_text):
            grounding_completions_text = self.processing_class.tokenizer.batch_decode(
                bbox_ids, skip_special_tokens=False
            )
        grounding_completions = _build_reward_completions(
            grounding_reward_prompts,
            grounding_completions_text,
        )
        log_prompts = reward_prompts
        log_completions_text = completions_text
        if gen_type == "image_gen":
            log_prompts = [
                _format_image_gen_prompt_log(grounding_prompt, answer_prompt)
                for grounding_prompt, answer_prompt in zip(grounding_reward_prompts, answer_reward_prompts)
            ]
            log_completions_text = [
                _format_image_gen_completion_log(grounding_completion, answer_completion)
                for grounding_completion, answer_completion in zip(
                    grounding_completions_text, answer_completions_text
                )
            ]

        rewards_per_func = torch.zeros(len(prompts_text), len(self.reward_funcs), device=device)
        last_reward_prompts = reward_prompts
        last_reward_completions = completions
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_func_name = _get_reward_func_name(reward_func)
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                current_reward_prompts = reward_prompts
                current_reward_completions = completions
                if gen_type == "image_gen":
                    if reward_func_name == "correct_grounding_reward_func":
                        current_reward_prompts = grounding_reward_prompts
                        current_reward_completions = grounding_completions_text
                    else:
                        current_reward_prompts = answer_reward_prompts
                        current_reward_completions = answer_completions
                if reward_func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                if reward_func_name == "correct_grounding_reward_func":
                    reward_kwargs["answer"] = [
                        example.get("gt_bbox", example.get("answer")) for example in inputs
                    ]
                last_reward_prompts = current_reward_prompts
                last_reward_completions = current_reward_completions
                output_reward_func = reward_func(
                    prompts=current_reward_prompts,
                    completions=current_reward_completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
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
        reward_columns_to_log = {"reward": rewards.tolist()}
        for i, reward_func in enumerate(self.reward_funcs):
            reward_columns_to_log[_get_reward_func_name(reward_func)] = rewards_per_func[:, i].tolist()

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
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = _get_reward_func_name(reward_func)
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
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

            def _align_len(values, target_len, pad_value=None):
                values = list(values)
                if len(values) < target_len:
                    values.extend([pad_value] * (target_len - len(values)))
                return values[:target_len]

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
        if return_debug_artifacts and gen_type == "image_gen":
            result["pred_bboxes"] = pred_bboxes_all
            result["edited_images"] = edited_images_all
            result["image_gen_debug"] = image_gen_debug_all
            result["grounding_bbox_completion_text"] = grounding_bbox_completion_text_all
        return result
