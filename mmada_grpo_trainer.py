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

class MMADAGRPOTrainer(GRPOTrainer):
    """
    !!!!!!!!!!!UNIMPLEMENTED YET!!!!!!!!!!
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
