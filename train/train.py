
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List,Union
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import deepspeed
from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN,DEFAULT_IMAGE_GEN_TOKEN,DEFAULT_IMAGE_GEN_TOKEN_XTD
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX,IMAGE_TOKEN_INDEX_GEN
from llava.train.llava_trainer import LLaVATrainer

import llava.conversation as conversation_lib
from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord
from llava.model.language_model.llava_llada import LlavaLladaForMaskedDiffusion

from torch.utils.data import ConcatDataset

import warnings
from transformers.trainer import Trainer
# Suppress specific deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers.trainer")


torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)
    mm_pooler_ratio: Optional[int] = field(default=2)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)
    prefix_lm: bool = False
    unified_gen: bool = False
    dual_tower: bool = False
    vqvae: str = None
    prompt_drop_rate: float = None
    image_enc_drop_rate: float = None
    dual_tower_layers: int = None
    mm_submask: Optional[bool] = None
    enc_use_image_branch: Optional[bool] = None
    flip_ratio: Optional[float] = None
    block_causal: Optional[bool] = None
    num_register_tokens: Optional[str] = None
    num_register_groups: Optional[str] = None
    gen_enc_add_pos_emb: Optional[bool] = None
    block_causal_purge: Optional[float] = None



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)
    image_gen_size: int = 512
    num_gen_image_tokens: int = 1024
    num_gen_image_tokens_enc_ds: int = 1
    mm_edit_area_weight: float = 5.0

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    dev: str = None
    load_vlm: bool = False
    policy: str = "uniform"
    policy_args:json.loads = field(default_factory=dict)
    lmms_eval_generate_tasks:  str = ""
    t2i_eval:  bool = False
    eval_only: bool = False
    add_loc_tokens: bool = False
    add_vision_tokens: bool = False
    lmms_eval_extra_tasks: str = ""
    lmms_eval_limit: int = field(default=-1, metadata={"help": "Limit number of samples per eval task (-1 for no limit)"})
    group_by_random_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


from PIL import Image
import pandas as pd
from train.data.process_functions import PROCESS_FUNCTIONs
from train.data.datasets import build_dataset_lazy

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value,extra_pad=-1):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        # breakpoint()
        input_ids = list(input_ids)
        max_k = max(range(len(input_ids)),key=lambda x:input_ids[x].shape[-1])

        # extra_pad = -1
        if extra_pad > 0 :
            extra_pad_seq = torch.tensor([padding_value]*extra_pad)
            input_ids[max_k] = torch.cat([input_ids[max_k],extra_pad_seq])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        extra_pad = np.random.randint(-128,128)
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id,extra_pad=extra_pad)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX,extra_pad=extra_pad)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        images_gen = list([instance["image_gen"] for instance in instances if instance["image_gen"] is not None])
        image_gen_enc = list([instance["image_gen_enc"] for instance in instances if instance["image_gen_enc"] is not None])
        image_gen_weight = list([instance["image_gen_weight"] for instance in instances if instance["image_gen_weight"] is not None])

        if len(images_gen)>0:
            batch['images_gen'] = images_gen
        else:
            batch['images_gen']  = None
        if len(image_gen_enc)>0:
            batch['images_gen_enc'] = image_gen_enc
        else:
            batch['images_gen_enc']  = None
        if len(image_gen_weight) > 0:
            batch['image_gen_weight'] = image_gen_weight
        else:
            batch['image_gen_weight'] = None

        if 'name' in instances[0]:
            batch['dataset_name'] = instances[0]['name']

        batch['do_not_mask_text'] = [x['do_not_mask_text'] for x in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = build_dataset_lazy(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args,
    )
    eval_dataset = train_dataset
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train(model_path=None, attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # hack
    data_args.group_by_random_length = training_args.group_by_random_length
    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # =====================================================================
    # Load pretrained LaViDa-O model using load_pretrained_model
    # This is a simpler initialization compared to train_copy.py since
    # the model already has vision tower, projector, and generation modules
    # =====================================================================
    rank0_print("Loading LaViDa-O Model...")

    vision_kwargs = dict(
        mm_vision_tower="google/siglip-so400m-patch14-384",
        mm_resampler_type=model_args.mm_resampler_type,
        mm_projector_type=model_args.mm_projector_type,
        mm_hidden_size=1152,
        use_mm_proj=True,
        mm_patch_merge_type=model_args.mm_patch_merge_type,
    )

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_args.model_name_or_path,
        None,
        "llava_llada",
        device_map="cpu",
        vision_kwargs=vision_kwargs,
        torch_dtype='bfloat16' if training_args.bf16 else 'float16',
    )

    model.config.use_cache = False

    # =====================================================================
    # Gradient checkpointing setup
    # =====================================================================
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # =====================================================================
    # Setup conversation template
    # =====================================================================
    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # =====================================================================
    # Initialize generation modules if unified_gen is enabled
    # =====================================================================
    if model_args.unified_gen:
        model.get_model().initialize_generation_modules(model_args=model_args, config=model.config)

    # =====================================================================
    # Setup vision tower and image processors
    # Note: Do NOT move model/vision_tower to GPU here - DeepSpeed will handle device placement
    # =====================================================================
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)

    data_args.image_processor = vision_tower.image_processor
    if model_args.unified_gen:
        data_args.image_processor_gen = model.get_model().image_processor_gen
    data_args.is_multimodal = True

    # =====================================================================
    # Setup model config with data args
    # =====================================================================
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.mm_patch_merge_type = model_args.mm_patch_merge_type

    # Parse image_grid_pinpoints
    if data_args.image_grid_pinpoints is not None:
        if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
            try:
                patch_size = data_args.image_processor.size[0]
            except Exception as e:
                patch_size = data_args.image_processor.size["shortest_edge"]

            assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
            matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
            range_start = tuple(map(int, matches[0]))
            range_end = tuple(map(int, matches[-1]))
            grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
            data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
        elif isinstance(data_args.image_grid_pinpoints, str):
            data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
    else:
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            data_args.image_grid_pinpoints = [[g[0]*base_size, g[1]*base_size] for g in grids]

    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position
    model.config.add_faster_video = model_args.add_faster_video
    model.config.faster_token_stride = model_args.faster_token_stride
    model.config.add_time_instruction = data_args.add_time_instruction
    model.config.force_sample = data_args.force_sample
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

    # =====================================================================
    # Setup tunable parts - freeze/unfreeze parameters based on mm_tunable_parts
    # =====================================================================
    rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
    model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts

    # Set the entire model to not require gradients by default
    model.requires_grad_(False)
    if vision_tower is not None:
        vision_tower.requires_grad_(False)
    if hasattr(model.get_model(), 'mm_projector') and model.get_model().mm_projector is not None:
        model.get_model().mm_projector.requires_grad_(False)
    if hasattr(model.get_model(), 'vision_resampler') and model.get_model().vision_resampler is not None:
        model.get_model().vision_resampler.requires_grad_(False)

    # Parse the mm_tunable_parts to decide which parts to unfreeze
    tunable_parts = model_args.mm_tunable_parts.split(",") if model_args.mm_tunable_parts else []

    if "mm_mlp_adapter" in tunable_parts:
        if hasattr(model.get_model(), 'mm_projector') and model.get_model().mm_projector is not None:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
    if "mm_vision_resampler" in tunable_parts:
        if hasattr(model.get_model(), 'vision_resampler') and model.get_model().vision_resampler is not None:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True
    if "mm_vision_tower" in tunable_parts:
        for name, param in model.named_parameters():
            if "vision_tower" in name:
                param.requires_grad_(True)
    if "mm_language_model" in tunable_parts:
        for name, param in model.named_parameters():
            if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                param.requires_grad_(True)

    tunable_parts_weight = []
    if "mm_language_model_vision_parms" in tunable_parts:
        for name, param in model.named_parameters():
            if "_vision" in name or "gen_embedding" in name or "gen_predictor" in name or '_gen' in name:
                param.requires_grad_(True)
                tunable_parts_weight.append(name)

    if "mm_updown_layers" in tunable_parts:
        for name, param in model.named_parameters():
            if "upsample_gen" in name or "downsample_gen" in name or 'gen_predictor_2' in name or 'gen_embedding_2' in name or 'downsample_gen_enc' in name:
                param.requires_grad_(True)
                tunable_parts_weight.append(name)

    if "mm_gen_input_output_layers" in tunable_parts:
        for name, param in model.named_parameters():
            if "upsample_gen" in name or "downsample_gen" in name or 'gen_predictor' in name or 'gen_embedding' in name or 'downsample_gen_enc' in name:
                param.requires_grad_(True)
                tunable_parts_weight.append(name)

    if 'extra_gen_dit' in tunable_parts:
        for name, param in model.named_parameters():
            if "extra_gen_dit" in name:
                param.requires_grad_(True)
                tunable_parts_weight.append(name)

    if 'gen_predictor' in tunable_parts:
        for name, param in model.named_parameters():
            if "gen_predictor" in name:
                param.requires_grad_(True)
                tunable_parts_weight.append(name)

    # Always freeze vqvae
    for name, param in model.named_parameters():
        if 'vqvae' in name or 'vq_model' in name or 'vqvae_model' in name:
            param.requires_grad_(False)

    print(f"Unfreezing the following parts: {tunable_parts}")
    print(f"Trainable weights {tunable_parts_weight}")

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

    # =====================================================================
    # Setup model config for vision tokenizer
    # =====================================================================
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # =====================================================================
    # Add LOC tokens if enabled
    # =====================================================================
    if training_args.add_loc_tokens:
        special_tokens = list([f'<LOC_{i}>' for i in range(1025)])
        special_tokens.extend(['<box_p>','</box_p>','<LOC_BEGIN>','<LOC_END>'])
        tokenizer.add_special_tokens(dict(additional_special_tokens=special_tokens))
        model.config.get_text_config(decoder=True).tie_word_embeddings = False
        model.resize_token_embeddings(len(tokenizer))

    if training_args.add_vision_tokens:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=["<vision_start>", "<vision_end>"])
        )
        model.config.get_text_config(decoder=True).tie_word_embeddings = False
        model.resize_token_embeddings(len(tokenizer))

    # =====================================================================
    # Build dataset and data module
    # =====================================================================
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer_cls = LLaVATrainer

    training_args.group_names = data_args.group_names
    training_args.group_lengths = data_args.group_lengths
    training_args.group_weights = data_args.group_weights
    training_args.group_bs_factor = data_args.group_bs_factor

    rank0_print(f"Training args group names: {training_args.group_names}")
    rank0_print(f"Training args group lengths: {training_args.group_lengths}")
    rank0_print(f"Training args group weights: {training_args.group_weights}")
    rank0_print(f"Training args group batch sizes: {training_args.group_bs_factor}")

    # Debug: Print model parameter dtypes and devices
    rank0_print("Checking model parameters before trainer initialization...")
    dtypes = set()
    devices = set()
    for name, param in model.named_parameters():
        dtypes.add(str(param.dtype))
        devices.add(str(param.device))
    rank0_print(f"Model parameter dtypes: {dtypes}")
    rank0_print(f"Model parameter devices: {devices}")

    trainer = trainer_cls(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if training_args.eval_only:
        log_dict = trainer.evaluate()
        print(log_dict)
        return

    training_args.image_gen_size = data_args.image_gen_size

    # =====================================================================
    # Start training
    # =====================================================================
    try:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception as e:
        rank0_print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    trainer.save_state()

    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
