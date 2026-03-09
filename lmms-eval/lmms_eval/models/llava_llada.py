# ADOBE CONFIDENTIAL
# Copyright 2025 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains
# the property of Adobe and its suppliers, if any. The intellectual
# and technical concepts contained herein are proprietary to Adobe
# and its suppliers and are protected by all applicable intellectual
# property laws, including trade secret and copyright laws.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Adobe.

import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import os
import PIL
import torch
import accelerate
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
import time
# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)

# Import LLaVA modules
DEBUG_LOAD_TRAINER = os.environ.get('DEBUG_LOAD_TRAINER',False)
from constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from conversation import conv_templates
from mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    pad_to_square_and_resize,
)
from llava.model.builder import load_pretrained_model
from llava.model.utils import pad_along_last_dim


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_llada")
class Llava_Llada(lmms):
    """
    Llava Model
    """
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "llava_llada",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        mc_num=16,
        chat_mode: Optional[str] = None,
        img_gen_guidance_scale: float = 1.2,
        img_gen_guidance_scale_image: float = 1.4,
        img_gen_conf_policy: str = "stratified",
        img_gen_edit_mode: int = 1,
        img_gen_n_steps: int = 64,
        img_gen_temperature: float = 0.8,
        img_gen_enable_stratified: bool = False,
        img_gen_save_dir: Optional[str] = None,
        img_gen_resolution: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate and store chat_mode
        VALID_CHAT_MODES = (None, "text_gen", "image_gen")
        if chat_mode not in VALID_CHAT_MODES:
            raise ValueError(f"Invalid chat_mode={chat_mode!r}. Must be one of {VALID_CHAT_MODES}")
        self.chat_mode = "text_gen" if chat_mode is None else chat_mode

        # Store image generation parameters with explicit type casts
        self.img_gen_guidance_scale = float(img_gen_guidance_scale)
        self.img_gen_guidance_scale_image = float(img_gen_guidance_scale_image)
        self.img_gen_conf_policy = str(img_gen_conf_policy)
        self.img_gen_edit_mode = int(img_gen_edit_mode)
        self.img_gen_n_steps = int(img_gen_n_steps)
        self.img_gen_temperature = float(img_gen_temperature)
        self.img_gen_enable_stratified = bool(img_gen_enable_stratified)
        self.img_gen_save_dir = str(img_gen_save_dir) if img_gen_save_dir is not None else None
        self.img_gen_resolution = int(img_gen_resolution)

        # Derive a sanitized model name for gen image directory
        # datetime_str will be appended once set by the evaluator (after __init__)
        _model_basename = os.path.basename(pretrained.rstrip("/")) or "model"
        _model_basename = re.sub(r'[^\w\-.]', '_', _model_basename)
        self._gen_img_dir_base = os.path.join("train_sft/outputs/gen_imgs", _model_basename)
        self._gen_img_dir = None  # will be resolved lazily via property
        self.datetime_str = None  # set by evaluator after model creation

        if kwargs:
            eval_logger.warning(f"Unexpected kwargs (ignored): {kwargs}")

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        self.mc_num = mc_num
        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = 'llava_llada'# if model_name is not None else get_model_name_from_path(pretrained)
        self.overwrite_image_aspect = os.environ.get("LLAVA_OVERWRITE_IMAGE_ASPECT", None)
        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode

        llava_model_args["overwrite_config"] = overwrite_config
        # try:
            # Try to load the model with the multimodal argument
            
        if os.path.exists('/data1/jacklishufan/siglip-so400m-patch14-384'):
            vision_tower_path = "/data1/jacklishufan/siglip-so400m-patch14-384"
        else:
            vision_tower_path="/data0/jacklishufan/siglip-so400m-patch14-384"
        print(vision_tower_path)
        vision_tower_path = "google/siglip-so400m-patch14-384"
        # vision_kwargs = dict(
        #     mm_vision_tower=os.environ.get('LLADA_VISION_ENCODER',vision_tower_path),
        #     mm_resampler_type=None,
        #     mm_projector_type=os.environ.get('LLADA_VISION_PROJECTOR','mlp2x_gelu'),
        #     mm_hidden_size=int(os.environ.get('LLADA_VISION_ENCODER_HIDDEN_SIZE',1152)),
        #     mm_pooler_ratio=int(os.environ.get('LLADA_MM_POOLER_RATIO',2)),
        #     use_mm_proj=True,
        #     mm_patch_merge_type='spatial_unpad',            
        # )
        vision_kwargs = None
        resize_embeddings = True # default behavior
        if DEBUG_LOAD_TRAINER:
            resize_embeddings = False
            
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args,vision_kwargs=vision_kwargs,resize_embeddings=resize_embeddings)
        
        assert self._tokenizer is not None

        self._config = self._model.config
        self.model.eval()
        self.model.model.set_activation_checkpointing(None)
        self.model.requires_grad_(False)
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # Image generation modes now support batched inference via text_to_image_batch

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            self.model.to(self._device).to(torch.bfloat16)
            self._model.model.transformer = accelerator.prepare(self.model.model.transformer)
        
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device).to(torch.bfloat16)
            self._rank = 0
            self._world_size = 1
        #self.model.model.transformer = accelerate.cpu_offload(self.model.model.transformer)

    @property
    def gen_img_dir(self):
        """Lazily resolve gen image directory with datetime_str appended."""
        if self._gen_img_dir is None:
            suffix = self.datetime_str if self.datetime_str else "unknown"
            self._gen_img_dir = f"{self._gen_img_dir_base}_{suffix}"
        return self._gen_img_dir

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                    task_type = "video"

                # elif type(visual[0]) == PIL.Image.Image:
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts


            if "llama_3" in self.conv_template or 'llada' in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            # conv.messages[-1][1] = continuation
            # full_prompt = conv.get_prompt()
            # full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            # labels = full_input_ids.clone()
            # labels[0, : input_ids.shape[1]] = -100
            input_prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(input_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            answers = continuation
            answer_ids = self.tokenizer(continuation)['input_ids']
            answer_ids = torch.tensor(continuation).to(input_ids.device).unsqueeze(0) 


            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            torch.cuda.empty_cache()
            # with torch.inference_mode():
                #outputs = self.model(input_ids=full_input_ids, labels=labels, images=image_tensor, use_cache=True, **kwargs)
            likelyhoods = self.model.log_likelyhood_inference(
                input_ids,
                images=image_tensor.to(torch.bfloat16),
                image_sizes=None,
                verbose=True,
                answer=answer_ids,
                mc_num=self.mc_num,
            ) 

            # loss = outputs["loss"]
            # logits = outputs["logits"]
            # greedy_tokens = logits.argmax(dim=-1)
            # cont_toks = full_input_ids[:, input_ids.shape[1] :]
            # greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            # max_equal = (greedy_tokens == cont_toks).all()
            # lmms eval return loss
            res.append((float(-likelyhoods.item()), False))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def _pad_image_for_gen(self, pil_image):
        """Pad image to square and resize to configured generation resolution."""
        return self.model.pad_image(pil_image, image_resolution=self.img_gen_resolution)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            # breakpoint()
            return -len(toks), x[0]

        metadata = requests[0].metadata
        if DEBUG_PRINT_OUTPUT:
            # do not sort by length, instead using lambda x:x[-3]
            re_ords = utils.Collator([reg.args for reg in requests], lambda x:x[-3], grouping=True)
        else:
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)
        if DEBUG_LOAD_TRAINER:
            ckpt1 = torch.load(DEBUG_LOAD_TRAINER, map_location='cpu')
            ckpt1 = {k.replace('module.model','model'):v for k,v in ckpt1.items()}
            _res = self.model.load_state_dict(ckpt1,strict=False)
            print(f"DEBUG_LOAD_TRAINER:{DEBUG_LOAD_TRAINER} {_res}")
            print("Something is broken if above line does not show all keys matched!!!")
            del ckpt1
        delta_t = 0
        num_generated = 0
        # Set up generation kwargs
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            batch_size = len(batched_contexts)
            batched_visuals = [
                doc_to_visual(self.task_dict[task_name][split_name][doc_id])
                for doc_to_visual, task_name, split_name, doc_id in zip(
                    batched_doc_to_visual, batched_task, batched_split, batched_doc_id
                )
            ]  # [B, N]

            image_gen_post_prompt = gen_kwargs.pop("image_gen_post_prompt", "")

            t0 = time.time()

            needs_image_gen = self.chat_mode == "image_gen"

            # --- Build per-sample PIL image lists and detect task type ---
            batch_pil_images = []   # List[List[PIL.Image.Image]]
            batch_pil_sizes = []    # List[List[tuple]]
            task_type = "text"      # updated below; same across a batch within a task

            for sample_idx, visual in enumerate(batched_visuals):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                if self.overwrite_image_aspect:
                    self._config.image_aspect_ratio = self.overwrite_image_aspect

                if visual is None or visual == []:
                    batch_pil_images.append([])
                    batch_pil_sizes.append([])

                elif "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:
                    # video frames stored as PIL images sampled from visual
                    assert type(visual) == list
                    sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    batch_pil_images.append(list(visual))
                    batch_pil_sizes.append([v.size for v in visual if isinstance(v, PIL.Image.Image)])
                    task_type = "video"

                elif isinstance(visual[0], PIL.Image.Image):
                    task_type = "image"
                    if needs_image_gen:
                        pil_imgs = [self._pad_image_for_gen(v)[0] for v in visual]
                        orig_sizes = [self._pad_image_for_gen(v)[1] for v in visual]
                    else:
                        pil_imgs = list(visual)
                        orig_sizes = [v.size for v in visual]
                    batch_pil_images.append(pil_imgs)
                    batch_pil_sizes.append(orig_sizes)

                elif type(visual[0]) == str:
                    # video stored as file path
                    task_type = "video"
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                        frames_tensor = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        # Store as placeholder; video goes through legacy path below
                        batch_pil_images.append([])
                        batch_pil_sizes.append([])
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        batch_pil_images.append([])
                        batch_pil_sizes.append([])

            if needs_image_gen:
                for sample_idx, pil_list in enumerate(batch_pil_images):
                    if len(pil_list) == 0:
                        raise ValueError(
                            f"chat_mode='image_gen' requires image inputs for every sample. "
                            f"Missing image at batch index {sample_idx} (doc_id={batched_doc_id[sample_idx]})."
                        )

            # --- Extract text_gen_kwargs from gen_kwargs ---
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            schedule_kwargs = {}
            for key in list(gen_kwargs.keys()):
                if key.startswith('schedule__'):
                    value = gen_kwargs.pop(key)
                    schedule_kwargs[key.replace('schedule__', '')] = value
            if schedule_kwargs:
                gen_kwargs['schedule_kwargs'] = schedule_kwargs
            if 'block_length' not in gen_kwargs:
                gen_kwargs['block_length'] = min(128, gen_kwargs["max_new_tokens"])
            if 'step_per_block' not in gen_kwargs and 'step_ratio' not in gen_kwargs:
                gen_kwargs['step_per_block'] = gen_kwargs['block_length']

            text_gen_kwargs = {
                "max_new_tokens": gen_kwargs.get("max_new_tokens", 256),
                "block_length": gen_kwargs.get("block_length", 64),
                "step_per_block": gen_kwargs.get("step_per_block", 32),
            }
            if "schedule_kwargs" in gen_kwargs:
                text_gen_kwargs["schedule_kwargs"] = gen_kwargs["schedule_kwargs"]

            device = self.model.get_model().device
            reserve_id = 126089
            reserve_id2 = 126090
            reserve_token = "<|reserved_token_5|>"
            reserve_token_2 = "<|reserved_token_6|>"
            mask_id = 126336

            core_model = self.model.get_model()
            image_processor_gen = getattr(core_model, "image_processor_gen", None)
            if image_processor_gen is None and hasattr(core_model, "model"):
                image_processor_gen = getattr(core_model.model, "image_processor_gen", None)
            if image_processor_gen is None:
                raise AttributeError(
                    "Could not find `image_processor_gen` on model. "
                    "Expected `self.model.get_model().image_processor_gen` or `self.model.get_model().model.image_processor_gen`."
                )

            def _copy_conv():
                if "llama_3" in self.conv_template or "llada" in self.conv_template:
                    return copy.deepcopy(conv_templates[self.conv_template])
                return conv_templates[self.conv_template].copy()

            def _left_pad_2d(tensors, pad_value, dtype_):
                max_len = max(t.shape[1] for t in tensors)
                out = []
                for t in tensors:
                    pad_len = max_len - t.shape[1]
                    if pad_len > 0:
                        pad_t = torch.full((t.shape[0], pad_len), pad_value, dtype=dtype_, device=t.device)
                        t = torch.cat([pad_t, t], dim=1)
                    out.append(t)
                return torch.cat(out, dim=0), max_len

            def _left_pad_3d(tensors):
                max_len = max(t.shape[1] for t in tensors)
                out = []
                for t in tensors:
                    pad_len = max_len - t.shape[1]
                    if pad_len > 0:
                        pad_t = torch.zeros((t.shape[0], pad_len, t.shape[2]), dtype=t.dtype, device=t.device)
                        t = torch.cat([pad_t, t], dim=1)
                    out.append(t)
                return torch.cat(out, dim=0), max_len

            def _normalize_text_params():
                max_new_tokens = int(text_gen_kwargs.get("max_new_tokens", 256))
                block_length = int(text_gen_kwargs.get("block_length", 64))
                step_per_block = int(text_gen_kwargs.get("step_per_block", block_length))
                block_length = max(1, min(block_length, max_new_tokens))
                num_blocks = max(1, math.ceil(max_new_tokens / block_length))
                gen_length = num_blocks * block_length
                steps = max(1, step_per_block * num_blocks)
                return steps, gen_length, block_length

            def _to_image_tensor_list(sample_images):
                image_tensor = process_images(sample_images, self._image_processor, self.model.config)
                if isinstance(image_tensor, torch.Tensor):
                    return [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]
                return [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]

            def _prepare_text_batch(sample_texts, sample_images, append_image_tokens=True):
                all_input_ids = []
                all_attn = []
                all_embeds = []

                for idx, ctx in enumerate(sample_texts):
                    ctx = ctx or ""
                    img_list = sample_images[idx] if idx < len(sample_images) else []
                    if img_list is None:
                        img_list = []
                    if not isinstance(img_list, (list, tuple)):
                        img_list = [img_list]
                    img_list = [img for img in img_list if img is not None]

                    if append_image_tokens and len(img_list) > 0 and DEFAULT_IMAGE_TOKEN not in ctx:
                        image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(img_list))
                        prompts_input = image_tokens + "\n" + ctx
                    else:
                        prompts_input = ctx

                    conv = _copy_conv()
                    conv.append_message(conv.roles[0], prompts_input)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).to(device)

                    if len(img_list) > 0:
                        image_tensor_list = _to_image_tensor_list(img_list)
                        ids_attn = ids.ne(self.tokenizer.pad_token_id).long()
                        (_, _, attn, _, embeds, _, _) = self.model.prepare_inputs_labels_for_multimodal(
                            input_ids=ids,
                            position_ids=None,
                            attention_mask=ids_attn,
                            past_key_values=None,
                            labels=None,
                            images=image_tensor_list,
                            modalities=["image"],
                            image_sizes=[img.size for img in img_list],
                            return_inputs=True,
                        )
                        if attn is None:
                            attn = torch.ones(
                                (embeds.shape[0], embeds.shape[1]),
                                dtype=torch.long,
                                device=embeds.device,
                            )
                    else:
                        attn = ids.ne(self.tokenizer.pad_token_id).long()
                        embeds = self.model.get_model().embed_tokens(ids)

                    all_input_ids.append(ids)
                    all_attn.append(attn.to(device))
                    all_embeds.append(embeds)

                _input_ids, _ = _left_pad_2d(all_input_ids, self.tokenizer.pad_token_id, torch.long)
                attention_mask, _ = _left_pad_2d(all_attn, 0, torch.long)
                inputs_embeds, _ = _left_pad_3d(all_embeds)
                return _input_ids, attention_mask, inputs_embeds

            def _prepare_grounding_batch(sample_texts, sample_images):
                all_input_ids = []
                all_attn = []
                all_embeds = []
                all_mask_pos = []
                embed_lens = []

                for idx, ctx in enumerate(sample_texts):
                    ctx = ctx or ""
                    img_list = sample_images[idx] if idx < len(sample_images) else []
                    if not isinstance(img_list, (list, tuple)):
                        img_list = [img_list]
                    img_list = [img for img in img_list if img is not None]
                    if len(img_list) == 0:
                        raise ValueError("image_gen mode requires at least one image per sample.")

                    if DEFAULT_IMAGE_TOKEN not in ctx:
                        prompts_input = f"{DEFAULT_IMAGE_TOKEN}\n{ctx} Give bounding boxes in LOC format."
                    else:
                        prompts_input = f"{ctx} Give bounding boxes in LOC format."

                    conv = _copy_conv()
                    conv.append_message(conv.roles[0], prompts_input)
                    conv.append_message(
                        conv.roles[1],
                        "<LOC_BEGIN><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><LOC_END>",
                    )
                    prompt = conv.get_prompt()
                    ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).to(device)

                    image_tensor_list = _to_image_tensor_list(img_list)
                    ids_attn = ids.ne(self.tokenizer.pad_token_id).long()
                    (_, _, attn, _, embeds, _, _) = self.model.prepare_inputs_labels_for_multimodal(
                        input_ids=ids,
                        position_ids=None,
                        attention_mask=ids_attn,
                        past_key_values=None,
                        labels=None,
                        images=image_tensor_list,
                        modalities=["image"],
                        image_sizes=[img.size for img in img_list],
                        return_inputs=True,
                    )
                    if attn is None:
                        attn = torch.ones(
                            (embeds.shape[0], embeds.shape[1]),
                            dtype=torch.long,
                            device=embeds.device,
                        )

                    mask_cols = torch.where(ids[0] == mask_id)[0]
                    if len(mask_cols) >= 4:
                        mask_cols = mask_cols[:4]
                    else:
                        fill = torch.zeros(4 - len(mask_cols), dtype=torch.long, device=device)
                        mask_cols = torch.cat([mask_cols, fill], dim=0)
                    exp_pos = embeds.shape[1] - ids.shape[1] + mask_cols

                    all_input_ids.append(ids)
                    all_attn.append(attn.to(device))
                    all_embeds.append(embeds)
                    all_mask_pos.append(exp_pos)
                    embed_lens.append(embeds.shape[1])

                _input_ids, _ = _left_pad_2d(all_input_ids, self.tokenizer.pad_token_id, torch.long)
                attention_mask, _ = _left_pad_2d(all_attn, 0, torch.long)
                inputs_embeds, max_embed_len = _left_pad_3d(all_embeds)

                final_mask_pos = []
                for mp, emb_len in zip(all_mask_pos, embed_lens):
                    final_mask_pos.append(mp + (max_embed_len - emb_len))
                mask_pos = torch.stack(final_mask_pos, dim=0)
                return _input_ids, attention_mask, inputs_embeds, mask_pos

            def _prepare_image_edit_batch(sample_texts, sample_images, image_resolution):
                all_input_ids = []
                all_attn = []
                all_embeds = []
                all_is_gen = []
                all_is_gen_enc = []
                all_is_prompt = []

                for idx, ctx in enumerate(sample_texts):
                    ctx = ctx or ""
                    img = sample_images[idx]
                    if img is None:
                        raise ValueError("image_gen mode requires non-empty init image.")

                    image_tensor_list = _to_image_tensor_list([img])
                    image_for_enc = pad_to_square_and_resize(img.convert("RGB"), image_resolution)
                    vq_latents = image_processor_gen.preprocess(image_for_enc).to(device, self.model.dtype)
                    enc_latents, _gen_shape = self.model.encode_image_gen(vq_latents, enc=True)
                    enc_embeddings = self.model.get_model().call_gen_embedding(enc_latents, _gen_shape, enc=True)
                    gen_latents, gen_shape = self.model.encode_image_gen(vq_latents)
                    gen_embeddings = self.model.get_model().call_gen_embedding(gen_latents, gen_shape, enc=False)
                    n_tokens_txt = int(gen_embeddings.shape[1])

                    conv = _copy_conv()
                    conv.append_message(
                        conv.roles[0],
                        f"<image> {reserve_token_2 * enc_embeddings.shape[1]}\n {ctx} ",
                    )
                    conv.append_message(conv.roles[1], f"{reserve_token * n_tokens_txt}")
                    prompt = conv.get_prompt()
                    ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).to(device)

                    ids_attn = ids.ne(self.tokenizer.pad_token_id).long()
                    (_, _, attn, _, embeds, _, raw_input_ids) = self.model.prepare_inputs_labels_for_multimodal(
                        input_ids=ids,
                        position_ids=None,
                        attention_mask=ids_attn,
                        past_key_values=None,
                        labels=None,
                        images=image_tensor_list,
                        modalities=["image"],
                        image_sizes=[img.size],
                        return_inputs=True,
                    )
                    if attn is None:
                        attn = torch.ones(
                            (embeds.shape[0], embeds.shape[1]),
                            dtype=torch.long,
                            device=embeds.device,
                        )
                    embeds = embeds.to(enc_embeddings.dtype)
                    embeds[raw_input_ids == reserve_id2] = 0
                    enc_pad = pad_along_last_dim(enc_embeddings, embeds.shape[-1])
                    embeds[raw_input_ids == reserve_id2] = enc_pad.flatten(0, 1)

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

                    all_input_ids.append(ids)
                    all_attn.append(attn.to(device))
                    all_embeds.append(embeds)
                    all_is_gen.append(is_gen)
                    all_is_gen_enc.append(is_gen_enc)
                    all_is_prompt.append(is_prompt)

                _input_ids, _ = _left_pad_2d(all_input_ids, self.tokenizer.pad_token_id, torch.long)
                attention_mask, _ = _left_pad_2d(all_attn, 0, torch.long)
                inputs_embeds, _ = _left_pad_3d(all_embeds)
                is_gen, _ = _left_pad_2d(all_is_gen, 0, torch.bool)
                is_gen_enc, _ = _left_pad_2d(all_is_gen_enc, 0, torch.bool)
                is_prompt, _ = _left_pad_2d(all_is_prompt, 0, torch.bool)
                return inputs_embeds, attention_mask, is_gen, is_gen_enc, is_prompt

            def _parse_bbox_or_full(text, size_hw):
                w, h = size_hw
                loc_vals = re.findall(r"<LOC_([0-9]+)>", text or "")
                vals = None
                if len(loc_vals) >= 4:
                    vals = [float(v) for v in loc_vals[:4]]
                else:
                    num_vals = re.findall(r"-?\d+(?:\.\d+)?", text or "")
                    if len(num_vals) >= 4:
                        vals = [float(v) for v in num_vals[:4]]

                if vals is None:
                    return (0.0, 0.0, float(w), float(h))

                x0, y0, x1, y1 = vals
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                x0 = min(max(x0, 0.0), float(w))
                y0 = min(max(y0, 0.0), float(h))
                x1 = min(max(x1, 0.0), float(w))
                y1 = min(max(y1, 0.0), float(h))
                if x1 <= x0:
                    x1 = min(float(w), x0 + 1.0)
                if y1 <= y0:
                    y1 = min(float(h), y0 + 1.0)
                return (x0, y0, x1, y1)

            steps, gen_length, block_length = _normalize_text_params()

            # Forward img_gen_* params as kwargs so _generate_image picks them up
            img_gen_kwargs = dict(
                guidance_scale=self.img_gen_guidance_scale,
                guidance_scale_image=self.img_gen_guidance_scale_image,
                confidence_policy=self.img_gen_conf_policy,
                edit_mode=self.img_gen_edit_mode,
                n_steps=self.img_gen_n_steps,
                temperature=self.img_gen_temperature,
                enable_stratified=self.img_gen_enable_stratified,
                image_resolution=self.img_gen_resolution,
            )

            if self.chat_mode == "text_gen":
                _, text_attn, text_embeds = _prepare_text_batch(
                    list(batched_contexts), batch_pil_images, append_image_tokens=True
                )
                gen_result = self.model._generate_mode(
                    gen_type="text_gen",
                    tokenizer=self.tokenizer,
                    input_embeds=text_embeds,
                    attention_mask=text_attn,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=mask_id,
                    generation_batch_size=text_embeds.size(0),
                )
                completion_ids = gen_result["completion_ids"]
                text_outputs = [
                    txt.lstrip("!").strip()
                    for txt in self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                ]
                generated_images_list = [None] * batch_size
            else:
                # image_gen path
                base_images = []
                base_sizes = []
                for idx, img_list in enumerate(batch_pil_images):
                    if not isinstance(img_list, (list, tuple)) or len(img_list) == 0 or img_list[0] is None:
                        raise ValueError(
                            f"image_gen mode requires at least one image per sample. sample_idx={idx}"
                        )
                    if not isinstance(img_list[0], PIL.Image.Image):
                        raise ValueError(
                            f"image_gen mode expects PIL image inputs. sample_idx={idx}, got {type(img_list[0])}"
                        )
                    base_images.append(img_list[0].convert("RGB"))
                    if batch_pil_sizes and idx < len(batch_pil_sizes) and batch_pil_sizes[idx]:
                        base_sizes.append(batch_pil_sizes[idx][0])
                    else:
                        base_sizes.append(img_list[0].size)

                grounding_inputs = []
                for ctx in batched_contexts:
                    full_ctx = f"{ctx} {image_gen_post_prompt}".strip() if image_gen_post_prompt else (ctx or "")
                    grounding_inputs.append(full_ctx)

                _, grd_attn, grd_embeds, grd_mask_pos = _prepare_grounding_batch(
                    grounding_inputs, [[img] for img in base_images]
                )
                grd_bbox_mask = torch.zeros_like(grd_attn, dtype=torch.bool, device=grd_attn.device)
                grd_bbox_mask.scatter_(1, grd_mask_pos.clamp(min=0, max=grd_attn.shape[1] - 1), True)

                image_resolution = int(img_gen_kwargs.get("image_resolution", 1024))
                edit_embeds, edit_attn, is_gen, is_gen_enc, is_prompt = _prepare_image_edit_batch(
                    list(batched_contexts), base_images, image_resolution=image_resolution
                )

                def _bbox_postprocess_fn(bbox_ids, batch_indices):
                    bbox_texts = self.tokenizer.batch_decode(bbox_ids, skip_special_tokens=True)
                    size_slice = [base_sizes[idx] for idx in batch_indices]
                    pred_bboxes = [
                        _parse_bbox_or_full(txt, sz) for txt, sz in zip(bbox_texts, size_slice)
                    ]
                    return pred_bboxes, bbox_texts

                def _reencode_fn(batch_generation_prompts, batch_edited_images, _batch_start, _batch_end):
                    _, final_attn, final_embeds = _prepare_text_batch(
                        batch_generation_prompts,
                        batch_edited_images,
                        append_image_tokens=True,
                    )
                    return final_embeds, final_attn

                gen_result = self.model._generate_mode(
                    gen_type="image_gen",
                    tokenizer=self.tokenizer,
                    input_embeds=grd_embeds,
                    attention_mask=grd_attn,
                    bbox_mask=grd_bbox_mask,
                    input_embeds_gen=edit_embeds,
                    attention_mask_gen=edit_attn,
                    is_gen=is_gen,
                    is_gen_enc=is_gen_enc,
                    is_prompt=is_prompt,
                    init_images=base_images,
                    image_sizes=[(image_resolution, image_resolution)] * len(base_images),
                    generation_prompts=list(batched_contexts),
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=mask_id,
                    generation_batch_size=grd_embeds.size(0),
                    bbox_postprocess_fn=_bbox_postprocess_fn,
                    reencode_fn=_reencode_fn,
                    image_gen_kwargs=img_gen_kwargs,
                )
                completion_ids = gen_result["completion_ids"]
                text_outputs = [
                    txt.lstrip("!").strip()
                    for txt in self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                ]
                generated_images_list = gen_result.get("edited_images", [None] * batch_size)

            t1 = time.time()
            delta_t += t1 - t0
            num_generated += batch_size
            print(f"Avg Latency (of {num_generated}): {delta_t/num_generated}")

            # --- Save generated images and update doc metadata ---
            if needs_image_gen and generated_images_list:
                os.makedirs(self.gen_img_dir, exist_ok=True)
                for b_idx, gen_img in enumerate(generated_images_list):
                    if gen_img is None:
                        continue
                    task_name = batched_task[b_idx]
                    split_name = batched_split[b_idx]
                    doc_id = batched_doc_id[b_idx]
                    img_save_path = os.path.join(self.gen_img_dir, f"{task_name}_{doc_id}.png")
                    gen_img.save(img_save_path)
                    self.task_dict[task_name][split_name][doc_id]["gen_img_path"] = img_save_path
                    eval_logger.info(f"Saved gen image: {img_save_path}")
                    if self.img_gen_save_dir:
                        os.makedirs(self.img_gen_save_dir, exist_ok=True)
                        gen_img.save(os.path.join(self.img_gen_save_dir, f"gen_{task_name}_{doc_id}.png"))

            if DEBUG_PRINT_OUTPUT:
                for b_idx in range(len(batched_doc_id)):
                    task_name = batched_task[b_idx]
                    split_name = batched_split[b_idx]
                    doc_id = batched_doc_id[b_idx]
                    print(f'\n--------Start of Sample {doc_id}---------')
                    print("Answer: ", text_outputs[b_idx] if b_idx < len(text_outputs) else "N/A")
                    gen_path = self.task_dict[task_name][split_name][doc_id].get("gen_img_path", None)
                    if gen_path:
                        print("Gen Image: ", gen_path)
                    print('--------End---------')

            if needs_image_gen:
                for b_idx in range(len(text_outputs)):
                    task_name = batched_task[b_idx]
                    split_name = batched_split[b_idx]
                    doc_id = batched_doc_id[b_idx]
                    doc = self.task_dict[task_name][split_name][doc_id]
                    doc["_img_gen_input"] = ""
                    doc["_text_gen_input"] = ""

            res.extend(text_outputs)
            for b_ctx, b_output in zip(batched_contexts, text_outputs):
                self.cache_hook.add_partial("generate_until", (b_ctx, gen_kwargs), b_output)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
