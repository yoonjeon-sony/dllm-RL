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

import os
LOG_BATCH_LENGTH = os.environ.get('LOG_BATCH_LENGTH', False)
DEBUG_PRINT_IMAGE_RES = os.environ.get("DEBUG_PRINT_IMAGE_RES", False)
DO_DEBUG = os.environ.get('DO_DEBUG',False)
SKIP_COMPLEMENTARY_MASKING = os.environ.get("SKIP_COMPLEMENTARY_MASKING", False)
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.model.language_model.llada import *
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import logging

from torch.nn import CrossEntropyLoss

from llava.model.language_model.llada.modeling_llada import LLaDAModel,LLaDAModelLM,LLaDAConfig,create_model_config_from_pretrained_config
from llava.model.language_model.llada.generate import generate as llada_generate, generate_with_dual_cache, add_gumbel_noise, get_num_transfer_tokens, get_num_transfer_tokens_sch, cosine_schedule_2, logit_normal_schedule, exp_schedule
from llava.model.language_model.llada.log_likelyhood import get_log_likelihood as get_log_likelihood
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from model.language_model.llada.generate import get_logits, wte as llada_wte
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from accelerate.utils import reduce
from llava.model.utils import maybe_truncate_last_dim,pad_along_last_dim
from llava.constants import IGNORE_TEXT_LOSS,SKIP_DOWN_SAMPLE
import math
import time
import copy
import numpy as np
import torch.distributions as dists
import PIL
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
ENFORCE_NUM_ITEMIN_BATCH = os.environ.get("ENFORCE_NUM_ITEMIN_BATCH", False)
logger = logging.get_logger(__name__)

class LlavaLladaConfig(LLaDAConfig):
    model_type = "llava_llada"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}
    
    
class LlavaLladaModel(LlavaMetaModel,LLaDAModel):
    config_class = LlavaLladaConfig
    dtype = torch.bfloat16 # hack

    def __init__(self, pretrained_config,llada_config,init_params=None,vision_kwargs=None):
        # breakpoint()
        
        LLaDAModel.__init__(self, llada_config)
        LlavaMetaModel.__init__(self, pretrained_config,vision_kwargs=vision_kwargs,skip_init=True)
        
    def embed_tokens(self, x):
        return self.transformer.wte(x)

def sample_t(b,device,policy='uniform',policy_args=None):
    if policy == 'uniform':
        return torch.rand(b, device=device)
    elif policy == 'logit_normal':
        if policy_args is None:
            policy_args = dict(logit_mean=0.0,logit_std=1.0)
        u = torch.normal(mean=policy_args['logit_mean'], std=policy_args['logit_std'], size=(b,), device="cpu")
        u = torch.nn.functional.sigmoid(u).to(device=device)
        return u
    elif policy == 'cosine':
        timesteps = torch.rand(b, device=device)
        mask_prob = torch.cos(timesteps * math.pi * 0.5)
        mask_prob = mask_prob.clip(0,1)
        return mask_prob
    elif policy == "mode":
        u = torch.rand(size=(b,), device="cpu")
        u = 1 - u - policy_args['mode_scale'] * (torch.cos(torch.pi * u / 2) ** 2 - 1 + u)
        return u
        
def forward_process(bsz,seq_len,device, eps=1e-3,policy='uniform',policy_args=None):
    b, l = bsz,seq_len
    t = sample_t(b,device,policy=policy,policy_args=policy_args)
    # t = torch.sigmoid(t)
    p_mask = (1 - eps) * t + eps
    
    p_mask = p_mask[:, None]#.repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=device)
    mask_cutoff =  torch.max(p_mask,masked_indices.min(-1,keepdim=True).values)
    masked_indices = masked_indices <= mask_cutoff
    # mask at least one token
    # 126336 is used for [MASK] token
    #noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    return masked_indices, p_mask


def _gumbel_noise(t: torch.Tensor):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise))


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


class LlavaLladaForMaskedDiffusion(LLaDAModelLM,LlavaMetaForCausalLM):
    
    config_class = LlavaLladaConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False,vision_kwargs=None,prefix_lm=False,**kwargs):
        if not hasattr(config,'d_model_gen') or config.d_model_gen < 0 :
            config.d_model_gen = config.d_model
        if not hasattr(config,'mlp_hidden_size_gen') or config.mlp_hidden_size_gen < 0:
            config.mlp_hidden_size_gen = config.mlp_hidden_size
        if not hasattr(config,'downsample'):
            config.downsample = False
        LLaDAModelLM.__init__(self, config,model,init_params)
        # hack

        # configure default generation settings
        config.model_type = "llava_llada"
        # config.rope_scaling = None
        self.prefix_lm = prefix_lm

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LlavaLladaModel(config,model_config, init_params=init_params,vision_kwargs=vision_kwargs)
        else:
            self.model = model
        self.model.set_activation_checkpointing('whole_layer')
        
        self.post_init() # TODO
        # self.eos_id = 126081 # hack
        # self.mask_id = 126336 # hack
        
    def get_model(self):
        return self.model

    def pad_image(self, pil_image: PIL.Image.Image, image_resolution: int = 1024):
        if not isinstance(pil_image, PIL.Image.Image):
            raise TypeError(f"pad_image expects PIL.Image.Image, got {type(pil_image)}")
        original_size = pil_image.size
        padded = ImageOps.pad(
            pil_image.convert("RGB"),
            (max(original_size), max(original_size)),
            color=(0, 0, 0),
        )
        resized = padded.resize((image_resolution, image_resolution))
        return resized, original_size

    @torch.no_grad()
    def _generate_mode(
        self,
        *,
        gen_type: str,
        tokenizer,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox_mask: Optional[torch.Tensor] = None,
        input_embeds_gen: Optional[torch.Tensor] = None,
        inputs_embeds_cond: Optional[torch.Tensor] = None,
        inputs_embeds_uncond: Optional[torch.Tensor] = None,
        inputs_embeds_uncond_enc: Optional[torch.Tensor] = None,
        attention_mask_gen: Optional[torch.Tensor] = None,
        is_gen: Optional[torch.Tensor] = None,
        is_gen_enc: Optional[torch.Tensor] = None,
        is_gen_enc_null: Optional[torch.Tensor] = None,
        is_gen_enc_ccc: Optional[torch.Tensor] = None,
        is_prompt: Optional[torch.Tensor] = None,
        init_images: Optional[list] = None,
        image_sizes: Optional[list] = None,
        generation_prompts: Optional[list] = None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        generation_batch_size: Optional[int] = None,
        bbox_postprocess_fn=None,
        reencode_fn=None,
        image_gen_kwargs: Optional[dict] = None,
        return_debug: bool = False,
    ):
        if gen_type not in ("text_gen", "image_gen", "grounding"):
            raise ValueError(f"gen_type must be 'text_gen', 'image_gen', or 'grounding', got {gen_type!r}")
        if attention_mask is None or input_embeds is None:
            raise ValueError("input_embeds and attention_mask are required.")
        if gen_type in {"grounding", "image_gen"} and bbox_mask is None:
            raise ValueError("bbox_mask is required for grounding and image_gen modes.")
        if gen_type == "image_gen":
            missing = [
                name
                for name, val in [
                    ("input_embeds_gen", input_embeds_gen),
                    ("attention_mask_gen", attention_mask_gen),
                    ("is_gen", is_gen),
                    ("is_gen_enc", is_gen_enc),
                    ("is_prompt", is_prompt),
                    ("init_images", init_images),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(f"image_gen mode missing required inputs: {', '.join(missing)}")
            if reencode_fn is None:
                raise ValueError("reencode_fn is required for image_gen mode.")
        if gen_type in {"grounding", "image_gen"} and bbox_postprocess_fn is None:
            raise ValueError("bbox_postprocess_fn is required for grounding/image_gen modes.")

        if generation_batch_size is None:
            generation_batch_size = input_embeds.size(0)

        base_model = self.get_model()
        image_gen_kwargs = dict(image_gen_kwargs or {})
        image_gen_kwargs.pop("return_debug", None)

        total = input_embeds.size(0)
        prompt_completion_ids_all = []
        prompt_masks_all = []
        bbox_ids_all = []
        pred_bboxes_all = [None] * total if gen_type in {"grounding", "image_gen"} else None
        bbox_texts_all = [None] * total if gen_type in {"grounding", "image_gen"} else None
        edited_images_all = [None] * total if gen_type == "image_gen" else None
        image_gen_debug_all = [None] * total if gen_type == "image_gen" else None

        for i in range(0, total, generation_batch_size):
            end_idx = min(i + generation_batch_size, total)
            batch_input_embeds = input_embeds[i:end_idx]
            batch_attention_mask = attention_mask[i:end_idx]
            batch_pred_bboxes = None
            batch_bbox_texts = None
            batch_bbox_ids = None

            if gen_type in {"grounding", "image_gen"}:
                batch_bbox_mask = bbox_mask[i:end_idx]
                logits = get_logits(base_model, batch_input_embeds)
                token_ids = logits.argmax(dim=-1)
                eos_id = tokenizer.eos_token_id
                row_ids_list = []
                for row_idx in range(token_ids.size(0)):
                    row_ids = token_ids[row_idx][batch_bbox_mask[row_idx]]
                    row_ids = row_ids[:4]
                    if row_ids.numel() < 4:
                        pad = torch.full(
                            (4 - row_ids.numel(),),
                            eos_id,
                            dtype=torch.long,
                            device=row_ids.device,
                        )
                        row_ids = torch.cat([row_ids, pad], dim=0)
                    row_ids_list.append(row_ids.to(torch.long))
                batch_bbox_ids = torch.stack(row_ids_list, dim=0)
                bbox_ids_all.append(batch_bbox_ids)

                post = bbox_postprocess_fn(batch_bbox_ids, list(range(i, end_idx)))
                if isinstance(post, tuple):
                    batch_pred_bboxes = post[0]
                    if len(post) > 1:
                        batch_bbox_texts = post[1]
                else:
                    batch_pred_bboxes = post

                if batch_pred_bboxes is not None:
                    for row_idx, pred_box in enumerate(batch_pred_bboxes):
                        pred_bboxes_all[i + row_idx] = pred_box
                if batch_bbox_texts is not None:
                    for row_idx, txt in enumerate(batch_bbox_texts):
                        bbox_texts_all[i + row_idx] = txt

            if gen_type == "image_gen":
                batch_images = init_images[i:end_idx]
                batch_image_sizes = image_sizes[i:end_idx] if image_sizes is not None else None
                batch_input_embeds_gen = input_embeds_gen[i:end_idx]
                batch_inputs_embeds_cond = inputs_embeds_cond[i:end_idx] if inputs_embeds_cond is not None else None
                batch_inputs_embeds_uncond = inputs_embeds_uncond[i:end_idx] if inputs_embeds_uncond is not None else None
                batch_inputs_embeds_uncond_enc = (
                    inputs_embeds_uncond_enc[i:end_idx] if inputs_embeds_uncond_enc is not None else None
                )
                batch_attention_mask_gen = attention_mask_gen[i:end_idx]
                batch_is_gen = is_gen[i:end_idx]
                batch_is_gen_enc = is_gen_enc[i:end_idx]
                batch_is_gen_enc_null = is_gen_enc_null[i:end_idx] if is_gen_enc_null is not None else None
                batch_is_gen_enc_ccc = is_gen_enc_ccc[i:end_idx] if is_gen_enc_ccc is not None else None
                batch_is_prompt = is_prompt[i:end_idx]

                if return_debug:
                    batch_edited_images, batch_image_debug = self.generate_image(
                        init_images=batch_images,
                        inputs_embeds=batch_input_embeds_gen,
                        inputs_embeds_cond=batch_inputs_embeds_cond,
                        inputs_embeds_uncond=batch_inputs_embeds_uncond,
                        inputs_embeds_uncond_enc=batch_inputs_embeds_uncond_enc,
                        is_gen=batch_is_gen,
                        is_gen_enc=batch_is_gen_enc,
                        is_gen_enc_null=batch_is_gen_enc_null,
                        is_gen_enc_ccc=batch_is_gen_enc_ccc,
                        is_prompt=batch_is_prompt,
                        attention_mask=batch_attention_mask_gen,
                        pred_bboxes=batch_pred_bboxes,
                        image_sizes=batch_image_sizes,
                        return_debug=True,
                        **image_gen_kwargs,
                    )
                else:
                    batch_edited_images = self.generate_image(
                        init_images=batch_images,
                        inputs_embeds=batch_input_embeds_gen,
                        inputs_embeds_cond=batch_inputs_embeds_cond,
                        inputs_embeds_uncond=batch_inputs_embeds_uncond,
                        inputs_embeds_uncond_enc=batch_inputs_embeds_uncond_enc,
                        is_gen=batch_is_gen,
                        is_gen_enc=batch_is_gen_enc,
                        is_gen_enc_null=batch_is_gen_enc_null,
                        is_gen_enc_ccc=batch_is_gen_enc_ccc,
                        is_prompt=batch_is_prompt,
                        attention_mask=batch_attention_mask_gen,
                        pred_bboxes=batch_pred_bboxes,
                        image_sizes=batch_image_sizes,
                        **image_gen_kwargs,
                    )
                    batch_image_debug = None

                for row_idx, edited_image in enumerate(batch_edited_images):
                    edited_images_all[i + row_idx] = edited_image

                if generation_prompts is None:
                    raise ValueError("generation_prompts is required for image_gen mode.")
                batch_generation_prompts = generation_prompts[i:end_idx]
                batch_all_images = [[orig, edited] for orig, edited in zip(batch_images, batch_edited_images)]
                # TODO: Append the edited image into conv.roles[1] (assistant) and continue text generation from this prompt.
                """ 
                    conv.append_message(conv.roles[0], f"<image> {reserve_token_2*enc_embeddings.shape[1]}\n {prompt} ")
                    reserve_token = '<|reserved_token_5|>'
                    conv.append_message(conv.roles[1], f"{plan}{reserve_token*n_tokens_txt}")
                    prompt_question = conv.get_prompt()
                    prompt_question.removesuffix('<|start_header_id|>assistant<|end_header_id|>\n\n') # TODO: Should remove this line to continue text gen.
                    print(prompt_question.replace('<|reserved_token_5|>','*').replace('<|reserved_token_6|>','*'))
                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                """
                batch_input_embeds, batch_attention_mask = reencode_fn(
                    batch_generation_prompts,
                    batch_all_images,
                    i,
                    end_idx,
                )
                batch_prompt_completion_ids = self.generate_text(
                    prompt=None,
                    inputs_embeds=batch_input_embeds,
                    attention_mask=batch_attention_mask,
                    position_ids=None,
                    tokenizer=tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    mask_id=mask_id,
                    t2i_inference=False,
                    do_sample=False,
                    prefix_lm=True,
                )
                batch_prompt_mask = batch_attention_mask
            elif gen_type == "grounding":
                batch_prompt_completion_ids = batch_bbox_ids
                batch_prompt_mask = batch_attention_mask
            else:
                batch_prompt_completion_ids = self.generate_text(
                    prompt=None,
                    inputs_embeds=batch_input_embeds,
                    attention_mask=batch_attention_mask,
                    position_ids=None,
                    tokenizer=tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    mask_id=mask_id,
                    t2i_inference=False,
                    do_sample=False,
                    prefix_lm=True,
                )
                batch_prompt_mask = batch_attention_mask

            prompt_completion_ids_all.append(batch_prompt_completion_ids)
            prompt_masks_all.append(batch_prompt_mask)
            del batch_prompt_completion_ids
            torch.cuda.empty_cache()

        completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
        max_prompt_len = max(m.shape[1] for m in prompt_masks_all)
        padded_prompt_masks = []
        for m in prompt_masks_all:
            pad_len = max_prompt_len - m.shape[1]
            if pad_len > 0:
                pad_m = torch.zeros((m.shape[0], pad_len), dtype=m.dtype, device=m.device)
                m = torch.cat([pad_m, m], dim=1)
            padded_prompt_masks.append(m)
        prompt_mask = torch.cat(padded_prompt_masks, dim=0)

        result = {
            "completion_ids": completion_ids,
            "prompt_mask": prompt_mask,
        }
        if gen_type in {"grounding", "image_gen"}:
            result["bbox_ids"] = torch.cat(bbox_ids_all, dim=0) if bbox_ids_all else None
            result["pred_bboxes"] = pred_bboxes_all
            if any(v is not None for v in bbox_texts_all):
                result["bbox_texts"] = bbox_texts_all
        if gen_type == "image_gen":
            result["edited_images"] = edited_images_all
            if return_debug:
                result["image_gen_debug"] = image_gen_debug_all
        return result
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        policy='uniform',
        policy_args=None,
        images_gen = None,
        gen_latents = None,
        t2i_inference=False,
        gen_shape=None,
        images_gen_enc = None,
        gen_latents_enc = None,
        image_gen_weight =None,
        dataset_name=None,
        do_not_mask_text=None,
        # **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(dataset_name)
        reserve_id = 126089 # reserve , used in training
        reserve_id_enc = 126090 # reserved_token_6 reserve , used in training
        OFFSET = 150000 # used in inference

        if images_gen is not None or gen_latents is not None:
            if gen_latents is None:
                images_gen = torch.cat([torch.cat(x) for x in images_gen])
                gen_latents, gen_shape = self.encode_image_gen(images_gen)
            
            token_mask = (input_ids==-300)

        eos_id = 126081 # hack
        mask_id = 126336
        img_mask_id =8193
        fim_id = 126085
        raw_inputs_ids = input_ids
        if input_ids is not None:
            input_ids[input_ids==-300] = reserve_id
            attention_mask_raw = attention_mask.clone()
            non_padding = ~(raw_inputs_ids==eos_id)
            attention_mask[raw_inputs_ids==eos_id] = True # no sequence attention mask per Sec B.1
            labels[raw_inputs_ids==eos_id] = eos_id # revert target
        # fix attention mask
        # pad_len = torch.randint(0,pad_len_max,(1,)).item()
        # padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device)
        skip_batch = 0
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, new_input_ids) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,return_inputs=True)
        else:
            # inputs_embeds are pre-computed; create a dummy new_input_ids with no special tokens
            new_input_ids = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device)
        assert input_ids is None
        # print(dataset_name,new_input_ids.shape[-1],new_input_ids.shape[0])
        new_token_mask:torch.Tensor = (new_input_ids == reserve_id)
        new_token_mask_enc: torch.Tensor = (new_input_ids ==  reserve_id_enc)
        if images_gen_enc is not None or gen_latents_enc is not None:
            if gen_latents_enc is None:
                images_gen_enc = torch.cat([torch.cat(x) for x in images_gen_enc])
                gen_latents_enc,gen_shape_enc = self.encode_image_gen(images_gen_enc,enc=True)
        
            gen_latents_enc_embeds = self.get_model().call_gen_embedding(gen_latents_enc,gen_shape=gen_shape_enc,enc=True)  
            gen_latents_enc_embeds = pad_along_last_dim(gen_latents_enc_embeds,self.config.d_model)
            
            if not new_token_mask_enc.sum() == gen_latents_enc_embeds.shape[0]*gen_latents_enc_embeds.shape[1]:
                skip_batch = 1
                print(f"SKIP BATCH!!! {dataset_name}")
            else:
                inputs_embeds_replaced = inputs_embeds.masked_scatter(new_token_mask_enc.unsqueeze(-1), gen_latents_enc_embeds.view(-1,4096))
                inputs_embeds = inputs_embeds_replaced
        do_inv = not SKIP_COMPLEMENTARY_MASKING
        if image_gen_weight is not None:
            image_gen_weight =torch.cat([torch.cat(x) for x in image_gen_weight])
        image_gen_weight_dup = image_gen_weight
        if do_inv:
            new_token_mask_dup = new_token_mask.repeat(2,1)
            new_token_mask_enc_dup = new_token_mask_enc.repeat(2,1)
            if image_gen_weight_dup is not None:
                image_gen_weight_dup = image_gen_weight_dup.repeat(2,1,1,1)
        else:
            new_token_mask_dup = new_token_mask
            new_token_mask_enc_dup = new_token_mask_enc
        modality_indices = new_token_mask_dup
        enc_use_image_branch = getattr(self.get_model().config,'enc_use_image_branch',False)
        if enc_use_image_branch:
            modality_indices = modality_indices| new_token_mask_enc_dup
        
        prompt_len = None
        
        if labels is not None:
            assert labels.min() == -100
            labels_mask = ~(labels == -100) # targets mask
            infill_token_pos = labels==fim_id
            # find index of the first non zero mask
            # labels_mask = labels_mask.cumsum(-1).eq(1)
            if self.prefix_lm:
                # breakpoint()
                prompt_len = labels_mask.float().argmax(dim=1)
                # print(prompt_len)
            noise_embeddings = self.get_model().transformer.wte(torch.tensor([mask_id]).to(raw_inputs_ids))
            # noise_embeddings is 1, 4096
            bsz,seq_len = labels_mask.shape
            noise_embeddings = noise_embeddings.view(1,1,-1)#.repeat(bsz,seq_len,1)
            # t = torch.rand(b, device=input_ids.device)
            masked_indices, p_mask = forward_process(bsz,seq_len,raw_inputs_ids.device,policy=policy,policy_args=policy_args)
            # torch.where()
            #breakpoint()
            prompt_drop_rate = getattr(self.config,'prompt_drop_rate',0)
            rand_drop = (torch.rand(labels_mask.shape[0], device=labels_mask.device) <prompt_drop_rate).view(-1,1)
            
            is_prompt = (~labels_mask) & (~new_token_mask)  # not text answer and not imag answer
            prompt_to_mask = is_prompt & rand_drop
            if images_gen_enc is not None:
                prompt_drop_rate_enc = getattr(self.config,'image_enc_drop_rate',0.5)
                rand_drop_enc = (torch.rand(labels_mask.shape[0], device=labels_mask.device) <prompt_drop_rate_enc).view(-1,1)
                image_enc_to_mask = rand_drop_enc & new_token_mask_enc_dup
                prompt_to_mask = prompt_to_mask & (~new_token_mask_enc_dup)
                prompt_to_mask = prompt_to_mask | image_enc_to_mask
                modality_indices = modality_indices & (~prompt_to_mask)
                # breakpoint()
                # N x 1
            #breakpoint()
            if do_not_mask_text is not None and sum(do_not_mask_text) > 0:
                do_not_mask = torch.tensor(do_not_mask_text)
                do_not_mask = do_not_mask & (torch.rand_like(do_not_mask,dtype=torch.float) < 0.8)
                do_not_mask = do_not_mask.unsqueeze(-1).to(masked_indices)
            else:
                do_not_mask = torch.zeros((masked_indices.shape[0],1),dtype=torch.bool).to(masked_indices)
            final_masked_indices = masked_indices & (~do_not_mask) &labels_mask & (~infill_token_pos) 
            final_masked_indices = final_masked_indices | prompt_to_mask
            
            if do_inv:
                final_masked_indices_inv = (~masked_indices) & (~do_not_mask) &labels_mask & (~infill_token_pos)
                final_masked_indices_inv = final_masked_indices_inv | prompt_to_mask
                inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)

            
            inputs_embeds = torch.where(final_masked_indices.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)

            if do_inv:
                labels_inv = labels.clone()
                labels_inv[~final_masked_indices_inv] = -100
            labels[~final_masked_indices] = -100
            labels[labels==fim_id] = -100 # kill infill token so we don't predict it
            is_unitok = 'unitok' in getattr(self.get_model().config,'mm_vqvae','')
            is_unitok_submask = getattr(self.get_model().config,'mm_submask',False)
            
            if do_inv:
                inputs_embeds = torch.cat([inputs_embeds,inputs_embeds_inv])
            if images_gen is not None:
                gen_latents_masked = gen_latents.clone()
                if is_unitok_submask:
                    # gen_latents is N 8 L 
                    # _latents_b, _, _latents_l = gen_latents.shape
                    # gen_latents = gen_latents,view()
                    masked_indices_gen, p_mask = forward_process(gen_latents_masked.shape[0],gen_latents.shape[-1]*8,raw_inputs_ids.device,policy=policy,policy_args=policy_args)
                    masked_indices_gen = masked_indices_gen.view(-1,8,gen_latents.shape[-1])
                else:
                    masked_indices_gen, p_mask = forward_process(gen_latents_masked.shape[0],gen_latents.shape[-1],raw_inputs_ids.device,policy=policy,policy_args=policy_args)
                    if is_unitok:
                        masked_indices_gen = masked_indices_gen.unsqueeze(1).repeat(1,8,1) # N 8
                        
                gen_latents_masked[masked_indices_gen] = img_mask_id
                if do_inv:
                    gen_latents_masked_inv = gen_latents.clone()
                    gen_latents_masked_inv[~masked_indices_gen] = img_mask_id
                    gen_latents_comp = torch.cat([gen_latents_masked,gen_latents_masked_inv])
                else:
                    gen_latents_comp = gen_latents_masked
                gen_latents_comp_embeds = self.get_model().call_gen_embedding(gen_latents_comp,gen_shape=gen_shape)  
                gen_latents_comp_embeds = pad_along_last_dim(gen_latents_comp_embeds,self.config.d_model)
                if do_inv:
                    gen_latents_comp_labels = torch.cat([gen_latents,gen_latents])
                else:
                    gen_latents_comp_labels = gen_latents.clone()
                    
                gen_latents_comp_labels_raw = gen_latents_comp_labels.clone().detach()
                gen_latents_comp_labels[~(gen_latents_comp==img_mask_id)] = -100
                gen_latents_comp_labels_is_mask = gen_latents_comp==img_mask_id
                
                if not new_token_mask_dup.sum() == gen_latents_comp_embeds.shape[0]*gen_latents_comp_embeds.shape[1]:
                    skip_batch = 1
                    print(f"SKIP BATCH!!! {dataset_name}")
                else:
                    inputs_embeds_replaced = inputs_embeds.masked_scatter(new_token_mask_dup.unsqueeze(-1), gen_latents_comp_embeds.view(-1,4096))
                    inputs_embeds = inputs_embeds_replaced
            
            if do_inv:
                labels =  torch.cat([labels,labels_inv])
                if self.prefix_lm:
                    prompt_len = prompt_len.repeat(2,1)
                final_masked_indices = torch.cat([final_masked_indices,final_masked_indices_inv])
            seq_len = labels.shape[-1]
            # print(seq_len)
            if LOG_BATCH_LENGTH:
                print("Batch Length",seq_len)
            if images_gen is not None:
                n_image_tokens = (
                    (new_input_ids == -200).sum(dim=-1).max().item()
                    + new_token_mask.sum(dim=-1).max().item()
                    + new_token_mask_enc.sum(dim=-1).max().item()
                )
                max_seq_len = 1024 + n_image_tokens
            else:
                max_seq_len = getattr(self.config, 'tokenizer_model_max_length', 8192)
            if seq_len > max_seq_len:
                labels = labels[:,:max_seq_len]
                inputs_embeds = inputs_embeds[:,:max_seq_len]
                attention_mask = attention_mask[:,:max_seq_len] if attention_mask is not None else None
                if position_ids is not None:
                    position_ids = position_ids[:,:max_seq_len]
                new_token_mask_dup = new_token_mask_dup[:,:max_seq_len]
                new_token_mask_enc_dup = new_token_mask_enc_dup[:,:max_seq_len]
                modality_indices = modality_indices[:,:max_seq_len]
                final_masked_indices = final_masked_indices[:,:max_seq_len]
                new_input_ids = new_input_ids[:,:max_seq_len]
                if prompt_len is not None:
                    prompt_len = prompt_len.clamp(max=max_seq_len)
                # If truncation removed gen tokens, skip gen loss for this batch
                if images_gen is not None:
                    expected_gen = gen_latents_comp_embeds.shape[0] * gen_latents_comp_embeds.shape[1] if not skip_batch else 0
                    if new_token_mask_dup.sum() != expected_gen:
                        skip_batch = 1
                assert input_ids is None
                assert past_key_values is None

        if dpo_forward:
            raise NotImplementedError() 
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            #assert attention_mask is None or torch.all(attention_mask)
            attention_mask = None
            num_items_in_batch = None
            if ENFORCE_NUM_ITEMIN_BATCH:
                num_items_in_batch = labels.ne(-100).float().sum()
                num_items_in_batch = reduce(num_items_in_batch)
                num_items_in_batch = num_items_in_batch.long()
            output_hidden_states = images_gen is not None or t2i_inference
            output =  super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                prompt_len=prompt_len,
                num_items_in_batch=num_items_in_batch,
                modality_indices= modality_indices
            )
            if skip_batch:
                print(f"SKIP BATCH!!! {dataset_name}")
            if images_gen is not None and not skip_batch:
                hidden_states = output.hidden_states[-1]
                gen_hidden_states = hidden_states[new_token_mask_dup]
                gen_hidden_states = maybe_truncate_last_dim(gen_hidden_states,self.config.d_model_gen)
                timesteps = gen_latents_comp_labels_is_mask.sum(-1) / gen_latents_comp_labels_is_mask.shape[-1]
                gen_logits = self.get_model().call_gen_predictor(gen_hidden_states,gen_shape,timesteps)
                # (B L, 8, V)
                #breakpoint()
                gen_targets = gen_latents_comp_labels

                if is_unitok:
                    _b,_d,_l = gen_latents_comp_labels_is_mask.shape
                    gen_logits = gen_logits.view(_b,_l,*gen_logits.shape[-2:]) # B L 8 2064
                    gen_logits = gen_logits.permute(0,2,1,3) # B 8 4096 2064
                    _loss_mask = gen_latents_comp_labels_is_mask.flatten()
                    gen_loss = torch.nn.functional.cross_entropy(gen_logits.flatten(0,2)[_loss_mask],gen_latents_comp_labels.flatten()[_loss_mask])
                else:
                    if image_gen_weight_dup is not None and image_gen_weight_dup.max() > 1:
                        _loss_mask = gen_latents_comp_labels_is_mask.flatten()
                        gen_loss = torch.nn.functional.cross_entropy(gen_logits[_loss_mask].float(),gen_latents_comp_labels.flatten()[_loss_mask],reduction='none') # 5170
                        image_gen_weight_dup_flt = image_gen_weight_dup.flatten(1,) # B L
                        image_gen_weight_dup_flt = image_gen_weight_dup_flt  * gen_latents_comp_labels_is_mask
                        factor_per_sample = gen_latents_comp_labels_is_mask.sum(-1,keepdims=True) #/ (gen_latents_comp_labels_is_mask.sum()+1e-7)
                        loss_weight = image_gen_weight_dup_flt.float()  * (factor_per_sample / image_gen_weight_dup_flt.float().sum(-1,keepdims=True) )
                        loss_weight = loss_weight.flatten()[_loss_mask].to(gen_loss.dtype)
                        gen_loss = (gen_loss * loss_weight).mean()
                    else:
                        _loss_mask = gen_latents_comp_labels_is_mask.flatten()
                        gen_loss = torch.nn.functional.cross_entropy(gen_logits[_loss_mask].float(),gen_targets.flatten()[_loss_mask])
                und_loss = output.loss
                if torch.isnan(und_loss):
                    und_loss = output.logits.mean() * 0
                if IGNORE_TEXT_LOSS:
                    und_loss = und_loss * 0
                    output.loss = gen_loss 
                else:
                    output.loss = und_loss + gen_loss 
                
                output['und_loss'] = und_loss.detach()
                output['gen_loss'] = gen_loss.detach()
                x_0 = gen_logits.argmax(-1)
                x_0 = x_0.view(gen_targets.shape)
                output['gen_x0_gt'] = gen_latents_comp_labels_raw.detach()
                output['gen_x_0_pred'] = x_0.detach()
                output['gen_x_mask'] = gen_latents_comp_labels_is_mask.detach()
                output['new_token_mask_dup'] = new_token_mask_dup.detach()
            elif t2i_inference:
                hidden_states = output.hidden_states[-1]
                gen_hidden_states = hidden_states[new_token_mask_dup]
                gen_logits = self.get_model().gen_predictor(gen_hidden_states)
                #output['gen_logits'] = gen_logits
                final_logits = torch.zeros(*hidden_states.shape[:-1],OFFSET+gen_logits.shape[-1])
                final_logits = final_logits + float('-inf')
                final_logits[...,:output.logits.shape[-1]] = output.logits
                final_logits[...,OFFSET:] = gen_logits
                # breakpoint()
            
            if do_inv:
                new_input_ids = new_input_ids.repeat(2,1)
            output['new_input_ids']=new_input_ids
            output['labels'] = labels
            output['final_masked_indices']=final_masked_indices
            output['p_mask'] = p_mask
            output['do_inv'] = do_inv
            output['skip_batch'] = skip_batch
            
            return output

    def pad_sequence(self, tokenizer, input_ids, batch_first, padding_value):
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @torch.no_grad()
    def generate_image(
        self,
        init_images,
        inputs_embeds,
        is_gen,
        is_gen_enc,
        is_prompt,
        attention_mask,
        inputs_embeds_cond=None,
        inputs_embeds_uncond=None,
        inputs_embeds_uncond_enc=None,
        is_gen_enc_null=None,
        is_gen_enc_ccc=None,
        pred_bboxes: List[tuple]=None,
        **kwargs,
    ):
        return_debug = bool(kwargs.get("return_debug", False))
        debug_label = kwargs.get("debug_label", "image_gen")
        model = self.get_model()
        device = inputs_embeds.device
        img_mask_id = 8193
        image_processor_gen = getattr(model, "image_processor_gen", None)
        if image_processor_gen is None and hasattr(model, "model"):
            image_processor_gen = getattr(model.model, "image_processor_gen", None)
        if image_processor_gen is None:
            raise AttributeError(
                "Could not find `image_processor_gen` on model. "
                "Expected `self.get_model().image_processor_gen` or `self.get_model().model.image_processor_gen`."
            )

        sample_policy = kwargs.get("sample_policy", "multinomial")
        confidence_policy = kwargs.get("confidence_policy", "stratified")
        enable_stratified = kwargs.get("enable_stratified", True)
        guidance_scale = float(kwargs.get("guidance_scale", 0.0))
        guidance_scale_image = float(kwargs.get("guidance_scale_image", 0.0))
        edit_mode = int(kwargs.get("edit_mode", 0))
        image_resolution = int(kwargs.get("image_resolution", 1024))
        n_tokens = int(kwargs.get("n_tokens", 4096))
        shift = int(kwargs.get("shift", 5))
        n_steps = int(kwargs.get("n_steps", 64))
        schedule = kwargs.get("schedule", "shift")
        alg_temp = float(kwargs.get("alg_temp", 5.0))
        dynamic_temperature = bool(kwargs.get("dynamic_temperature", True))
        temperature = float(kwargs.get("temperature", 0.8))
        schedule_temp = kwargs.get("schedule_temp", "cosine2")
        min_temperature = float(kwargs.get("min_temperature", 0.5))
        schedule_temp_samp = kwargs.get("schedule_temp_samp", "linear")
        dynamic_temperature_samp = bool(kwargs.get("dynamic_temperature_samp", False))
        min_temperature_samp = float(kwargs.get("min_temperature_samp", 1.0))
        cfg_interval = kwargs.get("cfg_interval", [0, 1])
        order_cutoff = float(kwargs.get("order_cutoff", 100.0))
        top_p = kwargs.get("top_p", None)
        top_k = kwargs.get("top_k", None)
        use_3d = bool(kwargs.get("use_3d", False))
        shift_alg = kwargs.get("shift_alg", shift)

        norm_init_images = []
        for x in init_images:
            img = x[0] if isinstance(x, (list, tuple)) else x
            if img is None:
                raise ValueError("Found None image in `init_images`.")
            if not isinstance(img, PIL.Image.Image):
                raise ValueError(f"init_images must contain PIL images; got {type(img)}")
            norm_init_images.append(img.convert("RGB"))

        batch_size = len(norm_init_images)
        if batch_size == 0:
            return []

        gen_shape_map = {1024: (64, 64), 512: (32, 32), 256: (16, 16)}
        has_custom_gen_shape = "gen_shape" in kwargs and kwargs["gen_shape"] is not None
        if has_custom_gen_shape:
            gen_shape = tuple(kwargs["gen_shape"])
        else:
            if image_resolution in gen_shape_map:
                gen_shape = gen_shape_map[image_resolution]
            else:
                side = int(round(math.sqrt(n_tokens)))
                gen_shape = (side, side)
        grid_h, grid_w = gen_shape
        grid_tokens = grid_h * grid_w
        image_sizes = kwargs.get("image_sizes", [img.size for img in norm_init_images])
        image_sizes = [
            tuple(sz) if isinstance(sz, (list, tuple)) and len(sz) >= 2 else None
            for sz in image_sizes
        ]

        if image_resolution not in gen_shape_map and not has_custom_gen_shape:
            raise ValueError(
                f"image_gen expects image_resolution in {sorted(gen_shape_map)}, got {image_resolution}."
            )
        for b, sz in enumerate(image_sizes):
            if sz is None:
                raise ValueError(f"image_sizes[{b}] is missing.")
            if int(sz[0]) != image_resolution or int(sz[1]) != image_resolution:
                raise ValueError(
                    f"image_gen expects image_sizes[{b}] == ({image_resolution}, {image_resolution}), got {sz}."
                )

        # preprocess can fail on heterogenous inputs; fallback to per-sample and concat.
        try:
            vq_latents = image_processor_gen.preprocess(norm_init_images).to(device, model.dtype)
        except Exception:
            vq_latents = torch.cat(
                [image_processor_gen.preprocess(img).to(device, model.dtype) for img in norm_init_images],
                dim=0,
            )
        init_latents, gen_shape_from_vq = self.encode_image_gen(vq_latents)

        if gen_shape_from_vq is not None:
            grid_h, grid_w = gen_shape_from_vq
            grid_tokens = grid_h * grid_w

        is_unitok = "unitok" in model.config.mm_vqvae
        mask_idx_2d = torch.zeros(batch_size, grid_tokens, dtype=torch.bool, device=device)
        debug_samples = []

        def _parse_bbox_xyxy(raw_bbox):
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
                vals = [float(v) for v in raw_bbox[:4]]
                return vals[0], vals[1], vals[2], vals[3]
            txt = str(raw_bbox) if raw_bbox is not None else ""
            loc_vals = re.findall(r"<LOC_([0-9]+)>", txt)
            if len(loc_vals) >= 4:
                vals = [float(v) for v in loc_vals[:4]]
                return vals[0], vals[1], vals[2], vals[3]
            num_vals = re.findall(r"-?\d+(?:\.\d+)?", txt)
            if len(num_vals) >= 4:
                vals = [float(v) for v in num_vals[:4]]
                return vals[0], vals[1], vals[2], vals[3]
            return None

        for b in range(batch_size):
            raw_bbox = pred_bboxes[b] if pred_bboxes is not None and b < len(pred_bboxes) else None
            bbox = _parse_bbox_xyxy(raw_bbox)
            bbox_parse_error = None
            if bbox is None:
                mask_idx_2d[b] = True
                all_tokens = torch.arange(grid_tokens, device=device, dtype=torch.long)
                debug_samples.append(
                    {
                        "sample_index": b,
                        "pred_bbox_xyxy_1024": None,
                        "grid_bbox_xyxy": None,
                        "tokens": all_tokens.detach().cpu().tolist(),
                        "n_mask_tokens": int(grid_tokens),
                        "bbox_parse_error": "Could not parse bbox; fallback to full-mask region.",
                        "mask_idx_2d_flat": mask_idx_2d[b].to(torch.int).detach().cpu().tolist(),
                    }
                )
                continue

            x0, y0, x1, y1 = bbox
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            x0 = min(max(x0, 0.0), float(image_resolution))
            x1 = min(max(x1, 0.0), float(image_resolution))
            y0 = min(max(y0, 0.0), float(image_resolution))
            y1 = min(max(y1, 0.0), float(image_resolution))

            if x1 <= x0:
                x1 = min(float(image_resolution), x0 + 1.0)
            if y1 <= y0:
                y1 = min(float(image_resolution), y0 + 1.0)

            gx0 = int(math.floor(x0 * grid_w / image_resolution))
            gy0 = int(math.floor(y0 * grid_h / image_resolution))
            gx1 = int(math.ceil(x1 * grid_w / image_resolution))
            gy1 = int(math.ceil(y1 * grid_h / image_resolution))

            gx0 = min(max(gx0, 0), grid_w - 1)
            gy0 = min(max(gy0, 0), grid_h - 1)
            gx1 = min(max(gx1, gx0 + 1), grid_w)
            gy1 = min(max(gy1, gy0 + 1), grid_h)

            rows = torch.arange(gy0, gy1, device=device, dtype=torch.long)
            cols = torch.arange(gx0, gx1, device=device, dtype=torch.long)
            tokens = (rows[:, None] * grid_w + cols[None, :]).reshape(-1)
            mask_idx_2d[b, tokens] = True

            debug_samples.append(
                {
                    "sample_index": b,
                    "pred_bbox_xyxy_1024": [float(x0), float(y0), float(x1), float(y1)],
                    "grid_bbox_xyxy": [int(gx0), int(gy0), int(gx1), int(gy1)],
                    "tokens": tokens.detach().cpu().tolist(),
                    "n_mask_tokens": int(tokens.numel()),
                    "bbox_parse_error": bbox_parse_error,
                    "mask_idx_2d_flat": mask_idx_2d[b].to(torch.int).detach().cpu().tolist(),
                }
            )

        xt = init_latents.clone()
        if is_unitok:
            for b in range(batch_size):
                xt[b, :, mask_idx_2d[b]] = img_mask_id
        else:
            xt[mask_idx_2d] = img_mask_id

        mask_idx_after_apply = xt == img_mask_id
        if is_unitok and not use_3d:
            mask_idx_after_apply = mask_idx_after_apply[:, 0, :]
        xt_mask_counts = mask_idx_after_apply.sum(dim=1).detach().cpu().tolist()
        for sample_dbg, xt_count in zip(debug_samples, xt_mask_counts):
            sample_dbg["xt_mask_count_after_apply"] = int(xt_count)

        mask_idx = xt == img_mask_id
        if is_unitok and not use_3d:
            mask_idx = mask_idx[:, 0, :]
        n_mask_per_sample = mask_idx.sum(dim=1)
        max_n_mask = int(n_mask_per_sample.max().item())
        if max_n_mask == 0:
            decoded_images = self.decode_image_gen(xt, image_resolution, image_resolution)
            edited_images = [Image.fromarray(decoded_images[i]) for i in range(batch_size)]
            if return_debug:
                debug_payload = {
                    "debug_label": debug_label,
                    "mask_idx_2d_shape": [int(mask_idx_2d.shape[0]), int(mask_idx_2d.shape[1])],
                    "grid_shape": [int(grid_h), int(grid_w)],
                    "image_sizes": [list(sz) if sz is not None else None for sz in image_sizes],
                    "n_tokens": int(grid_tokens),
                    "is_unitok": bool(is_unitok),
                    "samples": debug_samples,
                }
                return edited_images, debug_payload
            return edited_images

        schedule_positions = torch.arange(mask_idx.shape[1], device=device).unsqueeze(0)
        schedule_mask = schedule_positions < n_mask_per_sample.unsqueeze(1)
        n_steps = max(1, int(64 * max_n_mask / 4096))
        if schedule == "shift":
            num_transfer_tokens = get_num_transfer_tokens_sch(
                schedule_mask, n_steps, schedule="shift", schedule_kwargs=dict(shift=shift)
            )
        else:
            num_transfer_tokens = get_num_transfer_tokens_sch(
                schedule_mask, n_steps, schedule=schedule, schedule_kwargs=dict(shift=shift)
            )

        sch_temperatures = torch.linspace(0, 1, n_steps, device="cpu").numpy()
        if schedule_temp == "linear":
            sch_temperatures = (1 - sch_temperatures) * (1 - min_temperature) + min_temperature
        elif schedule_temp == "cosine2":
            sch_temperatures = cosine_schedule_2(1 - sch_temperatures) * (1 - min_temperature) + min_temperature
        elif schedule_temp == "shift":
            sch_temperatures = logit_normal_schedule(shift_alg, 1 - sch_temperatures) * (1 - min_temperature) + min_temperature
        elif schedule_temp == "exp":
            sch_temperatures = exp_schedule(1 - sch_temperatures) * (1 - min_temperature) + min_temperature
        else:
            raise NotImplementedError(f"Unknown schedule_temp: {schedule_temp}")

        sch_temperatures_samp = torch.linspace(0, 1, n_steps, device="cpu").numpy()
        if schedule_temp_samp == "linear":
            sch_temperatures_samp = (1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
        elif schedule_temp_samp == "cosine2":
            sch_temperatures_samp = cosine_schedule_2(1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
        elif schedule_temp_samp == "shift":
            sch_temperatures_samp = logit_normal_schedule(shift_alg, 1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
        elif schedule_temp_samp == "exp":
            sch_temperatures_samp = exp_schedule(1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
        else:
            raise NotImplementedError(f"Unknown schedule_temp_samp: {schedule_temp_samp}")

        sch_temperatures = torch.tensor(sch_temperatures, device=device, dtype=torch.float32)
        sch_temperatures_samp = torch.tensor(sch_temperatures_samp, device=device, dtype=torch.float32)
        cfg_start = int(cfg_interval[0] * n_steps)
        cfg_end = int(cfg_interval[1] * n_steps)

        unmask_order = None
        if confidence_policy == "stratified" and enable_stratified:
            try:
                from pmj import stratified_random

                _dim = int(np.sqrt(grid_tokens))
                unmask_order = stratified_random(n=_dim, seed=42, shuffle_blocks=True)
            except Exception:
                confidence_policy = "mmada"

        x0 = xt.clone()
        x0_history = []
        active_steps = torch.nonzero(num_transfer_tokens.sum(dim=0) > 0, as_tuple=False).squeeze(-1)
        for step_idx, step_col in enumerate(
            tqdm(active_steps, desc=f"Region editing {batch_size} images"), start=1
        ):
            num_transfer = num_transfer_tokens[:, step_col]
            local_temp = sch_temperatures[step_idx - 1]
            local_temp_samp = sch_temperatures_samp[step_idx - 1]
            if step_idx / n_steps > order_cutoff:
                confidence_policy = "mmada"

            mask_idx = xt == img_mask_id
            if is_unitok and not use_3d:
                mask_idx = mask_idx[:, 0, :]
            n_mask_per_sample = mask_idx.sum(dim=1)
            if n_mask_per_sample.max().item() == 0:
                break

            timesteps = n_mask_per_sample.float() / mask_idx.shape[1]
            do_cfg = guidance_scale > 0 and cfg_start <= step_idx <= cfg_end
            if do_cfg:
                if inputs_embeds_cond is None or inputs_embeds_uncond is None or inputs_embeds_uncond_enc is None:
                    raise ValueError("CFG requested but missing inputs_embeds_* tensors.")
                if is_gen_enc_null is None or is_gen_enc_ccc is None:
                    raise ValueError("CFG requested but missing is_gen_enc_* tensors.")

            if inputs_embeds_cond is None:
                inputs_embeds_cond = inputs_embeds
            if do_cfg:
                input_embeddings_input = torch.cat(
                    [inputs_embeds_uncond, inputs_embeds_uncond_enc, inputs_embeds_cond]
                ).clone()
                xt_input = torch.cat([xt, xt, xt])
                new_token_mask = is_gen.repeat(3, 1)
                is_gen_enc_mask = torch.cat([is_gen_enc_null, is_gen_enc_ccc, is_gen_enc])
                timesteps_input = timesteps.repeat(3)
            else:
                input_embeddings_input = inputs_embeds.clone()
                new_token_mask = is_gen
                xt_input = xt
                is_gen_enc_mask = is_gen_enc
                timesteps_input = timesteps

            enc_use_image_branch = getattr(model.config, "enc_use_image_branch", False)
            if enc_use_image_branch:
                modality_indices = new_token_mask | is_gen_enc_mask
            else:
                modality_indices = new_token_mask

            all_input_embeddings, new_token_mask = llada_wte(
                model,
                None,
                True,
                x_gen=xt_input,
                gen_shape=gen_shape,
                inputs_embeds_curr=input_embeddings_input,
                new_token_mask=new_token_mask,
            )
            logits = get_logits(
                model,
                all_input_embeddings,
                new_token_mask,
                True,
                gen_shape=gen_shape,
                input_modality_indices=modality_indices,
                timesteps=timesteps_input,
            )

            if do_cfg:
                new_token_mask, _, _ = new_token_mask.chunk(3)
                logits_un, logits_un_enc, logits = logits.chunk(3)
                logits_is_ninf = logits == -np.inf
                if edit_mode in [0, 3]:
                    logits_cond = (1 + guidance_scale_image) * logits - guidance_scale_image * logits_un_enc
                elif edit_mode in [1, 2]:
                    logits_cond = (logits + guidance_scale_image * logits_un_enc) / (1 + guidance_scale_image)
                else:
                    raise ValueError(f"Not Supported edit_mode: {edit_mode}")
                logits = (1 + guidance_scale) * logits_cond - guidance_scale * logits_un
                logits[logits_is_ninf] = -np.inf

            if top_p is not None or top_k is not None:
                _b, _l, _v = logits.shape
                logits = logits.view(_b * _l, _v)
                logits = _top_k_top_p_filtering(
                    logits, top_k=top_k or 0, top_p=top_p if top_p is not None else 1.0, filter_value=-np.inf, min_tokens_to_keep=1
                )
                logits = logits.view(_b, _l, _v)

            if is_unitok:
                logits[..., 4096:] = float("-inf")

            probs = logits.softmax(dim=-1)
            if sample_policy == "multinomial":
                _temperature = temperature
                if dynamic_temperature_samp:
                    _temperature = _temperature * local_temp_samp
                x0 = dists.Categorical(logits=logits / _temperature).sample()
                x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
            elif sample_policy == "argmax":
                x0 = logits.argmax(-1)
                x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
            else:
                raise NotImplementedError(f"Unknown sample_policy: {sample_policy}")

            if is_unitok:
                x0 = x0.permute(0, 2, 1)
                if use_3d:
                    x0 = torch.where(mask_idx, x0, xt)
                    x0_p = x0_p.permute(0, 2, 1).max(dim=1)[0]
                else:
                    x0 = torch.where(mask_idx.unsqueeze(1).repeat(1, 8, 1), x0, xt)
                    x0_p = x0_p.permute(0, 2, 1).max(dim=1)[0]
            else:
                x0 = torch.where(mask_idx, x0, xt)
            
            x0_history.append(x0.clone().cpu())

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for b in range(batch_size):
                b_mask = mask_idx[b]
                b_n_mask = int(b_mask.sum().item())
                if b_n_mask == 0:
                    continue
                k = min(int(num_transfer[b].item()), b_n_mask)
                if k <= 0:
                    continue

                if confidence_policy == "mmada":
                    _alg_temp = alg_temp * local_temp if dynamic_temperature else alg_temp
                    confidence_b = torch.log(x0_p[b].clamp(1e-20)) + _alg_temp * _gumbel_noise(x0_p[b])
                    confidence_b = torch.where(b_mask, confidence_b, -np.inf)
                    _, select_index = torch.topk(confidence_b, k=k)
                elif confidence_policy == "mask_git":
                    _alg_temp = alg_temp * local_temp if dynamic_temperature else alg_temp
                    confidence_b = x0_p[b] / _alg_temp
                    confidence_b = torch.where(b_mask, confidence_b, -np.inf)
                    confidence_b = torch.softmax(confidence_b, dim=-1)
                    select_index = torch.multinomial(confidence_b.unsqueeze(0), num_samples=k).squeeze(0)
                elif confidence_policy == "stratified" and unmask_order is not None:
                    start = grid_tokens - b_n_mask
                    select_index = torch.tensor(unmask_order[start : start + k], device=device, dtype=torch.long)
                else:
                    _alg_temp = alg_temp * local_temp if dynamic_temperature else alg_temp
                    confidence_b = torch.log(x0_p[b].clamp(1e-20)) + _alg_temp * _gumbel_noise(x0_p[b])
                    confidence_b = torch.where(b_mask, confidence_b, -np.inf)
                    _, select_index = torch.topk(confidence_b, k=k)

                if is_unitok:
                    transfer_index[b, :, select_index] = True
                else:
                    transfer_index[b, select_index] = True

            xt[transfer_index] = x0[transfer_index]

        xt = x0.clone()
        xt[xt == img_mask_id] = x0[xt == img_mask_id]
        decoded_images = self.decode_image_gen(xt, image_resolution, image_resolution)
        edited_images = [Image.fromarray(decoded_images[i]) for i in range(batch_size)]
        if return_debug:
            for img in x0_history:
                history_images = self.decode_image_gen(xt, image_resolution, image_resolution)
                debug_images = [Image.fromarray(decoded_images[i]) for i in range(batch_size)]
            debug_payload = {
                "debug_label": debug_label,
                "mask_idx_2d_shape": [int(mask_idx_2d.shape[0]), int(mask_idx_2d.shape[1])],
                "grid_shape": [int(grid_h), int(grid_w)],
                "image_sizes": [list(sz) if sz is not None else None for sz in image_sizes],
                "n_tokens": int(grid_tokens),
                "is_unitok": bool(is_unitok),
                "samples": debug_samples,
                "n_steps": n_steps,
                "x0_history": x0_history,
            }
            return edited_images, debug_payload
        return edited_images


    @torch.no_grad()
    def generate_text(
        self,
        prompt=None,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        tokenizer=None,
        t2i_inference=False,
        do_sample=False,
        prefix_lm=False,
        verbose=False,
        draft_tokens=None,
        prompt_index=None,
        input_modality_indices=None,
        bbox_mask=None,
        mask_pos=None,
    ):
        model = self.get_model()
        bsz, seq_len = inputs_embeds.shape[:2]
        if prompt is None:
            assert inputs_embeds is not None
            prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)

        if steps == 1:
            if bbox_mask is None and mask_pos is not None:
                # Backward-compatibility: allow legacy callers to pass explicit
                # token positions and convert them into a boolean mask.
                bbox_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=inputs_embeds.device)
                bbox_mask.scatter_(1, mask_pos.clamp(min=0, max=seq_len - 1), True)
            if bbox_mask is None:
                raise ValueError("bbox_mask (or legacy mask_pos) is required when steps == 1.")
            logits = get_logits(model, inputs_embeds) # (bsz, expanded_len, vocab_size)
            x = logits[bbox_mask].argmax(-1).view(-1,4)  # (bsz, 4)   
            
        else:
            past_key_values = None
            if prefix_lm:
                if input_modality_indices is None and t2i_inference:
                    input_modality_indices = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device)
                past_key_values = model(
                    None,
                    input_embeddings=inputs_embeds,
                    use_cache=True,
                    modality_indices=input_modality_indices
                ).attn_key_values
                x = torch.full((bsz, gen_length), mask_id, dtype=torch.long).to(model.device)
                prompt = torch.full((bsz, 0), 0, dtype=torch.long).to(model.device)
            else:
                x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
                x[:, :prompt.shape[1]] = prompt.clone()

            if prompt_index is None:
                prompt_index = (x != mask_id)
            if draft_tokens is not None:
                assert draft_tokens.shape[1] <= gen_length
                x[:, prompt.shape[1]:prompt.shape[1]+draft_tokens.shape[1]] = draft_tokens.clone()

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)
            if verbose:
                history = []
            noise_embeddings = model.transformer.wte(torch.tensor([mask_id]).to(model.device)) # 1, 4096
            nfe = 0
            hist = []
            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id
                    if mask_index.sum() == 0:
                        continue

                    # Build embeddings using the reference wte helper
                    inputs_embeds_curr, new_token_mask = llada_wte(model, x, t2i_inference)

                    if cfg_scale > 0.0:
                        un_inputs_embeds_curr = inputs_embeds_curr.clone()
                        if prefix_lm:
                            raise NotImplementedError()
                        else:
                            inputs_embeds_curr[:, :inputs_embeds.shape[1]] = inputs_embeds
                            un_inputs_embeds_curr[:, :inputs_embeds.shape[1]] = inputs_embeds
                            un_inputs_embeds_curr[prompt_index] = noise_embeddings
                            inputs_embeds_curr_cat = torch.cat([un_inputs_embeds_curr, inputs_embeds_curr], dim=0)
                            logits = get_logits(model, inputs_embeds_curr_cat, new_token_mask.repeat(2, 1), t2i_inference)
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        if prefix_lm:
                            logits = get_logits(model, inputs_embeds_curr, new_token_mask, t2i_inference, past_key_values=past_key_values)
                        else:
                            if inputs_embeds is not None:
                                inputs_embeds_curr[:, :inputs_embeds.shape[1]] = inputs_embeds
                            logits = get_logits(model, inputs_embeds_curr, new_token_mask, t2i_inference)

                    # Apply Gumbel noise for sampling (use float64 per reference)
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    del logits_with_noise

                    # Handle remasking strategy (use float32 — sufficient precision for topk ranking)
                    if remasking in ("low_confidence", "low_confidence_dynamic"):
                        p = F.softmax(logits.to(torch.float32), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )
                        del p
                    elif remasking == "random":
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    elif remasking == "entropy":
                        eps = 1e-10
                        probs = F.softmax(logits.to(torch.float32), dim=-1)
                        log_probs = torch.log(probs + eps)
                        x0_p = torch.sum(probs * log_probs, dim=-1)
                        del probs, log_probs
                    elif remasking == "margin":
                        p = F.softmax(logits.to(torch.float32), dim=-1)
                        sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                        top1_probs = sorted_probs[:, :, 0]
                        top2_probs = sorted_probs[:, :, 1]
                        x0_p = top1_probs - top2_probs
                        del p, sorted_probs, top1_probs, top2_probs
                    else:
                        raise NotImplementedError(remasking)

                    # Ensure we don't process tokens beyond the current block
                    x0_p[:, end_idx:] = -np.inf

                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    # Select tokens to transfer based on confidence
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        num_tokens = num_transfer_tokens[j, i].item()
                        if num_tokens > 0:
                            _, select_index = torch.topk(confidence[j], k=num_tokens)
                            transfer_index[j, select_index] = True

                    x[transfer_index] = x0[transfer_index]
                    del x0, confidence, transfer_index

        return x


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        use_fast_dlm: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            # Flatten List[5D (n_imgs, n_patches, C, H, W)] → List[4D (n_patches, C, H, W)]
            # so prepare_inputs_labels_for_multimodal gets one element per <image> token
            flat_images = []
            for img_group in images:
                if isinstance(img_group, (list, tuple)):
                    # img_group is a list of tensors (e.g. multiple images per sample)
                    for single in img_group:
                        if isinstance(single, torch.Tensor) and single.dim() == 5:
                            for s in single:
                                flat_images.append(s)
                        else:
                            flat_images.append(single)
                elif img_group.dim() == 5:
                    for single in img_group:
                        flat_images.append(single)
                else:
                    flat_images.append(img_group)
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, flat_images, modalities, image_sizes=image_sizes)
        else:
            # breakpoint()
            inputs_embeds = self.get_model().embed_tokens(inputs)
        if use_fast_dlm:
            return generate_with_dual_cache(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
        return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
    
    
    @torch.no_grad()
    def log_likelyhood_inference(
        self,
        inputs: Optional[torch.Tensor] = None,
        answer: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        mc_num=128,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        max_seq_len = 5000
        if inputs_embeds.shape[1] > max_seq_len:
            inputs_embeds = inputs_embeds[:, -max_seq_len:]
        answer = answer[:, :300]
        return get_log_likelihood(self.get_model(), None, inputs_embeds=inputs_embeds, answer=answer, mc_num=mc_num,**kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llada", LlavaLladaConfig)
AutoModelForCausalLM.register(LlavaLladaConfig, LlavaLladaForMaskedDiffusion)

    
    
            
    
