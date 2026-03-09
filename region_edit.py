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


print("Importing necessary libraries...")
import os
if __name__ == "__main__":
    os.environ['DEBUG_FIX_PADDING'] = '1'
    os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'

from llava.mm_utils import process_images, tokenizer_image_token, pad_to_square_and_resize
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from PIL import Image
import copy
import cv2

import torch

from llava.model.language_model.llada.generate import cosine_schedule_2,logit_normal_schedule,exp_schedule,wte,get_logits,get_num_transfer_tokens_sch
import time
from tqdm.auto import tqdm
import numpy as np 
import torch.distributions as dists
from einops import rearrange
from llava.model.utils import pad_along_last_dim
from predict_t2i_edit import gumbel_noise, top_k_top_p_filtering
from pmj import stratified_random


import numpy as np
import cv2

class LavidaOBboxTokenMapper:
    """
    Maps bounding box coordinates to visual token indices for LaVidaO models.

    LaVidaO uses a vision transformer that processes images in patches,
    with each patch corresponding to one or more visual tokens.
    """
    def __init__(self,
                 patch_size: int = 16,
                 ):
        """
        Image dimensions will be set dynamically when processing each image.

        Args:
            patch_size: Size of each patch in pixels
        """
        self.patch_size = patch_size

        # These will be set dynamically for each image
        self.image_height = None
        self.image_width = None
        self.grid_height = None
        self.grid_width = None
        self.token_grid_height = None
        self.token_grid_width = None
        self.total_tokens = None

    def _setup_image_dimensions(self, image_height: int, image_width: int):
        """
        Set up grid dimensions for a specific image size.

        Args:
            image_height: Height of the input image in pixels
            image_width: Width of the input image in pixels
        """
        self.image_height = image_height
        self.image_width = image_width

        # Calculate grid dimensions after patching
        # Calculate grid dimensions after patching
        self.grid_height = self.image_height // self.patch_size
        self.grid_width = self.image_width // self.patch_size

        # In this simple mapper, token grid matches patch grid
        self.token_grid_height = self.grid_height
        self.token_grid_width = self.grid_width

        self.total_tokens = self.token_grid_height * self.token_grid_width

    def rotated_rect_to_token_indices(self, rect_tuple):
        """
        Convert rotated rectangle to a list of token indices.

        Args:
           rect_tuple: ((center_x, center_y), (width, height), angle) in original image coordinates.

        Returns:
           List of integer token indices (0 to 4095).
        """
        center, size, angle = rect_tuple
        # Scale to grid coordinates
        grid_center = (center[0] / self.patch_size, center[1] / self.patch_size)
        grid_size = (size[0] / self.patch_size, size[1] / self.patch_size)
        scaled_rect = (grid_center, grid_size, angle)

        box_points = cv2.boxPoints(scaled_rect)
        box_points = np.int32(box_points)

        # Create mask on grid
        mask = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        cv2.drawContours(mask, [box_points], 0, 1, -1) # -1 fills the contour

        # Get indices
        # indices are (row, col) -> (y, x)
        ys, xs = np.where(mask == 1)

        # Convert to flat indices: index = y * width + x
        flat_indices = ys * self.token_grid_width + xs
        return flat_indices.tolist()

    def token_indices_to_rotated_rect(self, indices):
        """
        Convert token indices back to a rotated rectangle in original image coordinates.

        Args:
            indices: List of flattened token indices.

        Returns:
            rect_tuple: ((center_x, center_y), (width, height), angle) or None if no indices.
        """
        if not indices:
            return None

        mask = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        # Convert flat indices to (y, x)
        indices = np.array(indices)
        ys = indices // self.token_grid_width
        xs = indices % self.token_grid_width

        mask[ys, xs] = 255

        coords = cv2.findNonZero(mask)
        if coords is None:
            return None

        # Get rotated rect in grid coordinates
        # minAreaRect returns ((cx, cy), (w, h), angle)
        rect_grid = cv2.minAreaRect(coords)
        center_grid, size_grid, angle = rect_grid

        # Scale back to image coordinates
        center_img = (center_grid[0] * self.patch_size, center_grid[1] * self.patch_size)
        size_img = (size_grid[0] * self.patch_size, size_grid[1] * self.patch_size)

        return (center_img, size_img, angle)


@torch.no_grad()
def regional_text2edit(
    model, 
    prompt,
    init_image,
    rotated_rect=None,
    original_img_size=None,
    sample_policy = 'multinomial',
    confidence_policy = 'mmada',
    guidance_scale = 5,
    n_steps = 20,
    batch_size = 1,
    tokenizer=None,
    image_resolution=512,
    n_tokens=1024,
    shift=3,
    alg_temp=1,
    schedule='shift',
    min_temperature=0.01,
    dynamic_temperature=False,
    micro_cond = 'ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.5',
    temperature=1,
    schedule_temp='linear',
    shift_alg=None,
    is_4k=False,
    is_4k_2=False,
    top_p=None,
    top_k=None,
    unmask_order=None,
    use_3d=False,
    schedule_temp_samp='linear',
    dynamic_temperature_samp=False,
    min_temperature_samp=1,
    cfg_interval=[0,1],
    order_cutoff = 100,
    image_processor=None,
    guidance_scale_image=5,              
    edit_mode=0,
    n_refinement=0,
    local_temp_samp_refinement=0.00001,
    remask_ratio = 0.01,
    template='Generate an image with the caption:\n <prompt>',
    guidance_scale_refinement=1,
    enable_stratified=False,
    plan=None,
    feedback_imgs=None,
    feedback_texts=None,
    *args,
    **kwargs
):
    if shift_alg is None:
        shift_alg = shift
    device = model.get_model().device
    conv_template = "llada" 
    reserve_token = '<|reserved_token_5|>'
    reserve_id = 126089
    reserve_id2 = 126090
    #OFFSET = 150000
    INT_MAX = 1_000_000
    img_mask_id =8193
    txt_mask_id = 126336
    plan_range = 126349

    gen_shape_map = {
        1024: (64,64),
        512: (32,32),
        256: (16,16),
    }
    gen_shape = gen_shape_map[image_resolution]
        
    micro_cond = micro_cond
    # if len(prompt) > 800:
    #     prompt = prompt[:800]
    micro_cond = ''
    prompt = f"{prompt} {micro_cond}"
    if feedback_texts is not None:
        feedback_str = [f'Generation {i+1}: <image>\n Feedback {i+1}: {feedback}' for i,feedback in enumerate(feedback_texts)]
        feedback_str = '\n'.join(feedback_str)
        feedback_str = f' Please also consider past generations and their feedbacks. Do not repeat these errors {feedback_str}'
        prompt = f"{prompt} {feedback_str}"
    if template is not None:
        question = template.replace('<prompt>', prompt)
    else:
        question = prompt
    n_tokens_txt = n_tokens
    
    if image_resolution == 1024:
        n_tokens_txt = 1024
    
    if plan is None:
        plan = ""
    
    if init_image is None:
        raise ValueError("init_image is required for regional_text2edit")
    image = init_image
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=model.device) for _image in image_tensor]
    modalities = ['image']
    image_sizes = [image.size]
    conv_template = 'llada'
    n_tokens_txt = 1024
    conv = copy.deepcopy(conv_templates[conv_template])
    reserve_token_2 = '<|reserved_token_6|>'
    # breakpoint()
    image = image.convert('RGB')
    image_1024 = pad_to_square_and_resize(image, image_resolution)
    enc_latents,_gen_shape = model.encode_image_gen(model.model.image_processor_gen.preprocess(image_1024).to(model.device,model.dtype),enc=True)
    enc_embeddings = model.model.call_gen_embedding(enc_latents,_gen_shape,enc=True)
    conv.append_message(conv.roles[0], f"<image> {reserve_token_2*enc_embeddings.shape[1]}\n {prompt} ")
    reserve_token = '<|reserved_token_5|>'
    conv.append_message(conv.roles[1], f"{plan}{reserve_token*n_tokens_txt}")
    prompt_question = conv.get_prompt()
    prompt_question.removesuffix('<|start_header_id|>assistant<|end_header_id|>\n\n')
    input_log = prompt_question.replace('<|reserved_token_5|>','@').replace('<|reserved_token_6|>','*')
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    attention_mask=None
    position_ids=None
    image_sizes = [image.size]
    modalities=["image"]
    
    (inputs, position_ids, attention_mask, _, inputs_embeds, _,raw_input_ids) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids, 
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
        modalities=modalities,
        image_sizes=image_sizes,
        return_inputs=True
        )
    # print(raw_input_ids)
    
    # Under FSDP summon_full_params, prepare_inputs_labels_for_multimodal returns fp32
    # (vision tower has explicit .float() casts). Cast to match enc_embeddings compute dtype.
    inputs_embeds = inputs_embeds.to(enc_embeddings.dtype)
    inputs_embeds[raw_input_ids == reserve_id2] = 0
    _enc = pad_along_last_dim(enc_embeddings,inputs_embeds.shape[-1])
    # _enc = torch.zeros_like(inputs_embeds[raw_input_ids == reserve_id2])
    # assert _enc[:,:enc_embeddings.shape[-1]].shape ==enc_embeddings.flatten(0,1).shape
    # _enc[:,:enc_embeddings.shape[-1]] = enc_embeddings.flatten(0,1)
    inputs_embeds[raw_input_ids == reserve_id2] = _enc.flatten(0,1)
    is_eot = torch.where(raw_input_ids==126348)[1]
    assert len(is_eot) == 3
    prompt_cutoff = is_eot[1]
    is_prompt = torch.zeros_like(raw_input_ids,dtype=torch.bool)
    is_prompt[:,:prompt_cutoff+1] = True
    is_gen = raw_input_ids == reserve_id
    is_gen_enc = raw_input_ids == reserve_id2
    if plan:
        is_gen_min = torch.where(raw_input_ids==reserve_id)[1][0]
        is_prompt[:,:is_gen_min] = True

    inputs_embeds_uncond = inputs_embeds.clone()
    # wte returns fp32 under FSDP summon_full_params (master weight dtype); cast to compute dtype.
    noise_embed = model.model.transformer.wte(torch.tensor([txt_mask_id]).to(model.device)).to(inputs_embeds.dtype) # 1, d
    inputs_embeds_uncond[is_prompt] =  noise_embed
    inputs_embeds_uncond_enc = inputs_embeds.clone()
    if edit_mode == 0:
        inputs_embeds_uncond_enc[~is_gen_enc] = noise_embed
        is_gen_enc_ccc = is_gen_enc
    elif edit_mode == 1:
        inputs_embeds_uncond_enc[is_gen_enc] = noise_embed
        is_gen_enc_ccc =  torch.zeros_like(is_gen_enc,dtype=torch.bool)
    elif edit_mode == 2:
        inputs_embeds_uncond_enc[is_gen_enc|(raw_input_ids<0)] = noise_embed
        is_gen_enc_ccc =  torch.zeros_like(is_gen_enc,dtype=torch.bool)
    elif edit_mode == 3:
        inputs_embeds_uncond_enc[(~is_gen_enc) & (raw_input_ids>0)] = noise_embed
        is_gen_enc_ccc = is_gen_enc
    else:
        raise ValueError("Not Supported edit_mode")
    
    image_gen_latents_offset = torch.zeros(batch_size, n_tokens, dtype=torch.long, device=input_ids.device) #+ img_mask_offset
    is_unitok = 'unitok' in model.get_model().config.mm_vqvae
    if is_unitok:
        image_gen_latents_offset = image_gen_latents_offset.unsqueeze(1).repeat(1,8,1) # N, Num codebooks, Length1
    image_gen_latents_offset[:] =  img_mask_id

    xt = image_gen_latents_offset.clone()
    if init_image is not None:
        init_latents, _gen_shape = model.encode_image_gen(model.model.image_processor_gen.preprocess(init_image).to(model.device,model.dtype))
        """
        # Random selection
        n_mask_remask = max(int(n_tokens * remask_ratio),1)
        indices = np.arange(n_tokens)
        np.random.shuffle(indices)
        init_mask_indices = indices[:n_mask_remask]
        xt[:,init_mask_indices] = init_latents[:, init_mask_indices]
        """
        if rotated_rect is None:
             raise ValueError("rotated_rect (or bbox in kwargs) is required when init_image is provided")

        # 1. resize the rotated rectangle to match padded -> center cropped
        if original_img_size is None:
             raise ValueError("original_img_size is required for coordinate mapping")
             
        img_w, img_h = original_img_size
        target_size = 1024 
        max_dim = max(img_w, img_h)
        scale = target_size / max_dim
        valid_w = int(img_w * scale)
        valid_h = int(img_h * scale)
        pad_left = (target_size - valid_w) // 2
        pad_top = (target_size - valid_h) // 2
        
        # Transform Rect
        center, size, angle = rotated_rect
        new_center_x = center[0] * scale + pad_left
        new_center_y = center[1] * scale + pad_top
        new_size_w = size[0] * scale
        new_size_h = size[1] * scale
        padded_rect_tuple = ((new_center_x, new_center_y), (new_size_w, new_size_h), angle)
        
        # Visualize on padded image
        padded_pil = init_image
        
        # 4. Mapper Test
        grid_h, grid_w = gen_shape
        patch_size = image_resolution // grid_h
        mapper = LavidaOBboxTokenMapper(patch_size=patch_size)
        
        # Set dimensions
        mapper._setup_image_dimensions(target_size, target_size)
        mapper.token_grid_height = target_size // patch_size
        mapper.token_grid_width = target_size // patch_size
        
        tokens = mapper.rotated_rect_to_token_indices(padded_rect_tuple)
        xt = init_latents.clone()
        xt[:,tokens] = img_mask_id
        print("TOKEN INDICES MASKED", tokens)
        
        # Verification: Visualize the mask
        import numpy as np
        
        # 1. Create a mask image from tokens
        mask_vis = np.zeros((64, 64), dtype=np.uint8)
        # tokens are flat indices 0-4095
        rows = [t // 64 for t in tokens]
        cols = [t % 64 for t in tokens]
        mask_vis[rows, cols] = 255
        
        # 2. Resize mask to match image size (1024x1024)
        mask_vis_resized = cv2.resize(mask_vis, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        # 3. Overlay on the padded image (convert PIL to CV2)
        padded_cv = cv2.cvtColor(np.array(padded_pil), cv2.COLOR_RGB2BGR)
        
        # Draw the original rotated rect for comparison
        box = cv2.boxPoints(padded_rect_tuple)
        box = np.int32(box)
        cv2.drawContours(padded_cv, [box], 0, (0, 255, 0), 2) # Green for original rect
        
        # Overlay mask (Red)
        heatmap = np.zeros_like(padded_cv)
        heatmap[mask_vis_resized == 255] = [0, 0, 255] # Red
        
        vis_result = cv2.addWeighted(padded_cv, 0.7, heatmap, 0.3, 0)
    
    mask_idx = xt == img_mask_id
    if is_unitok: 
        mask_idx = mask_idx[:,0,:]
    n_mask = mask_idx.sum()
    if is_4k:
        mask_idx = mask_idx[:, :1024]
    mask_idx_sch = mask_idx
    if use_3d:
        mask_idx_sch = mask_idx_sch.repeat(1,8)
    if schedule == 'shift':
        num_transfer_tokens = get_num_transfer_tokens_sch(mask_idx_sch, n_steps,schedule='shift',schedule_kwargs=dict(shift=shift))
    else:
        num_transfer_tokens = get_num_transfer_tokens_sch(mask_idx_sch, n_steps,schedule=schedule,schedule_kwargs=dict(shift=shift))
    print(num_transfer_tokens)
    #sch_temperatures = #torch.linspace(1, min_temperature, n_steps, device=device)
    sch_temperatures = torch.linspace(0,1, n_steps, device='cpu').numpy()
    if schedule_temp == 'linear':
        sch_temperatures = (1- sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp == 'cosine2':
        sch_temperatures = cosine_schedule_2(1-sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp == 'shift':
        sch_temperatures = logit_normal_schedule(shift_alg,1-sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp_samp == 'exp':
        sch_temperatures = exp_schedule(1-sch_temperatures) * (1 - min_temperature) + min_temperature
    else:
        raise NotImplementedError(f"Unknown schedule_temp: {schedule_temp}")

    sch_temperatures_samp = torch.linspace(0,1, n_steps, device='cpu').numpy()
    if schedule_temp_samp == 'linear':
        sch_temperatures_samp = (1- sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'cosine2':
        sch_temperatures_samp = cosine_schedule_2(1-sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'shift':
        sch_temperatures_samp = logit_normal_schedule(shift_alg,1-sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'exp':
        sch_temperatures_samp = exp_schedule(1-sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    else:
        raise NotImplementedError(f"Unknown schedule_temp_samp: {schedule_temp_samp}")
        
    sch_temperatures = torch.tensor(sch_temperatures, device=device, dtype=torch.float32)
    sch_temperatures_samp = torch.tensor(sch_temperatures_samp, device=device, dtype=torch.float32)
    temp_idx = 0
    cfg_start = int(cfg_interval[0] * n_steps)
    cfg_end = int(cfg_interval[1] * n_steps)
    if isinstance(unmask_order, str):
        unmask_order =  torch.load(unmask_order).tolist()
    if confidence_policy == 'stratified' and unmask_order is None:
        _dim = int(np.sqrt(n_tokens))
        seq = stratified_random(n=_dim, seed=42, shuffle_blocks=True)
        unmask_order = seq
    if confidence_policy == 'stratified':
        assert unmask_order is not  None
    n_c = 0
    cat_cfg_time = 0
    post_cfg_time = 0
    wte_time = 0
    fwd_time = 0
    sample_time = 0
    all_time = 0
    for num_transfer in tqdm(num_transfer_tokens[0]):
        t0 = time.time()
        local_temp = sch_temperatures[temp_idx]
        local_temp_samp = sch_temperatures_samp[temp_idx]
        temp_idx += 1
        if temp_idx / n_steps > order_cutoff:
            confidence_policy = 'mmada'
        # if temp_idx == n_steps:
        #     sample_policy = 'argmax'
        mask_idx = xt == img_mask_id
        if is_unitok and not use_3d:
            mask_idx = mask_idx[:,0,:] 
        n_mask = mask_idx.sum()

        timesteps = n_mask / mask_idx.numel()
        timesteps = timesteps.view(1)
        # if n_mask==0:
        #     break
        
        # Get Logits
        do_cfg = guidance_scale > 0 and cfg_start <= temp_idx and  cfg_end >= temp_idx
        if do_cfg:
            input_embeddings_input = torch.cat([inputs_embeds_uncond,inputs_embeds_uncond_enc,inputs_embeds]).clone()
            # nothing, masking vq tokens, final inputs
            xt_input = torch.cat([xt,xt,xt])
            new_token_mask = is_gen.repeat(3,1)
            is_gen_enc_null = torch.zeros_like(is_gen_enc,dtype=torch.bool)
            is_gen_enc_mask = torch.cat([is_gen_enc_null,is_gen_enc_ccc,is_gen_enc])
            timesteps = timesteps.repeat(3)
        else:
            input_embeddings_input = inputs_embeds.clone()
            # x_txt_raw_input = x_txt_raw
            new_token_mask = is_gen
            xt_input = xt
            is_gen_enc_mask = is_gen_enc
        # breakpoint()
        enc_use_image_branch = getattr(model.get_model().config,'enc_use_image_branch',False)
        if enc_use_image_branch:
            modality_indices = new_token_mask | is_gen_enc_mask
        else:
            modality_indices = new_token_mask
        t1 = time.time()
        all_input_embeddings,new_token_mask = wte(model.get_model(),None,True,x_gen=xt_input,gen_shape=gen_shape,inputs_embeds_curr=input_embeddings_input,new_token_mask=new_token_mask)
        #torch.cuda.synchronize()
        t2 = time.time()
        logits = get_logits(model.get_model(),all_input_embeddings,new_token_mask,True,gen_shape=gen_shape,input_modality_indices=modality_indices,timesteps=timesteps)
        t3 = time.time()
        if do_cfg:
            new_token_mask,_,_ = new_token_mask.chunk(3)
            logits_un,logits_un_enc,logits = logits.chunk(3)
            logits_is_ninf = logits == -np.inf
            #logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)
            if edit_mode in [0, 3]:
                logits_cond = (1 + guidance_scale_image) * logits - guidance_scale_image * logits_un_enc
            elif edit_mode in [1,2]:
                logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)                    
            #logits_cond = logits_cond
            logits = (1 + guidance_scale) * logits_cond - guidance_scale * logits_un
            logits[logits_is_ninf] = -np.inf
        # 4096 8 2064
        #torch.cuda.synchronize()
        t4 = time.time()
        if top_p is not None or top_k is not None:
            _b,_l,_v = logits.shape
            logits = logits.view(_b*_l,_v)
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, filter_value=-np.inf, min_tokens_to_keep=1)
            logits = logits.view(_b,_l,_v)
        if is_unitok:
            logits[...,4096:] = float('-inf')
        probs = logits.softmax(dim=-1) # N L 8 D
        
        if sample_policy == 'multinomial':
            _temperature = temperature
            if dynamic_temperature_samp:
                _temperature = _temperature * local_temp_samp
            x0 = dists.Categorical(logits=logits/_temperature).sample()
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        elif sample_policy == 'argmax':
            x0 = logits.argmax(-1)
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        else:
            raise NotImplementedError
        if is_unitok:
            # x0 is N L 8
            x0 = x0.permute(0,2,1) #  N 8 L
            # x0 = x0.unsqueeze(0) 
            if use_3d:
                x0 = torch.where(mask_idx, x0, xt)
                x0_p = x0_p.permute(0,2,1)[0] # N 8 L
            else:
                x0 = torch.where(mask_idx.unsqueeze(1).repeat(1,8,1), x0, xt)
                x0_p = x0_p.permute(0,2,1).max(dim=1)[0] # N L
        else:
            x0 = torch.where(mask_idx, x0, xt)
        # get confidence
        if confidence_policy == 'mask_git':
            _alg_temp = alg_temp
            if dynamic_temperature:
                _alg_temp = _alg_temp * local_temp
            confidence = x0_p / _alg_temp
            confidence = torch.where(mask_idx, confidence, -np.inf)
            if use_3d:
                confidence = confidence.flatten(1,2)
            confidence = torch.softmax(confidence, dim=-1)
            select_index = torch.multinomial(confidence, num_samples=num_transfer)
        elif confidence_policy == 'mmada':
            _alg_temp = alg_temp
            if dynamic_temperature:
                _alg_temp = _alg_temp * local_temp
            if is_4k:
                # x0_p = rearrange(x0_p,'b (h w) -> b h w', h=64, w=64)
                x0_p = rearrange(x0_p,'b (h p w q) -> b h w (p q)', h=32, w=32, p=2, q=2)
                x0_p = x0_p.max(dim=-1)[0]
                #print("b h w",x0_p.shape)
                x0_p = rearrange(x0_p,'b h w -> b (h w)')
                #print(mask_idx.shape)
                mask_idx = rearrange(mask_idx,'b (h p w q) -> b (h w) (p q)', h=32, w=32, p=2, q=2) 
                mask_idx = mask_idx.any(dim=-1)
                #print(mask_idx.shape)
            confidence = torch.log(x0_p.clamp(1e-20)) + _alg_temp * gumbel_noise(x0_p, generator=None)
            confidence = torch.where(mask_idx, confidence, -np.inf)
            if is_4k_2:
                confidence = rearrange(confidence,'b (h p w q) -> b h w (p q)', h=32, w=32, p=2, q=2)
                confidence_max = confidence.max(dim=-1,keepdims=True)[0]
                confidence[(confidence<confidence_max )&( confidence > -np.inf) ] = -9999 
                confidence = rearrange(confidence,'b h w (p q)-> b (h p w q)', h=32, w=32, p=2, q=2)
            if use_3d:
                confidence = confidence.flatten(1,2)
            # print(confidence.shape,num_transfer)
            _, select_index = torch.topk(confidence[0], k=num_transfer)
            if is_4k:
                z = torch.arange(4096, device=select_index.device, dtype=torch.long)
                z = rearrange(z,'(h w p q) -> (h w) (p q)', h=32, w=32, p=2, q=2)
                select_index = z[select_index].reshape(-1)
            # return select_index
        elif confidence_policy == 'stratified':
            # unmask_order
            start = n_tokens - n_mask
            if use_3d:
                start = n_tokens * 8 - n_mask
            
            select_index = torch.tensor(unmask_order[start:start+num_transfer], device=x0.device, dtype=torch.long)
            # print(select_index.shape)
            #print(select_index,n_mask,len(unmask_order))
        else:
            raise NotImplementedError
        
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        j = 0
        # _, select_index = torch.topk(confidence[j], k=num_transfer)
        # print(transfer_index.shape,select_index)
        # return
 
        if is_unitok:
            if use_3d:
                _b,_d,_l =  transfer_index.shape
                transfer_index = transfer_index.view(_b,-1)
                transfer_index[j, select_index] = True
                transfer_index = transfer_index.view(_b,_d,_l)
            else:
                transfer_index[j,:, select_index] = True
        else:
            transfer_index[j, select_index] = True
        xt[transfer_index] = x0[transfer_index]
        xt_is_mask = xt == img_mask_id
        t5 = time.time()
        cat_cfg_time += t1-t0
        post_cfg_time += t2-t1
        wte_time  += t3-t2
        fwd_time += t4-t3
        sample_time += t5-t4
        all_time += t5-t0
        n_c += 1
        # xt = x0
        # xt[xt_is_mask] = img_mask_id
    xt = x0.clone()
    # print('cat_cfg_time',cat_cfg_time/n_c)
    # print('post_cfg_time',post_cfg_time/n_c)
    # print('wte_time',wte_time/n_c)
    # print('fwd_time',fwd_time/n_c)
    # print('sample_time',sample_time/n_c)
    # print('all_time',all_time/n_c)
    guidance_scale = guidance_scale_refinement
    for _ in tqdm(range(n_refinement)):
        mask = torch.rand_like(xt,dtype=torch.float) < remask_ratio
        # print(mask)
        xt[mask] = img_mask_id
        n_refinement -= 1
        local_temp = 1 #sch_temperatures[temp_idx]
        local_temp_samp = 0.8 #sch_temperatures_samp[temp_idx]
        # temp_idx += 1
       
        mask_idx = xt == img_mask_id
        # print(mask_idx.sum())
        if is_unitok and not use_3d:
            mask_idx = mask_idx[:,0,:] 
        n_mask = mask_idx.sum()
        do_cfg = guidance_scale > 0
        if do_cfg:
            input_embeddings_input = torch.cat([inputs_embeds_uncond,inputs_embeds_uncond_enc,inputs_embeds]).clone()
            # nothing, masking vq tokens, final inputs
            xt_input = torch.cat([xt,xt,xt])
            new_token_mask = is_gen.repeat(3,1)
            is_gen_enc_null = torch.zeros_like(is_gen_enc,dtype=torch.bool)
            is_gen_enc_mask = torch.cat([is_gen_enc_null,is_gen_enc_ccc,is_gen_enc])
        else:
            input_embeddings_input = inputs_embeds.clone()
            # x_txt_raw_input = x_txt_raw
            new_token_mask = is_gen
            xt_input = xt
            is_gen_enc_mask = is_gen_enc
        # breakpoint()
        enc_use_image_branch = getattr(model.get_model().config,'enc_use_image_branch',False)
        if enc_use_image_branch:
            modality_indices = new_token_mask| is_gen_enc_mask
        else:
            modality_indices = new_token_mask
        all_input_embeddings,new_token_mask = wte(model.get_model(),None,True,x_gen=xt_input,gen_shape=gen_shape,inputs_embeds_curr=input_embeddings_input,new_token_mask=new_token_mask)
        logits = get_logits(model.get_model(),all_input_embeddings,new_token_mask,True,gen_shape=gen_shape,input_modality_indices=modality_indices)
        if do_cfg:
            new_token_mask,_,_ = new_token_mask.chunk(3)
            logits_un,logits_un_enc,logits = logits.chunk(3)
            logits_is_ninf = logits == -np.inf
            #logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)
            if edit_mode in [0, 3]:
                logits_cond = (1 + guidance_scale_image) * logits - guidance_scale_image * logits_un_enc
            elif edit_mode in [1,2]:
                logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)                    
            #logits_cond = logits_cond
            logits = (1 + guidance_scale) * logits_cond - guidance_scale * logits_un
            logits[logits_is_ninf] = -np.inf
        probs = logits.softmax(dim=-1) # N L 8 D
        if sample_policy == 'multinomial':
            _temperature = temperature
            _temperature = _temperature * local_temp_samp_refinement
            x0 = dists.Categorical(logits=logits/_temperature).sample()
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        elif sample_policy == 'argmax':
            x0 = logits.argmax(-1)
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        else:
            raise NotImplementedError
        xt[mask] = x0[mask]
    
    # print(( xt == img_mask_id).sum()) # hack
    xt[xt==img_mask_id] = x0[xt==img_mask_id]
    if is_unitok:
        x0_img = x0 #-OFFSET
    else:
        x0_img = xt[None]
    
    # print(x0_img.shape)
    # # breakpoint()
    return Image.fromarray(model.decode_image_gen(x0_img,image_resolution,image_resolution)[0]), input_log


@torch.no_grad()
def regional_text2edit_batch(
    model,
    prompts,  # List[str]
    init_images,  # List[PIL.Image]
    rotated_rects,  # List[tuple] - per-sample rotated rects
    original_img_sizes,  # List[tuple] - per-sample (w, h)
    sample_policy='multinomial',
    confidence_policy='mmada',
    guidance_scale=5,
    n_steps=20,
    tokenizer=None,
    image_resolution=512,
    n_tokens=1024,
    shift=3,
    alg_temp=1,
    schedule='shift',
    min_temperature=0.01,
    dynamic_temperature=False,
    temperature=1,
    schedule_temp='linear',
    shift_alg=None,
    top_p=None,
    top_k=None,
    use_3d=False,
    schedule_temp_samp='linear',
    dynamic_temperature_samp=False,
    min_temperature_samp=1,
    cfg_interval=[0, 1],
    order_cutoff=100,
    image_processor=None,
    guidance_scale_image=5,
    edit_mode=0,
    template='Generate an image with the caption:\n <prompt>',
    enable_stratified=False,
    plans=None,  # List[str] or None
    *args,
    **kwargs
):
    """
    Batched version of regional_text2edit for generating multiple region-edited
    images in parallel.

    Args:
        prompts: List of prompt strings
        init_images: List of PIL Images (padded/resized to square)
        rotated_rects: List of rotated rect tuples ((cx, cy), (w, h), angle)
        original_img_sizes: List of (width, height) tuples for coordinate mapping
        plans: List of plan strings or None
        ... (other args same as regional_text2edit)

    Returns:
        List of PIL Images, List of input_logs
    """
    if shift_alg is None:
        shift_alg = shift

    device = model.get_model().device
    batch_size = len(prompts)

    assert len(init_images) == batch_size, "init_images must match prompts length"
    assert len(rotated_rects) == batch_size, "rotated_rects must match prompts length"
    assert len(original_img_sizes) == batch_size, "original_img_sizes must match prompts length"

    if plans is None:
        plans = [''] * batch_size

    conv_template = "llada"
    reserve_token = '<|reserved_token_5|>'
    reserve_id = 126089
    reserve_id2 = 126090
    img_mask_id = 8193
    txt_mask_id = 126336

    gen_shape_map = {
        1024: (64, 64),
        512: (32, 32),
        256: (16, 16),
    }
    gen_shape = gen_shape_map[image_resolution]
    n_tokens_txt = 1024 if image_resolution == 1024 else n_tokens

    is_unitok = 'unitok' in model.get_model().config.mm_vqvae

    # ── Per-sample preparation ──
    all_inputs_embeds = []
    all_inputs_embeds_uncond = []
    all_inputs_embeds_uncond_enc = []
    all_is_gen = []
    all_is_gen_enc = []
    all_is_gen_enc_ccc = []
    all_is_prompt = []
    all_xt = []
    input_logs = []
    max_seq_len = 0

    # wte returns fp32 under FSDP summon_full_params (master weight dtype); cast to bfloat16
    # so later index-puts into bf16 inputs_embeds tensors don't raise dtype mismatch errors.
    noise_embed = model.model.transformer.wte(torch.tensor([txt_mask_id]).to(device)).to(torch.bfloat16)
    all_tokens_to_edit = []
    for idx in range(batch_size):
        prompt = prompts[idx]
        init_image = init_images[idx]
        rotated_rect = rotated_rects[idx]
        original_img_size = original_img_sizes[idx]
        plan = plans[idx]

        # Format prompt (no feedback, no micro_cond for edit)
        full_prompt = f"{prompt} "
        if template is not None:
            question = template.replace('<prompt>', full_prompt)
        else:
            question = full_prompt

        # Process init_image for understanding branch
        image = init_image
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

        image = image.convert('RGB')
        image_1024 = pad_to_square_and_resize(image, image_resolution)
        enc_latents, _gen_shape = model.encode_image_gen(
            model.model.image_processor_gen.preprocess(image_1024).to(device, model.dtype), enc=True
        )
        enc_embeddings = model.model.call_gen_embedding(enc_latents, _gen_shape, enc=True)

        # Build conversation
        conv = copy.deepcopy(conv_templates[conv_template])
        reserve_token_2 = '<|reserved_token_6|>'
        conv.append_message(conv.roles[0], f"<image> {reserve_token_2 * enc_embeddings.shape[1]}\n {full_prompt} ")
        conv.append_message(conv.roles[1], f"{plan}{reserve_token * n_tokens_txt}")
        prompt_question = conv.get_prompt()

        input_log = prompt_question.replace('<|reserved_token_5|>', '@').replace('<|reserved_token_6|>', '*')
        input_logs.append(input_log)

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        image_sizes = [image.size]
        modalities = ["image"]

        (_, position_ids, attention_mask, _, inputs_embeds, _, raw_input_ids) = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            images=image_tensor,
            modalities=modalities,
            image_sizes=image_sizes,
            return_inputs=True
        )

        # Under FSDP summon_full_params, prepare_inputs_labels_for_multimodal returns fp32
        # (vision tower has explicit .float() casts). Cast to match enc_embeddings compute dtype.
        inputs_embeds = inputs_embeds.to(enc_embeddings.dtype)
        # Replace reserve tokens with enc_embeddings
        inputs_embeds[raw_input_ids == reserve_id2] = 0
        _enc = pad_along_last_dim(enc_embeddings, inputs_embeds.shape[-1])
        inputs_embeds[raw_input_ids == reserve_id2] = _enc.flatten(0, 1)

        # Find prompt cutoff
        is_eot = torch.where(raw_input_ids == 126348)[1]
        assert len(is_eot) == 3
        prompt_cutoff = is_eot[1]

        is_prompt = torch.zeros_like(raw_input_ids, dtype=torch.bool)
        is_prompt[:, :prompt_cutoff + 1] = True
        is_gen = raw_input_ids == reserve_id
        is_gen_enc = raw_input_ids == reserve_id2

        if plan:
            is_gen_min = torch.where(raw_input_ids == reserve_id)[1][0]
            is_prompt[:, :is_gen_min] = True

        # Create unconditional embeddings
        inputs_embeds_uncond = inputs_embeds.clone()
        inputs_embeds_uncond[is_prompt] = noise_embed

        inputs_embeds_uncond_enc = inputs_embeds.clone()
        if edit_mode == 0:
            inputs_embeds_uncond_enc[~is_gen_enc] = noise_embed
            is_gen_enc_ccc = is_gen_enc.clone()
        elif edit_mode == 1:
            inputs_embeds_uncond_enc[is_gen_enc] = noise_embed
            is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
        elif edit_mode == 2:
            inputs_embeds_uncond_enc[is_gen_enc | (raw_input_ids < 0)] = noise_embed
            is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
        elif edit_mode == 3:
            inputs_embeds_uncond_enc[(~is_gen_enc) & (raw_input_ids > 0)] = noise_embed
            is_gen_enc_ccc = is_gen_enc.clone()
        else:
            raise ValueError(f"Not Supported edit_mode: {edit_mode}")

        # ── Initialize xt with region masking ──
        init_latents, _ = model.encode_image_gen(
            model.model.image_processor_gen.preprocess(init_image).to(device, model.dtype)
        )

        x0, y0, x1, y1 = rotated_rect
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        grid_h, grid_w = gen_shape
        sx = grid_w / float(image_resolution)
        sy = grid_h / float(image_resolution)

        gx0 = int(np.floor(x0 * sx))
        gy0 = int(np.floor(y0 * sy))
        gx1 = int(np.ceil(x1 * sx))
        gy1 = int(np.ceil(y1 * sy))

        gx0 = min(max(gx0, 0), grid_w - 1)
        gy0 = min(max(gy0, 0), grid_h - 1)
        gx1 = min(max(gx1, gx0 + 1), grid_w)
        gy1 = min(max(gy1, gy0 + 1), grid_h)

        rows = torch.arange(gy0, gy1, device=init_latents.device, dtype=torch.long)
        cols = torch.arange(gx0, gx1, device=init_latents.device, dtype=torch.long)
        tokens = (rows[:, None] * grid_w + cols[None, :]).reshape(-1)
        
        xt_sample = init_latents.clone()  # [1, n_tokens]
    
        xt_sample[:, tokens] = img_mask_id

        max_seq_len = max(max_seq_len, inputs_embeds.shape[1])

        all_inputs_embeds.append(inputs_embeds)
        all_inputs_embeds_uncond.append(inputs_embeds_uncond)
        all_inputs_embeds_uncond_enc.append(inputs_embeds_uncond_enc)
        all_is_gen.append(is_gen)
        all_is_gen_enc.append(is_gen_enc)
        all_is_gen_enc_ccc.append(is_gen_enc_ccc)
        all_is_prompt.append(is_prompt)
        all_xt.append(xt_sample)
        all_tokens_to_edit.append(len(tokens))

    # ── Pad and stack into batched tensors ──
    def pad_tensor(t, target_len, pad_value=0):
        if t is None:
            return None
        curr_len = t.shape[1]
        if curr_len >= target_len:
            return t
        pad_size = target_len - curr_len
        if t.dtype == torch.bool:
            padding = torch.zeros(t.shape[0], pad_size, device=t.device, dtype=torch.bool)
        elif len(t.shape) == 3:
            padding = torch.zeros(t.shape[0], pad_size, t.shape[2], device=t.device, dtype=t.dtype)
        else:
            padding = torch.full((t.shape[0], pad_size), pad_value, device=t.device, dtype=t.dtype)
        return torch.cat([padding, t], dim=1)  # Left padding

    inputs_embeds = torch.cat([pad_tensor(e, max_seq_len) for e in all_inputs_embeds], dim=0)
    inputs_embeds_uncond = torch.cat([pad_tensor(e, max_seq_len) for e in all_inputs_embeds_uncond], dim=0)
    inputs_embeds_uncond_enc = torch.cat([pad_tensor(e, max_seq_len) for e in all_inputs_embeds_uncond_enc], dim=0)
    is_gen = torch.cat([pad_tensor(e, max_seq_len) for e in all_is_gen], dim=0)
    is_gen_enc = torch.cat([pad_tensor(e, max_seq_len) for e in all_is_gen_enc], dim=0)
    is_gen_enc_ccc = torch.cat([pad_tensor(e, max_seq_len) for e in all_is_gen_enc_ccc], dim=0)
    is_prompt = torch.cat([pad_tensor(e, max_seq_len) for e in all_is_prompt], dim=0)

    # Stack xt - each sample may have different mask patterns
    xt = torch.cat(all_xt, dim=0)  # [batch_size, n_tokens]

    # ── Setup denoising schedule ──
    # Use first sample's mask pattern for schedule (they may differ per sample)
    mask_idx = xt == img_mask_id
    if is_unitok:
        mask_idx = mask_idx[:, 0, :]

    # Per-sample mask counts for schedule - use max to ensure enough steps
    n_mask_per_sample = mask_idx.sum(dim=1)
    max_n_mask = n_mask_per_sample.max().item()

    # Build schedule from the sample with most masks
    n_steps = max(1, int(64 * max(all_tokens_to_edit) / 4096))
    schedule_mask = torch.zeros(1, mask_idx.shape[1], dtype=torch.bool, device=device)
    schedule_mask[0, :max_n_mask] = True
    if schedule == 'shift':
        num_transfer_tokens = get_num_transfer_tokens_sch(schedule_mask, n_steps, schedule='shift', schedule_kwargs=dict(shift=shift))
    else:
        num_transfer_tokens = get_num_transfer_tokens_sch(schedule_mask, n_steps, schedule=schedule, schedule_kwargs=dict(shift=shift))

    # Temperature schedules
    sch_temperatures = torch.linspace(0, 1, n_steps, device='cpu').numpy()
    if schedule_temp == 'linear':
        sch_temperatures = (1 - sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp == 'cosine2':
        sch_temperatures = cosine_schedule_2(1 - sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp == 'shift':
        sch_temperatures = logit_normal_schedule(shift_alg, 1 - sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp == 'exp':
        sch_temperatures = exp_schedule(1 - sch_temperatures) * (1 - min_temperature) + min_temperature
    else:
        raise NotImplementedError(f"Unknown schedule_temp: {schedule_temp}")

    sch_temperatures_samp = torch.linspace(0, 1, n_steps, device='cpu').numpy()
    if schedule_temp_samp == 'linear':
        sch_temperatures_samp = (1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'cosine2':
        sch_temperatures_samp = cosine_schedule_2(1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'shift':
        sch_temperatures_samp = logit_normal_schedule(shift_alg, 1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'exp':
        sch_temperatures_samp = exp_schedule(1 - sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    else:
        raise NotImplementedError(f"Unknown schedule_temp_samp: {schedule_temp_samp}")

    sch_temperatures = torch.tensor(sch_temperatures, device=device, dtype=torch.float32)
    sch_temperatures_samp = torch.tensor(sch_temperatures_samp, device=device, dtype=torch.float32)

    cfg_start = int(cfg_interval[0] * n_steps)
    cfg_end = int(cfg_interval[1] * n_steps)

    if confidence_policy == 'stratified':
        _dim = int(np.sqrt(n_tokens))
        unmask_order = stratified_random(n=_dim, seed=42, shuffle_blocks=True)

    # ── Denoising loop ──
    temp_idx = 0
    for num_transfer in tqdm(num_transfer_tokens[0], desc=f"Region editing {batch_size} images"):
        local_temp = sch_temperatures[temp_idx]
        local_temp_samp = sch_temperatures_samp[temp_idx]
        temp_idx += 1

        if temp_idx / n_steps > order_cutoff:
            confidence_policy = 'mmada'

        mask_idx = xt == img_mask_id
        if is_unitok and not use_3d:
            mask_idx = mask_idx[:, 0, :]

        # Per-sample timesteps
        n_mask_per_sample = mask_idx.sum(dim=1)
        timesteps = n_mask_per_sample.float() / mask_idx.shape[1]

        # CFG setup
        do_cfg = guidance_scale > 0 and cfg_start <= temp_idx <= cfg_end

        if do_cfg:
            input_embeddings_input = torch.cat([inputs_embeds_uncond, inputs_embeds_uncond_enc, inputs_embeds], dim=0)
            xt_input = torch.cat([xt, xt, xt], dim=0)
            new_token_mask = torch.cat([is_gen, is_gen, is_gen], dim=0)
            is_gen_enc_null = torch.zeros_like(is_gen_enc, dtype=torch.bool)
            is_gen_enc_mask = torch.cat([is_gen_enc_null, is_gen_enc_ccc, is_gen_enc], dim=0)
            timesteps_input = timesteps.repeat(3)
        else:
            input_embeddings_input = inputs_embeds.clone()
            new_token_mask = is_gen
            xt_input = xt
            is_gen_enc_mask = is_gen_enc
            timesteps_input = timesteps

        enc_use_image_branch = getattr(model.get_model().config, 'enc_use_image_branch', False)
        if enc_use_image_branch:
            modality_indices = new_token_mask | is_gen_enc_mask
        else:
            modality_indices = new_token_mask

        all_input_embeddings, new_token_mask = wte(
            model.get_model(), None, True, x_gen=xt_input, gen_shape=gen_shape,
            inputs_embeds_curr=input_embeddings_input, new_token_mask=new_token_mask
        )
        logits = get_logits(
            model.get_model(), all_input_embeddings, new_token_mask, True,
            gen_shape=gen_shape, input_modality_indices=modality_indices, timesteps=timesteps_input
        )

        # Apply CFG
        if do_cfg:
            new_token_mask = new_token_mask[:batch_size]
            logits_un, logits_un_enc, logits = logits.chunk(3, dim=0)
            logits_is_ninf = logits == -np.inf
            if edit_mode in [0, 3]:
                logits_cond = (1 + guidance_scale_image) * logits - guidance_scale_image * logits_un_enc
            else:
                logits_cond = (logits + guidance_scale_image * logits_un_enc) / (1 + guidance_scale_image)
            logits = (1 + guidance_scale) * logits_cond - guidance_scale * logits_un
            logits[logits_is_ninf] = -np.inf

        # Top-k/top-p filtering
        if top_p is not None or top_k is not None:
            _b, _l, _v = logits.shape
            logits = logits.view(_b * _l, _v)
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, filter_value=-np.inf, min_tokens_to_keep=1)
            logits = logits.view(_b, _l, _v)

        if is_unitok:
            logits[..., 4096:] = float('-inf')

        probs = logits.softmax(dim=-1)

        # Sample
        if sample_policy == 'multinomial':
            _temperature = temperature
            if dynamic_temperature_samp:
                _temperature = _temperature * local_temp_samp
            x0 = dists.Categorical(logits=logits / _temperature).sample()
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        elif sample_policy == 'argmax':
            x0 = logits.argmax(-1)
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        else:
            raise NotImplementedError

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

        # Per-sample token selection
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)

        for b in range(batch_size):
            b_mask = mask_idx[b]
            b_n_mask = b_mask.sum().item()
            if b_n_mask == 0:
                continue

            k = min(num_transfer, b_n_mask)

            if confidence_policy == 'mmada':
                _alg_temp = alg_temp
                if dynamic_temperature:
                    _alg_temp = _alg_temp * local_temp
                confidence_b = torch.log(x0_p[b].clamp(1e-20)) + _alg_temp * gumbel_noise(x0_p[b])
                confidence_b = torch.where(b_mask, confidence_b, -np.inf)
                _, select_index = torch.topk(confidence_b, k=k)

            elif confidence_policy == 'mask_git':
                _alg_temp = alg_temp
                if dynamic_temperature:
                    _alg_temp = _alg_temp * local_temp
                confidence_b = x0_p[b] / _alg_temp
                confidence_b = torch.where(b_mask, confidence_b, -np.inf)
                confidence_b = torch.softmax(confidence_b, dim=-1)
                select_index = torch.multinomial(confidence_b.unsqueeze(0), num_samples=k).squeeze(0)

            elif confidence_policy == 'stratified':
                start = n_tokens - b_n_mask
                select_index = torch.tensor(unmask_order[start:start + k], device=device, dtype=torch.long)

            else:
                raise NotImplementedError

            if is_unitok:
                transfer_index[b, :, select_index] = True
            else:
                transfer_index[b, select_index] = True

        xt[transfer_index] = x0[transfer_index]

    # Final cleanup
    xt = x0.clone()
    xt[xt == img_mask_id] = x0[xt == img_mask_id]

    # Decode all images
    if is_unitok:
        x0_img = xt
    else:
        x0_img = xt

    decoded_images = model.decode_image_gen(x0_img, image_resolution, image_resolution)
    result_images = [Image.fromarray(decoded_images[i]) for i in range(batch_size)]

    return result_images, input_logs
