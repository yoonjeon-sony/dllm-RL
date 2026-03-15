import os
import torch
from typing import Optional

def _get_prompt_mask(completion_ids, masks):
    max_prompt_len = max(m.shape[1] for m in masks)
    padded_prompt_masks = []
    for m in masks:
        pad_len = max_prompt_len - m.shape[1]
        if pad_len > 0:
            pad_m = torch.zeros((m.shape[0], pad_len), dtype=m.dtype, device=m.device)
            m = torch.cat([pad_m, m], dim=1)
        padded_prompt_masks.append(m)
    prompt_mask = torch.cat(padded_prompt_masks, dim=0)
    return prompt_mask
    
class InterleavedInferencer:
    def __init__(self, model):
        self.model = model

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
        image_gen_kwargs: Optional[dict] = {},
        return_debug: bool = False,
        processing_class=None,
        max_prompt_length: Optional[int] = None,
        device=None,
    ):
        if gen_type not in ("text_gen", "image_gen", "grounding"):
            raise ValueError(f"gen_type must be 'text_gen', 'image_gen', or 'grounding', got {gen_type!r}")
        if attention_mask is None or input_embeds is None:
            raise ValueError("input_embeds and attention_mask are required.")
        if gen_type in {"grounding", "image_gen"} and bbox_mask is None:
            raise ValueError("bbox_mask is required for grounding and image_gen modes.")
        if gen_type == "image_gen":
            required_inputs = {
                "input_embeds_gen": input_embeds_gen,
                "attention_mask_gen": attention_mask_gen,
                "is_gen": is_gen,
                "is_gen_enc": is_gen_enc,
                "is_prompt": is_prompt,
                "init_images": init_images,
                "generation_prompts": generation_prompts,
            }
            missing = [name for name, value in required_inputs.items() if value is None]
            if missing:
                raise ValueError(f"image_gen is missing required inputs: {', '.join(missing)}")

        if generation_batch_size is None:
            generation_batch_size = input_embeds.size(0)

        if device is None:
            device = input_embeds.device

        total = input_embeds.size(0)
        prompt_completion_ids_all = []
        grounding_completion_ids_all, grounding_masks_all, pred_bboxes_all, bbox_texts_all = [], [], [], []
        image_completion_ids_all = []
        image_masks_all = []
        prompt_masks_all = []
        edited_images_all = []
        

        for i in range(0, total, generation_batch_size):
            end_idx = min(i + generation_batch_size, total)
            batch_input_embeds = input_embeds[i:end_idx]
            batch_attention_mask = attention_mask[i:end_idx]

            if gen_type in {"grounding", "image_gen"}:
                #! Grounding Generation
                batch_bbox_mask = bbox_mask[i:end_idx]
                batch_bbox_ids, pred_bboxes, bbox_texts = self.model.generate_bbox(
                    tokenizer,
                    batch_input_embeds,
                    batch_bbox_mask,
                )
                batch_pred_bboxes = pred_bboxes
                grounding_completion_ids_all.append(batch_bbox_ids)
                grounding_masks_all.append(batch_attention_mask)
                pred_bboxes_all.extend(pred_bboxes)
                bbox_texts_all.extend(bbox_texts)

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
                #! Image Generation
                batch_edited_images, batch_image_completion_ids, batch_edit_region_mask = self.model.generate_image(
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
                image_completion_ids_all.append(batch_image_completion_ids)
                image_masks_all.append(batch_edit_region_mask)
                edited_images_all.extend(batch_edited_images)

                #! Text Generation
                batch_generation_prompts = generation_prompts[i:end_idx]
                batch_all_images = [[orig, edited] for orig, edited in zip(batch_images, batch_edited_images)]
                
                # TODO: Append the edited image into conv.roles[1] (assistant) and continue text generation from this prompt.
                re_batch_inputs = processing_class(
                    texts=batch_generation_prompts,
                    images=batch_all_images,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                    device=device,
                    dtype=torch.bfloat16,
                    mask_id=mask_id,
                    mode="text_gen",
                )
                batch_input_embeds = re_batch_inputs["input_embeds"]
                batch_attention_mask = re_batch_inputs["attention_mask"]
                if max_prompt_length is not None:
                    batch_input_embeds = batch_input_embeds[:, -max_prompt_length :]
                    batch_attention_mask = batch_attention_mask[:, -max_prompt_length :]

                batch_prompt_completion_ids = self.model.generate_text(
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
                
                prompt_completion_ids_all.append(batch_prompt_completion_ids)
                prompt_masks_all.append(batch_attention_mask)

        result = {}
        if prompt_completion_ids_all:
            completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
            prompt_mask = _get_prompt_mask(completion_ids, prompt_masks_all)
            result.update({
                "completion_ids": completion_ids,
                "prompt_mask": prompt_mask,
            })
        if gen_type in {"grounding", "image_gen"}:
            ground_completion_ids = torch.cat(grounding_completion_ids_all, dim=0)
            ground_mask = _get_prompt_mask(ground_completion_ids, grounding_masks_all)
            result.update({
                "ground_completion_ids": ground_completion_ids,
                "ground_mask": ground_mask,
                "bbox_texts": bbox_texts_all,
                "pred_bboxes": pred_bboxes_all,
            })
        if gen_type == "image_gen":
            edit_completion_ids = torch.cat(image_completion_ids_all, dim=0)
            edit_region_mask = torch.cat(image_masks_all, dim=0)
            result.update({
                "edit_completion_ids": edit_completion_ids,
                "edit_region_mask": edit_region_mask,
                "edited_images": edited_images_all,
                "image_input_embeds_gen": input_embeds_gen,
                "image_attention_mask_gen": attention_mask_gen,
                "image_is_gen": is_gen,
                "image_is_gen_enc": is_gen_enc,
            })

        return result
    
