import os
import torch
from typing import Optional

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
        image_gen_kwargs: Optional[dict] = None,
        return_debug: bool = False,
        bbox_postprocess_fn=None,
        reencode_fn=None,
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
            if processing_class is None and reencode_fn is None:
                raise ValueError(
                    "image_gen text continuation requires `processing_class` or `reencode_fn`."
                )

        if generation_batch_size is None:
            generation_batch_size = input_embeds.size(0)

        image_gen_kwargs = dict(image_gen_kwargs or {})
        image_gen_kwargs.pop("return_debug", None)
        debug_dir = image_gen_kwargs.get("debug_dir", "image_gen_debug")
        if device is None:
            device = input_embeds.device

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
                batch_bbox_ids, pred_bboxes, bbox_texts = self.model.generate_bbox(
                    tokenizer,
                    batch_input_embeds,
                    batch_bbox_mask,
                )
                if bbox_postprocess_fn is not None:
                    pred_bboxes, bbox_texts = bbox_postprocess_fn(
                        batch_bbox_ids, list(range(i, end_idx))
                    )
                batch_pred_bboxes = pred_bboxes
                bbox_ids_all.append(batch_bbox_ids)

                for row_idx, pred_box in enumerate(pred_bboxes):
                    pred_bboxes_all[i + row_idx] = pred_box

                for row_idx, txt in enumerate(bbox_texts):
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
                    batch_edited_images, batch_image_debug = self.model.generate_image(
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
                    batch_edited_images = self.model.generate_image(
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
                    if batch_image_debug is not None:
                        row_debug = batch_image_debug
                        if isinstance(batch_image_debug, dict):
                            row_debug = dict(batch_image_debug)
                            samples = batch_image_debug.get("samples")
                            if isinstance(samples, list) and row_idx < len(samples):
                                row_debug["sample"] = samples[row_idx]
                            step_images = batch_image_debug.get("step_images")
                            if isinstance(step_images, list) and row_idx < len(step_images):
                                row_debug["step_images"] = step_images[row_idx]
                                save_root = debug_dir
                                debug_label = batch_image_debug.get("debug_label")
                                if debug_label:
                                    save_root = os.path.join(save_root, str(debug_label))
                                os.makedirs(save_root, exist_ok=True)
                                for step_idx, step_image in enumerate(step_images[row_idx]):
                                    step_image.save(
                                        os.path.join(save_root, f"{i + row_idx}_{step_idx}.png")
                                    )
                        image_gen_debug_all[i + row_idx] = row_debug


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
                batch_prompt_mask = batch_attention_mask
            elif gen_type == "grounding":
                batch_prompt_completion_ids = batch_bbox_ids
                batch_prompt_mask = batch_attention_mask
            else:
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

    
