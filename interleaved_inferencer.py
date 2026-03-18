import os
from pathlib import Path
import torch
from typing import Optional

from llava.constants import DEFAULT_IMAGE_TOKEN
from log_utils import _format_image_gen_completion_log, _format_image_gen_prompt_log

def _log_text_samples_rich(prompts, completions, batch_idx):
    if not prompts:
        return
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.rule import Rule
    except ImportError as exc:
        raise ImportError("rich is required for terminal logging.") from exc

    console = Console()
    console.print(Rule(f"[bold cyan]Interleaved Inference Batch {batch_idx}[/bold cyan]"))
    for sample_idx, (prompt_text, completion_text) in enumerate(zip(prompts, completions), start=1):
        console.print(
            Panel(
                "" if prompt_text is None else str(prompt_text),
                title=f"[bold blue]Prompt {sample_idx}[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print(
            Panel(
                "" if completion_text is None else str(completion_text),
                title=f"[bold green]Completion {sample_idx}[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

def _sanitize_token_ids_for_decode(token_ids, tokenizer):
    replacement_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    if replacement_id is None or replacement_id < 0 or replacement_id == tokenizer.unk_token_id:
        replacement_id = tokenizer.pad_token_id
    if replacement_id is None:
        replacement_id = tokenizer.eos_token_id
    if replacement_id is None:
        replacement_id = 0

    vocab_size = None
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = getattr(tokenizer, "vocab_size", None)

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.clone().to(dtype=torch.long)
        token_ids[token_ids < 0] = replacement_id
        if vocab_size is not None:
            token_ids[token_ids >= vocab_size] = replacement_id
        return token_ids

    sanitized_ids = []
    for seq in token_ids:
        sanitized_seq = []
        for token_id in seq:
            token_id = int(token_id)
            if token_id < 0 or (vocab_size is not None and token_id >= vocab_size):
                token_id = replacement_id
            sanitized_seq.append(token_id)
        sanitized_ids.append(sanitized_seq)
    return sanitized_ids

class InterleavedInferencer:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def _generate_mode(
        self,
        gen_type: str,
        tokenizer,
        # text generation inputs
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # grounding inputs
        input_embeds_grounding: torch.Tensor = None,
        input_ids_grounding: torch.Tensor = None,
        attention_mask_grounding: torch.Tensor = None,
        bbox_mask_grounding: Optional[torch.Tensor] = None,
        # image editing inputs
        input_ids_gen: Optional[torch.Tensor] = None,
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
        answer_prompts: Optional[list] = None,
    ):
        if generation_batch_size is None:
            generation_batch_size = input_embeds.size(0)

        if device is None:
            device = input_embeds.device

        total = input_embeds.size(0)
        prompt_completion_ids_all = []
        grounding_completion_ids_all, pred_bboxes_all, bbox_texts_all = [], [], []
        image_completion_ids_all, image_masks_all, edited_images_all = [], [], []

        for i in range(0, total, generation_batch_size):
            end_idx = min(i + generation_batch_size, total)
            if gen_type == "text_gen":
                batch_input_embeds = input_embeds[i:end_idx]
                batch_attention_mask = attention_mask[i:end_idx]
            elif gen_type == "image_gen":
                #! Grounding Generation
                batch_input_embeds_grounding = input_embeds_grounding[i:end_idx]
                batch_attention_mask = attention_mask_grounding[i:end_idx]
                batch_bbox_mask = bbox_mask_grounding[i:end_idx]
                batch_bbox_ids, pred_bboxes, bbox_texts = self.model.generate_bbox(
                    tokenizer,
                    batch_input_embeds_grounding,
                    batch_bbox_mask,
                )
                batch_pred_bboxes = pred_bboxes
                grounding_completion_ids_all.append(batch_bbox_ids)
                pred_bboxes_all.extend(pred_bboxes)
                bbox_texts_all.extend(bbox_texts)

                #! Image Editing Generation
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
                batch_answer_prompts = answer_prompts[i:end_idx]
                batch_all_images = [[orig, edited] for orig, edited in zip(batch_images, batch_edited_images)]
                
                re_batch_inputs = processing_class(
                    texts=batch_answer_prompts,
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
        
        completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
        result = {
            "completion_ids": completion_ids,
        }
        if gen_type == "image_gen":
            ground_completion_ids = torch.cat(grounding_completion_ids_all, dim=0)
            edit_completion_ids = torch.cat(image_completion_ids_all, dim=0)
            edit_region_mask = torch.cat(image_masks_all, dim=0)
            result.update({
                "ground_completion_ids": ground_completion_ids,
                "bbox_texts": bbox_texts_all,
                "pred_bboxes": pred_bboxes_all,
                "edit_completion_ids": edit_completion_ids,
                "edit_region_mask": edit_region_mask,
                "edited_images": edited_images_all,
            })

        return result

    @torch.no_grad()
    def _generate_mode_mmada(
        self,
        gen_type: str,
        tokenizer,
        # text generation inputs
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # image generation inputs
        input_ids_gen: Optional[torch.Tensor] = None,
        attention_mask_gen: Optional[torch.Tensor] = None,
        uncond_input_ids_gen: Optional[torch.Tensor] = None,
        uncond_attention_mask_gen: Optional[torch.Tensor] = None,
        # compatibility aliases
        uncond_input_ids: Optional[torch.Tensor] = None,
        uncond_attention_mask: Optional[torch.Tensor] = None,
        mask_schedule=None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        generation_batch_size: Optional[int] = None,
        image_gen_kwargs: Optional[dict] = None,
        processing_class=None,
    ):
        from PIL import Image

        if image_gen_kwargs is None:
            image_gen_kwargs = {}

        if generation_batch_size is None:
            generation_batch_size = input_ids.size(0)

        if uncond_input_ids_gen is None:
            uncond_input_ids_gen = uncond_input_ids
        if uncond_attention_mask_gen is None:
            uncond_attention_mask_gen = uncond_attention_mask

        total = input_ids.size(0)
        prompt_completion_ids_all = []
        image_completion_ids_all, edited_images_all = [], []

        if gen_type == "image_gen" and processing_class is None:
            raise ValueError("processing_class is required for MMADA image generation.")

        for i in range(0, total, generation_batch_size):
            end_idx = min(i + generation_batch_size, total)

            if gen_type == "image_gen":
                if input_ids_gen is None or attention_mask_gen is None:
                    raise ValueError("input_ids_gen and attention_mask_gen are required for MMADA image_gen.")

                batch_input_ids_gen = input_ids_gen[i:end_idx]
                batch_attention_mask_gen = attention_mask_gen[i:end_idx]
                batch_uncond_input_ids_gen = (
                    uncond_input_ids_gen[i:end_idx] if uncond_input_ids_gen is not None else None
                )
                batch_uncond_attention_mask_gen = (
                    uncond_attention_mask_gen[i:end_idx] if uncond_attention_mask_gen is not None else None
                )

                seq_len = image_gen_kwargs.get(
                    "seq_len",
                    getattr(processing_class, "num_vq_tokens", getattr(self.model.config, "num_vq_tokens", 1024)),
                )
                codebook_size = image_gen_kwargs.get(
                    "codebook_size",
                    getattr(processing_class, "codebook_size", getattr(self.model.config, "codebook_size", 8192)),
                )

                mmada_image_kwargs = dict(image_gen_kwargs)
                mmada_image_kwargs.pop("seq_len", None)
                mmada_image_kwargs.pop("codebook_size", None)

                image_generate_kwargs = {
                    "input_ids": batch_input_ids_gen,
                    "uncond_input_ids": batch_uncond_input_ids_gen,
                    "attention_mask": batch_attention_mask_gen,
                    "uncond_attention_mask": batch_uncond_attention_mask_gen,
                    "guidance_scale": mmada_image_kwargs.pop("guidance_scale", cfg_scale),
                    "temperature": mmada_image_kwargs.pop("temperature", temperature),
                    "timesteps": mmada_image_kwargs.pop("timesteps", steps),
                    "seq_len": seq_len,
                    "mask_token_id": mmada_image_kwargs.pop("mask_token_id", mask_id),
                    "codebook_size": codebook_size,
                    "uni_prompting": mmada_image_kwargs.pop(
                        "uni_prompting",
                        getattr(processing_class, "uni_prompting", None),
                    ),
                }
                image_noise_schedule = mmada_image_kwargs.pop("noise_schedule", mask_schedule)
                if image_noise_schedule is not None:
                    image_generate_kwargs["noise_schedule"] = image_noise_schedule
                image_generate_kwargs.update(mmada_image_kwargs)

                batch_image_completion_ids = self.model.generate_image(**image_generate_kwargs)
                batch_image_completion_ids = torch.clamp(
                    batch_image_completion_ids,
                    min=0,
                    max=codebook_size - 1,
                ).to(dtype=torch.long)
                image_completion_ids_all.append(batch_image_completion_ids)

                decoded_images = processing_class.vq_model.decode_code(batch_image_completion_ids)
                decoded_images = torch.clamp((decoded_images + 1.0) / 2.0, min=0.0, max=1.0)
                decoded_images = (decoded_images * 255.0).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
                edited_images_all.extend([Image.fromarray(image) for image in decoded_images])

            batch_input_ids = input_ids[i:end_idx]
            batch_attention_mask = attention_mask[i:end_idx]
            batch_output_ids = self.model.generate_text(
                idx=batch_input_ids,
                max_new_tokens=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                attention_mask=batch_attention_mask,
            )
            batch_prompt_completion_ids = batch_output_ids[:, batch_input_ids.shape[1]:]
            prompt_completion_ids_all.append(batch_prompt_completion_ids)

        completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
        result = {"completion_ids": completion_ids}
        if gen_type == "image_gen":
            result.update(
                {
                    "edit_completion_ids": torch.cat(image_completion_ids_all, dim=0),
                    "edited_images": edited_images_all,
                }
            )
        return result
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data_utils import get_thinkmorph_image_editing_questions
    from diffu_grpo_train import model_init_fn
    from input_processor import LavidaOProcessor, MMADAProcessor
    from llava.mm_utils import pad_to_square_and_resize

    # TODO: Test the interleaved inferencer with four dataset variants of thinkmorph and lmms-eval tasks (vstar_bench, cv_bench, VisualPuzzles, chartqa)
    dataset = get_thinkmorph_image_editing_questions(
        data_root="./data",
        image_root="/group2/dgm/yoonjeon/data",
        gen_type="image_gen",
        per_task_sample=10,
    )

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    gen_type = "image_gen"
    model_path = "/group2/dgm/yoonjeon/MMaDA-8B-Base" # "/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph-complete/checkpoint-2420",
    
    tokenizer, model, image_processor, _ = model_init_fn(
        model_path
    )
    model.to(device)
    is_mmada = "mmada" in model_path.lower()
    if is_mmada:
        processor = MMADAProcessor(model, tokenizer, image_processor, max_seq_length=256)
    else:
        processor = LavidaOProcessor(model, tokenizer, image_processor)

    inferencer = InterleavedInferencer(model)
    is_local_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0

    bsz = 4
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, collate_fn=list)

    output_dir = Path("tmp/mmada") if is_mmada else Path("outputs/tmp")
    if is_local_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        # answer_prompt - answer_gt / edit_prompt - answer_gt / grounding_prompt - grounding_gt
        answer_prompts = [example["answer_prompt"] for example in batch]
        if batch[0].get("grounding_prompt") is not None:
            grounding_prompts = [example["grounding_prompt"] for example in batch]
        else:
            grounding_prompts = None
        if batch[0].get("edit_prompt") is not None:
            edit_prompts = [example["edit_prompt"] for example in batch]
        else:
            edit_prompts = None
        
        if "images" in batch[0]:
            images = [example.get("images") for example in batch]
        elif "image" in batch[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in batch] # single image
        else:
            images = None # no image

        processor_kwargs = dict(
            texts=answer_prompts,
            grounding_texts=grounding_prompts,
            edit_texts=edit_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            device=device,
            dtype=torch.bfloat16,
            mode="image_gen",
        )
        if not is_mmada:
            processor_kwargs["edit_mode"] = 0
            processor_kwargs["do_cfg"] = False

        batch_inputs = processor(**processor_kwargs)
        if not isinstance(batch_inputs, dict):
            raise TypeError("Processor must return a dict for interleaved inference.")

        if is_mmada:
            gen_result = inferencer._generate_mode_mmada(
                gen_type=gen_type,
                tokenizer=tokenizer,
                steps=64,
                gen_length=128,
                block_length=32,
                temperature=0.8,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=getattr(processor, "mask_token_id", 126336),
                generation_batch_size=bsz,
                image_gen_kwargs={
                    "guidance_scale": 0.0,
                    "timesteps": 64,
                    "temperature": 0.8,
                    "seq_len": getattr(processor, "num_vq_tokens", 1024),
                    "codebook_size": getattr(processor, "codebook_size", 8192),
                    "uni_prompting": getattr(processor, "uni_prompting", None),
                },
                processing_class=processor,
                **batch_inputs,
            )
        else:
            target_resolution = 1024
            init_images = []
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

            gen_result = inferencer._generate_mode(
                gen_type=gen_type,
                tokenizer=tokenizer,
                init_images=init_images,
                image_sizes=image_sizes,
                image_gen_kwargs={
                    "guidance_scale": 0.0,
                    "guidance_scale_image": 0.0,
                    "edit_mode": 0,
                    "confidence_policy": "stratified",
                    "enable_stratified": True,
                    "image_resolution": 1024,
                    "n_tokens": 4096,
                    "n_steps": 64,
                    "shift": 5,
                    "schedule": "shift",
                    "alg_temp": 5,
                    "dynamic_temperature": True,
                    "temperature": 0.8,
                    "schedule_temp": "cosine2",
                    "min_temperature": 0.5,
                },
                processing_class=processor,
                device=device,
                answer_prompts=answer_prompts,
                **batch_inputs,
            )

        text_completion_ids = gen_result.get("completion_ids")
        completions_text = tokenizer.batch_decode(
            _sanitize_token_ids_for_decode(text_completion_ids, tokenizer),
            skip_special_tokens=False,
        )

        if is_local_main_process:
            for prompt, completion in zip(answer_prompts, completions_text):
                print(f"Prompt: {prompt}\n")
                print(f"Completion: {completion}\n")
                print("-" * 100)
            if gen_result.get("edited_images") is not None:
                for sample_idx, img in enumerate(gen_result["edited_images"]):
                    task_type = str(batch[sample_idx].get("task_type", "sample")).replace("/", "_").replace(" ", "_")
                    image_path = output_dir / f"{batch_idx:04d}_{sample_idx:02d}_{task_type}.png"
                    img.save(image_path)
