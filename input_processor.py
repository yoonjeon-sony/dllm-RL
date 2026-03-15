from transformers import ProcessorMixin
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token, pad_to_square_and_resize
from llava.model.utils import pad_along_last_dim
from llava.conversation import conv_templates
import torch
import copy


class MyProcessor(ProcessorMixin):
    attributes = []
    optional_attributes = []

    def __init__(self, model, tokenizer, image_processor, image_resolution=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.reserve_id = 126089
        self.reserve_id2 = 126090
        self.reserve_token = "<|reserved_token_5|>"
        self.reserve_token_2 = "<|reserved_token_6|>"
        self.mask_id = 126336
        self.image_resolution = image_resolution
        self.img_mask_id = 8193
        self.txt_mask_id = 126336
        self.plan_range = 126349
        self.conv_template = "llada"
        gen_shape_map = {
            1024: (64, 64),
            512: (32, 32),
            256: (16, 16),
        }
        if image_resolution not in gen_shape_map:
            raise ValueError(f"Unsupported image_resolution: {image_resolution}")
        self.gen_shape = gen_shape_map[image_resolution]

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def _get_device(self):
        return self.model.get_model().device

    def _ensure_image_groups(self, image_groups):
        if image_groups is None:
            raise ValueError("images are required for multimodal processing.")
        normalized = []
        for idx, sample_images in enumerate(image_groups):
            if sample_images is None:
                raise ValueError(f"Missing image for sample index {idx}.")
            if isinstance(sample_images, (list, tuple)):
                sample_list = [img for img in sample_images if img is not None]
            else:
                sample_list = [sample_images]
            if len(sample_list) == 0:
                raise ValueError(f"Empty image list for sample index {idx}.")
            normalized.append(sample_list)
        return normalized

    def _select_single_image(self, sample_image_or_group, idx):
        if sample_image_or_group is None:
            raise ValueError(f"Missing image for sample index {idx} in image_gen mode.")
        if isinstance(sample_image_or_group, (list, tuple)):
            sample_images = [img for img in sample_image_or_group if img is not None]
            if len(sample_images) == 0:
                raise ValueError(f"Empty image list for sample index {idx} in image_gen mode.")
            return sample_images[0]
        return sample_image_or_group

    def _left_pad_2d(self, tensors, pad_value, dtype_):
        if len(tensors) == 0:
            return torch.empty(0, 0, dtype=dtype_, device=self._get_device())
        max_len = max(t.shape[1] for t in tensors)
        out = []
        for t in tensors:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                pad_t = torch.full((t.shape[0], pad_len), pad_value, dtype=dtype_, device=t.device)
                t = torch.cat([pad_t, t], dim=1)
            out.append(t)
        return torch.cat(out, dim=0)

    def _left_pad_3d(self, tensors):
        if len(tensors) == 0:
            return torch.empty(0, 0, 0, dtype=torch.float32, device=self._get_device())
        max_len = max(t.shape[1] for t in tensors)
        out = []
        for t in tensors:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                pad_t = torch.zeros((t.shape[0], pad_len, t.shape[2]), dtype=t.dtype, device=t.device)
                t = torch.cat([pad_t, t], dim=1)
            out.append(t)
        return torch.cat(out, dim=0)

    def prepare_text_only(self, text_batch):
        device = self._get_device()
        encoded = self.tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        inputs_embeds = self.model.get_model().embed_tokens(input_ids)
        return {
            "input_ids": input_ids,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

    def prepare_text_with_images(self, text_batch, image_groups, include_bbox_mask=False):
        device = self._get_device()
        text_batch = list(text_batch)
        image_groups = self._ensure_image_groups(image_groups)

        if len(text_batch) != len(image_groups):
            raise ValueError("texts and images must have the same batch size.")

        flat_images = []
        image_sizes = []

        for idx, (txt, sample_images) in enumerate(zip(text_batch, image_groups)):
            num_image_tokens = txt.count(DEFAULT_IMAGE_TOKEN)
            if num_image_tokens < len(sample_images):
                missing_tokens = len(sample_images) - num_image_tokens
                image_prefix = "\n".join([DEFAULT_IMAGE_TOKEN] * missing_tokens)
                txt = f"{image_prefix}\n{txt}"
            text_batch[idx] = txt

            flat_images.extend(sample_images)
            image_sizes.extend([image.size for image in sample_images])

        image_tensor = process_images(flat_images, self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]
        else:
            image_tensor = [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]

        all_input_ids = [
            tokenizer_image_token(txt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
            for txt in text_batch
        ]
        input_ids = self._left_pad_2d(
            [torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0) for ids in all_input_ids],
            self.tokenizer.pad_token_id,
            torch.long,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        (_, _, attention_mask, _, inputs_embeds, _, raw_input_ids) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, None, image_tensor, ["image"] * len(text_batch), image_sizes=image_sizes, return_inputs=True
        )
        out = {
            "input_ids": input_ids,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        if include_bbox_mask:
            out["bbox_mask"] = raw_input_ids == self.mask_id
        return out

    def prepare_edit_generation_embeddings(self, edit_text_batch, image_batch, edit_mode, image_tokens=None, do_cfg=False):
        device = self._get_device()
        model = self.model
        tokenizer = self.tokenizer
        n_tokens_txt = (self.gen_shape[0] // 2) * (self.gen_shape[1] // 2)

        all_input_ids = []
        all_attention_mask = []
        all_inputs_embeds = []
        all_inputs_embeds_cond = []
        all_inputs_embeds_uncond = []
        all_inputs_embeds_uncond_enc = []
        all_is_gen = []
        all_is_gen_enc = []
        all_is_gen_enc_ccc = []
        all_is_prompt = []

        for idx, text in enumerate(edit_text_batch):
            sample_image = self._select_single_image(image_batch[idx], idx)
            image_for_enc = pad_to_square_and_resize(sample_image.convert("RGB"), self.image_resolution)
            image_tensor = process_images([image_for_enc], self.image_processor, self.model.config)
            image_tensor = [img.to(dtype=torch.bfloat16, device=device) for img in image_tensor]

            model_dtype = getattr(model, "dtype", torch.bfloat16)
            vq_latents = self.model.model.image_processor_gen.preprocess(image_for_enc).to(device=device, dtype=model_dtype)
            enc_latents, _gen_shape = model.encode_image_gen(vq_latents, enc=True)
            enc_embeddings = model.model.call_gen_embedding(enc_latents, _gen_shape, enc=True)

            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], f"<image> {self.reserve_token_2 * enc_embeddings.shape[1]}\n {text} ")
            conv.append_message(conv.roles[1], f"{self.reserve_token * n_tokens_txt}")
            prompt_question = conv.get_prompt().removesuffix("<|start_header_id|>assistant<|end_header_id|>\n\n")

            input_ids = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            (_, _, attention_mask, _, inputs_embeds, _, raw_input_ids) = self.model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=image_tensor,
                modalities=["image"],
                image_sizes=[image_for_enc.size],
                return_inputs=True,
            )
            if attention_mask is None:
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )

            inputs_embeds = inputs_embeds.to(enc_embeddings.dtype)
            inputs_embeds[raw_input_ids == self.reserve_id2] = 0
            enc_pad = pad_along_last_dim(enc_embeddings, inputs_embeds.shape[-1])
            inputs_embeds[raw_input_ids == self.reserve_id2] = enc_pad.flatten(0, 1)

            eot_pos = torch.where(raw_input_ids[0] == 126348)[0]
            if len(eot_pos) >= 2:
                prompt_cutoff = eot_pos[1].item()
            elif len(eot_pos) == 1:
                prompt_cutoff = eot_pos[0].item()
            else:
                gen_pos = torch.where(raw_input_ids[0] == self.reserve_id)[0]
                prompt_cutoff = max(0, gen_pos[0].item() - 1) if len(gen_pos) > 0 else raw_input_ids.shape[1] - 1

            is_prompt = torch.zeros_like(raw_input_ids, dtype=torch.bool)
            is_prompt[:, :prompt_cutoff + 1] = True
            is_gen = raw_input_ids == self.reserve_id
            is_gen_enc = raw_input_ids == self.reserve_id2

            # Prepare CFG embeddings (mirrors region_edit.py logic)
            inputs_embeds_cond = inputs_embeds
            noise_embed = self.model.get_model().transformer.wte(
                torch.tensor([self.mask_id], device=inputs_embeds.device)
            ).to(inputs_embeds.dtype)
            inputs_embeds_uncond = inputs_embeds.clone()
            inputs_embeds_uncond[is_prompt] = noise_embed
            inputs_embeds_uncond_enc = inputs_embeds.clone()
            if edit_mode == 0:
                inputs_embeds_uncond_enc[~is_gen_enc] = noise_embed
                is_gen_enc_ccc = is_gen_enc
            elif edit_mode == 1:
                inputs_embeds_uncond_enc[is_gen_enc] = noise_embed
                is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
            elif edit_mode == 2:
                inputs_embeds_uncond_enc[is_gen_enc | (raw_input_ids < 0)] = noise_embed
                is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
            elif edit_mode == 3:
                inputs_embeds_uncond_enc[(~is_gen_enc) & (raw_input_ids > 0)] = noise_embed
                is_gen_enc_ccc = is_gen_enc
            else:
                raise ValueError(f"Not Supported edit_mode: {edit_mode}")

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask.to(device))
            all_inputs_embeds.append(inputs_embeds)
            all_inputs_embeds_cond.append(inputs_embeds_cond)
            all_inputs_embeds_uncond.append(inputs_embeds_uncond)
            all_inputs_embeds_uncond_enc.append(inputs_embeds_uncond_enc)
            all_is_gen.append(is_gen)
            all_is_gen_enc.append(is_gen_enc)
            all_is_gen_enc_ccc.append(is_gen_enc_ccc)
            all_is_prompt.append(is_prompt)

        input_ids = self._left_pad_2d(all_input_ids, self.tokenizer.pad_token_id, torch.long)
        attention_mask = self._left_pad_2d(all_attention_mask, 0, torch.long)
        inputs_embeds = self._left_pad_3d(all_inputs_embeds)
        inputs_embeds_cond = self._left_pad_3d(all_inputs_embeds_cond)
        inputs_embeds_uncond = self._left_pad_3d(all_inputs_embeds_uncond)
        inputs_embeds_uncond_enc = self._left_pad_3d(all_inputs_embeds_uncond_enc)
        is_gen = self._left_pad_2d(all_is_gen, 0, torch.bool)
        is_gen_enc = self._left_pad_2d(all_is_gen_enc, 0, torch.bool)
        is_gen_enc_ccc = self._left_pad_2d(all_is_gen_enc_ccc, 0, torch.bool)
        is_prompt = self._left_pad_2d(all_is_prompt, 0, torch.bool)
        is_gen_enc_null = torch.zeros_like(is_gen_enc, dtype=torch.bool)

        return {
            "is_gen": is_gen,
            "is_gen_enc": is_gen_enc,
            "is_gen_enc_ccc": is_gen_enc_ccc,
            "is_gen_enc_null": is_gen_enc_null,
            "is_prompt": is_prompt,
            "input_embeds_gen": inputs_embeds,
            "inputs_embeds_cond": inputs_embeds_cond,
            "inputs_embeds_uncond": inputs_embeds_uncond,
            "inputs_embeds_uncond_enc": inputs_embeds_uncond_enc,
            "attention_mask_gen": attention_mask,
            "input_ids_gen": input_ids,
        }

    def __call__(
        self,
        texts,
        grounding_texts=None,
        edit_texts=None,
        images=None,
        edit_mode=0,
        mode="text_gen",
        do_cfg=False,
        **kwargs,
    ):
        if mode == "text_gen":
            return self.prepare_text_only(texts)
        elif mode == "image_gen":
            assert grounding_texts is not None, "grounding_texts is required for image_gen mode"
            assert edit_texts is not None, "edit_texts is required for image_gen mode"
            grounding_batch = self.prepare_text_with_images(
                grounding_texts,
                images,
                include_bbox_mask=True,
            )
            edit_batch = self.prepare_edit_generation_embeddings(
                edit_texts,
                images,
                edit_mode=edit_mode,
                do_cfg=do_cfg,
            )
            return {**grounding_batch, **edit_batch}
        elif mode == "grounding":
            assert texts is not None, "texts is required for grounding mode"
            return self.prepare_text_with_images(grounding_texts, images, include_bbox_mask=True)
        else:
            raise ValueError(f"Invalid mode: {mode}")
