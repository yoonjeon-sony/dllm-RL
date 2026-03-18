from transformers import ProcessorMixin
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token, pad_to_square_and_resize
from llava.model.utils import pad_along_last_dim
from llava.conversation import conv_templates
import torch
import copy


from transformers import ProcessorMixin
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token, pad_to_square_and_resize
from llava.model.utils import pad_along_last_dim
from llava.model.sampling import get_mask_schedule
from llava.model.prompting_utils import UniversalPrompting
from llava.conversation import conv_templates
import torch
import copy

from torchvision import transforms
def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

def image_transform_squash(image, resolution=256, normalize=True):
    image = transforms.Resize((resolution,resolution), interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

class MMADAProcessor(ProcessorMixin):
    
    def __init__(self, model, tokenizer, vq_model, max_seq_len):

        self.model = model # MMadaModelLM
        self.tokenizer = tokenizer
        self.vq_model = vq_model
        self.uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=max_seq_len,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
            ignore_id=-100,
            cond_dropout_prob=0.1,
            use_reserved_token=True
        )
        self.resolution = 512
        self.w_clip_vit = False
        self.new_vocab_size = 134656
        self.llm_vocab_size = 126464
        self.codebook_size = 8192
        self.num_vq_tokens = 1024
        self.num_new_special_tokens = 0
        self.tie_word_embeddings = False
        self.mask_token_id = 126336
        self.guidance_scale = 0 # Maybe make it 5
        self.mask_schedule = "cosine"
        

    def prepare_mmu(self, questions, image_paths):
        
        messages = [[{"role": "user", "content": question}] for question in questions]
        image_list = []
        for image_path in image_paths:
            image_ori = Image.open(image_path).convert("RGB")
            if any(tag in file_name for tag in ['ai2d', 'clevr', 'docvqa', 'geo', 'llava']):
                image = image_transform_squash(image_ori, resolution=self.resolution).to(device)
            else:
                image = image_transform(image_ori, resolution=self.resolution).to(device)
            image_list.append(image)
        image = torch.stack(image_list, dim=0)
        image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)
        text_token_ids = self.uni_prompting.text_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        attention_mask = (text_token_ids != self.uni_prompting.pad_id).long()
        assert image_tokens.shape[0] == text_token_ids.shape[0]
        
        batch_size = image_tokens.shape[0]
        # [task token] [soi] [image tokens] [eoi] [sot] [text tokens] [eot]
        input_ids = torch.cat([
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(device),
            text_token_ids
        ], dim=1).long()
        
        
        
        # output_ids = model.generate_text(
        #     input_ids,
        #     max_new_tokens=config.dataset.preprocessing.max_seq_length,
        #     steps=max(1, config.dataset.preprocessing.max_seq_length // 2),
        #     block_length=max(1, config.dataset.preprocessing.max_seq_length // 4),
        # )
        # generated_ids = output_ids[:, input_ids.shape[1]:]
        # response_text = uni_prompting.text_tokenizer.batch_decode(
        #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]
        # conversation_str = f"Image: {file_name}\n" + "=" * 20 + "\n"
        # for msg in messages:
        #     role_prefix = "User: " if msg.get('role') == 'user' else "Assistant: "
        #     conversation_str += f"{role_prefix}{msg.get('content', '')}\n"
        # conversation_str += f"Assistant (Generated): {response_text}\n"
        # vis_img = torch.clamp((image.squeeze(0) + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
        # vis_img = vis_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return input_ids, attention_mask

    def prepare_t2i(self, questions, image_paths):
        image_tokens = torch.ones((len(questions), self.num_vq_tokens), dtype=torch.long, device="cpu") * mask_token_id
        input_ids, attention_mask = self.uni_prompting((prompts, image_tokens), 't2i_gen')
        if self.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = self.uni_prompting(([''] * len(questions), image_tokens), 't2i_gen')
        else:
            uncond_input_ids = None
            uncond_attention_mask = None
        mask_schedule = get_mask_schedule(self.mask_schedule)
        #  with torch.no_grad():
        #     gen_token_ids = model.generate_image(
        #         input_ids=input_ids,
        #         uncond_input_ids=uncond_input_ids,
        #         attention_mask=attention_mask,
        #         uncond_attention_mask=uncond_attention_mask,
        #         guidance_scale=config.training.guidance_scale,
        #         temperature=config.training.get("generation_temperature", 1.0),
        #         timesteps=config.training.generation_timesteps,
        #         noise_schedule=mask_schedule,
        #         noise_type=config.training.get("noise_type", "mask"),
        #         seq_len=config.model.mmada.num_vq_tokens,
        #         uni_prompting=uni_prompting,
        #         config=config,
        #     )

        # gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
        # images = vq_model.decode_code(gen_token_ids)

        # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # images *= 255.0
        # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # pil_images = [Image.fromarray(image) for image in images]
        return input_ids, attention_mask, uncond_input_ids, uncond_attention_mask, mask_schedule
    
    def __call__(
        self,
        texts,
        grounding_texts=None,
        edit_texts=None,
        images=None,
        mode="text_gen",
        **kwargs,
    ):
        answer_batch = self.prepare_mmu(texts, images)
        if mode=="text_gen":
            return answer_batch
        else:
            grounding_batch = self.prepare_mmu(
                grounding_texts,
                images,
            )
            grounding_batch = {f"{k}_grounding": v for k, v in grounding_batch.items()}
            edit_batch = self.prepare_t2i(
                edit_texts,
                images,
            )
            return {**answer_batch, **grounding_batch, **edit_batch}
        

        
class LavidaOProcessor(ProcessorMixin):
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

    def _right_pad_2d(self, tensors, pad_value, dtype_):
        if len(tensors) == 0:
            return torch.empty(0, 0, dtype=dtype_, device=self._get_device())
        max_len = max(t.shape[1] for t in tensors)
        out = []
        for t in tensors:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                pad_t = torch.full((t.shape[0], pad_len), pad_value, dtype=dtype_, device=t.device)
                t = torch.cat([t, pad_t], dim=1)
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

    def _detach_batch_tensors(self, batch):
        return {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in batch.items()
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
        return self._detach_batch_tensors(out)

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

        return self._detach_batch_tensors({
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
        })

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
        answer_batch = self.prepare_text_with_images(
            texts,
            images,
        )
        if mode=="text_gen":
            return answer_batch
        else:
            grounding_batch = self.prepare_text_with_images(
                grounding_texts,
                images,
                include_bbox_mask=True,
            )
            grounding_batch = {f"{k}_grounding": v for k, v in grounding_batch.items()}
            edit_batch = self.prepare_edit_generation_embeddings(
                edit_texts,
                images,
                edit_mode=edit_mode,
                do_cfg=do_cfg,
            )
            return {**answer_batch, **grounding_batch, **edit_batch}
        
