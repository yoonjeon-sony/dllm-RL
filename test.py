import unittest
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

import diffu_grpo_trainer as trainer_mod


class _FakeTokenizer:
    eos_token_id = 0

    def batch_decode(self, sequences, skip_special_tokens=True):
        outputs = []
        for seq in sequences.tolist():
            if seq[:4] == [101, 102, 103, 104]:
                outputs.append("<LOC_10><LOC_20><LOC_30><LOC_40>")
            elif seq[:3] == [201, 202, 0]:
                outputs.append("<reasoning>\nsteps\n</reasoning>\n<answer>\n42\n</answer>")
            else:
                outputs.append(str(seq))
        return outputs


class _FakeProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def __call__(self, texts=None, images=None, **kwargs):
        self.calls.append(
            {
                "texts": list(texts or []),
                "images": images,
                "kwargs": kwargs,
            }
        )
        batch_size = len(texts or [])
        return {
            "input_embeds": torch.zeros(batch_size, 2, 3, dtype=torch.float32),
            "attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
            "bbox_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        }


class _FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.process_index = 0

    def gather_for_metrics(self, value):
        return value


class _FakeGenerationModel:
    def _generate_mode(self, **kwargs):
        batch_size = kwargs["input_embeds"].size(0)
        completion_ids = torch.tensor([[201, 202, 0], [201, 202, 0]], dtype=torch.long)
        prompt_mask = torch.ones(batch_size, 2, dtype=torch.long)
        if kwargs["gen_type"] == "text_gen":
            return {
                "completion_ids": completion_ids,
                "prompt_mask": prompt_mask,
            }

        bbox_ids = torch.tensor([[101, 102, 103, 104], [101, 102, 103, 104]], dtype=torch.long)
        result = {
            "completion_ids": completion_ids if kwargs["gen_type"] == "image_gen" else bbox_ids,
            "prompt_mask": prompt_mask,
            "bbox_ids": bbox_ids,
            "bbox_texts": ["<LOC_10><LOC_20><LOC_30><LOC_40>"] * batch_size,
            "pred_bboxes": [[10, 20, 30, 40]] * batch_size,
        }
        if kwargs["gen_type"] == "image_gen":
            edited_images = [Image.new("RGB", (8, 8), color="blue") for _ in range(batch_size)]
            kwargs["reencode_fn"](
                kwargs["generation_prompts"],
                [[orig, edited] for orig, edited in zip(kwargs["init_images"], edited_images)],
                0,
                batch_size,
            )
            result["edited_images"] = edited_images
        return result


def _make_dummy_trainer(reward_funcs):
    trainer = SimpleNamespace()
    trainer.accelerator = _FakeAccelerator()
    trainer.processing_class = _FakeProcessor(_FakeTokenizer())
    trainer.args = SimpleNamespace(
        edit_mode=0,
        mask_id=126336,
        max_completion_length=8,
        block_length=4,
        diffusion_steps=4,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        generation_batch_size=2,
        guidance_scale=1.0,
        guidance_scale_image=1.0,
        logging_steps=1,
        report_to=["wandb"],
        output_dir="/tmp/diffu_grpo_test",
        random_masking=False,
        beta=0.0,
        max_prompt_length=None,
        completion_log_sample_ratio=1.0,
        image_log_sample_ratio=1.0,
        image_log_steps=50,
    )
    trainer.reward_funcs = reward_funcs
    trainer.reward_processing_classes = [None] * len(reward_funcs)
    trainer.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
    trainer.num_generations = 2
    trainer.num_iterations = 1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.control = SimpleNamespace(should_evaluate=False)
    trainer._step = 0
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    trainer.log_completions = True
    trainer.model_wrapped = object()
    trainer.model = object()
    return trainer


def _make_inputs(gen_type, prompts=None):
    image = Image.new("RGB", (8, 8), color="white")
    prompts = prompts or ["text prompt 1", "text prompt 2"]
    return [
        {
            "prompt": prompts[0],
            "grounding_prompt": "ground prompt 1",
            "edit_prompt": "edit prompt 1",
            "answer": "42",
            "gt_bbox": [10, 20, 30, 40],
            "image": image,
            "gen_type": gen_type,
        },
        {
            "prompt": prompts[1],
            "grounding_prompt": "ground prompt 2",
            "edit_prompt": "edit prompt 2",
            "answer": "42",
            "gt_bbox": [10, 20, 30, 40],
            "image": image,
            "gen_type": gen_type,
        },
    ]


class DiffuGRPOTrainerGenerationLoggingTest(unittest.TestCase):
    def _patch_runtime(self):
        logged = {}

        @contextmanager
        def fake_unwrap_model_for_generation(model_wrapped, accelerator):
            yield _FakeGenerationModel()

        def fake_wandb_log(payload):
            logged.update(payload)

        patches = [
            patch.object(trainer_mod, "unwrap_model_for_generation", fake_unwrap_model_for_generation),
            patch.object(trainer_mod, "gather", lambda value: value),
            patch.object(trainer_mod, "gather_object", lambda value: value),
            patch.object(trainer_mod, "profiling_context", lambda *args, **kwargs: nullcontext()),
            patch.object(trainer_mod, "is_rich_available", lambda: False),
            patch.object(trainer_mod.wandb, "run", object()),
            patch.object(trainer_mod.wandb, "Image", lambda image: image),
            patch.object(trainer_mod.wandb, "Table", lambda dataframe: dataframe),
            patch.object(trainer_mod.wandb, "log", fake_wandb_log),
        ]
        return logged, patches

    def test_image_gen_routes_rewards_and_logs_both_views(self):
        calls = {}

        def boxed_and_answer_tags_format_reward(prompts, completions, answer, **kwargs):
            calls["format"] = {
                "prompts": prompts,
                "completions": completions,
                "answer": answer,
            }
            return [0.1] * len(completions)

        def correct_grounding_reward_func(prompts, completions, answer, **kwargs):
            calls["grounding"] = {
                "prompts": prompts,
                "completions": completions,
                "answer": answer,
            }
            return [0.2] * len(completions)

        def correctness_reward_func(prompts, completions, answer, **kwargs):
            calls["correctness"] = {
                "prompts": prompts,
                "completions": completions,
                "answer": answer,
            }
            return [0.3] * len(completions)

        trainer = _make_dummy_trainer(
            [
                boxed_and_answer_tags_format_reward,
                correct_grounding_reward_func,
                correctness_reward_func,
            ]
        )
        logged, patches = self._patch_runtime()

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            trainer_mod.DiffuGRPOTrainer._generate_and_score_completions(
                trainer, _make_inputs("image_gen")
            )

        self.assertEqual(calls["format"]["prompts"], ["text prompt 1", "text prompt 2"])
        self.assertEqual(
            calls["format"]["completions"],
            [
                "<reasoning>\nsteps\n</reasoning>\n<answer>\n42\n</answer>",
                "<reasoning>\nsteps\n</reasoning>\n<answer>\n42\n</answer>",
            ],
        )
        self.assertEqual(calls["correctness"]["prompts"], ["text prompt 1", "text prompt 2"])
        self.assertEqual(calls["grounding"]["prompts"], ["ground prompt 1", "ground prompt 2"])
        self.assertEqual(
            calls["grounding"]["completions"],
            ["<LOC_10><LOC_20><LOC_30><LOC_40>"] * 2,
        )
        self.assertEqual(calls["grounding"]["answer"], [[10, 20, 30, 40], [10, 20, 30, 40]])
        reencode_call = trainer.processing_class.calls[-1]
        self.assertEqual(
            reencode_call["texts"],
            [
                f"{trainer_mod.DEFAULT_IMAGE_TOKEN} {trainer_mod.DEFAULT_IMAGE_TOKEN}\ntext prompt 1",
                f"{trainer_mod.DEFAULT_IMAGE_TOKEN} {trainer_mod.DEFAULT_IMAGE_TOKEN}\ntext prompt 2",
            ],
        )

        completions_df = logged["completions"]
        self.assertIn("[grounding_input]", completions_df["prompt"].iloc[0])
        self.assertIn("[text_input]", completions_df["prompt"].iloc[0])
        self.assertIn("[grounding_output]", completions_df["completion"].iloc[0])
        self.assertIn("[text_output]", completions_df["completion"].iloc[0])
        self.assertIn("reward", completions_df.columns)
        self.assertIn("boxed_and_answer_tags_format_reward", completions_df.columns)
        self.assertIn("correct_grounding_reward_func", completions_df.columns)
        self.assertIn("correctness_reward_func", completions_df.columns)
        self.assertIn("original_image", completions_df.columns)
        self.assertIn("edited_image", completions_df.columns)
        self.assertIsNotNone(completions_df["original_image"].iloc[0])
        self.assertIsNotNone(completions_df["edited_image"].iloc[0])

    def test_image_gen_reencode_tops_up_existing_image_tokens(self):
        trainer = _make_dummy_trainer([lambda prompts, completions, answer, **kwargs: [0.0] * len(completions)])
        _, patches = self._patch_runtime()
        prompt_inputs = _make_inputs(
            "image_gen",
            prompts=[
                f"{trainer_mod.DEFAULT_IMAGE_TOKEN}\ntext prompt 1",
                f"{trainer_mod.DEFAULT_IMAGE_TOKEN}\ntext prompt 2",
            ],
        )

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            trainer_mod.DiffuGRPOTrainer._generate_and_score_completions(trainer, prompt_inputs)

        reencode_call = trainer.processing_class.calls[-1]
        self.assertEqual(
            reencode_call["texts"],
            [
                f"{trainer_mod.DEFAULT_IMAGE_TOKEN} {trainer_mod.DEFAULT_IMAGE_TOKEN}\ntext prompt 1",
                f"{trainer_mod.DEFAULT_IMAGE_TOKEN} {trainer_mod.DEFAULT_IMAGE_TOKEN}\ntext prompt 2",
            ],
        )

    def test_grounding_mode_keeps_grounding_only_view(self):
        calls = {}

        def correct_grounding_reward_func(prompts, completions, answer, **kwargs):
            calls["grounding"] = {
                "prompts": prompts,
                "completions": completions,
            }
            return [0.5] * len(completions)

        trainer = _make_dummy_trainer([correct_grounding_reward_func])
        logged, patches = self._patch_runtime()

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            trainer_mod.DiffuGRPOTrainer._generate_and_score_completions(
                trainer, _make_inputs("grounding")
            )

        self.assertEqual(calls["grounding"]["prompts"], ["ground prompt 1", "ground prompt 2"])
        self.assertEqual(
            calls["grounding"]["completions"],
            ["<LOC_10><LOC_20><LOC_30><LOC_40>"] * 2,
        )
        completions_df = logged["completions"]
        self.assertEqual(completions_df["prompt"].tolist(), ["ground prompt 1", "ground prompt 2"])
        self.assertEqual(completions_df["completion"].tolist(), ["<LOC_10><LOC_20><LOC_30><LOC_40>"] * 2)
        self.assertIn("correct_grounding_reward_func", completions_df.columns)

    def test_text_gen_mode_keeps_text_only_view(self):
        calls = {}

        def correctness_reward_func(prompts, completions, answer, **kwargs):
            calls["correctness"] = {
                "prompts": prompts,
                "completions": completions,
            }
            return [0.7] * len(completions)

        trainer = _make_dummy_trainer([correctness_reward_func])
        logged, patches = self._patch_runtime()

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            trainer_mod.DiffuGRPOTrainer._generate_and_score_completions(
                trainer, _make_inputs("text_gen")
            )

        self.assertEqual(calls["correctness"]["prompts"], ["text prompt 1", "text prompt 2"])
        self.assertEqual(
            calls["correctness"]["completions"],
            [
                "<reasoning>\nsteps\n</reasoning>\n<answer>\n42\n</answer>",
                "<reasoning>\nsteps\n</reasoning>\n<answer>\n42\n</answer>",
            ],
        )
        completions_df = logged["completions"]
        self.assertEqual(completions_df["prompt"].tolist(), ["text prompt 1", "text prompt 2"])
        self.assertEqual(
            completions_df["completion"].tolist(),
            [
                "<reasoning>\nsteps\n</reasoning>\n<answer>\n42\n</answer>",
                "<reasoning>\nsteps\n</reasoning>\n<answer>\n42\n</answer>",
            ],
        )
        self.assertIn("correctness_reward_func", completions_df.columns)


if __name__ == "__main__":
    unittest.main()
