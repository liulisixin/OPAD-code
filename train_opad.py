#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Custom Diffusion authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
import copy
from contextlib import nullcontext

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from util_ip_adapter import IPAdapter
# from util_ip_adapter_faceid import IPAdapterFaceIDPlus
import lpips
from MS_SWD import MS_SWD
from util_disc import DinoDiscriminator, IPAdapterDisc


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0")

logger = get_logger(__name__)


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def save_model_card(repo_id: str, images=None, base_model=str, prompt=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- custom-diffusion
inference: true
---
    """
    model_card = f"""
# Custom Diffusion - {repo_id}

These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
{img_str}

\nFor more details on the training, please follow [this link](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion).
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    instance_prompts = [example["instance_prompt"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {"instance_prompt_ids": input_ids, "instance_images": pixel_values, "mask": mask.unsqueeze(1),
             "instance_prompt": instance_prompts}
    return batch


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


class PromptDatasetV2(Dataset):
    def __init__(self, prompt_file_txt, tokenizer):
        with open(prompt_file_txt) as f:
            prompt_list = f.read().splitlines()
        self.prompt_list = prompt_list
        self.prompt_ids_all = tokenizer(
            self.prompt_list,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, index):
        example = {}
        example["instance_prompt"] = self.prompt_list[index]
        example["instance_prompt_ids"] = self.prompt_ids_all[index]
        return example

    # def shuffle(self, *args, **kwargs):
    #     random.shuffle(self.train_data_paths)
    #     return self

    # def select(self, selected_range):
    #     self.train_data_paths = [self.train_data_paths[idx] for idx in selected_range]
    #     return self


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = list(zip(class_images_path, class_prompt))
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt"] = instance_prompt   # add
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


class CustomDiffusionDatasetV2(CustomDiffusionDataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation

        for concept in concepts_list:
            prompt_list = self.prompt_template(concept["instance_prompt"], filter_dulp=self.aug)
            inst_img_path = [
                (x, y) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file() for y in prompt_list
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = list(zip(class_images_path, class_prompt))
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def prompt_template(self, case_name, filter_dulp=False):
        imagenet_templates_small = [
            "a photo of a {}",
            "a rendering of a {}",
            "a cropped photo of the {}",
            "the photo of a {}",
            "a photo of a clean {}",
            "a photo of a dirty {}",
            "a dark photo of the {}",
            "a photo of my {}",
            "a photo of the cool {}",
            "a close-up photo of a {}",
            "a bright photo of the {}",
            "a cropped photo of a {}",
            "a photo of the {}",
            "a good photo of the {}",
            "a photo of one {}",
            "a close-up photo of the {}",
            "a rendition of the {}",
            "a photo of the clean {}",
            "a rendition of a {}",
            "a photo of a nice {}",
            "a good photo of a {}",
            "a photo of the nice {}",
            "a photo of the small {}",
            "a photo of the weird {}",
            "a photo of the large {}",
            "a photo of a cool {}",
            "a photo of a small {}",
        ]
        if filter_dulp:
            prompt_list = [x for x in imagenet_templates_small if not 'close' in x]
        else:
            prompt_list = imagenet_templates_small
        prompt_list = [x.replace('{}', case_name) for x in prompt_list]
        return prompt_list
    
    def merge_prompt(self, sentence, insert_word):
        words = sentence.split()
        words.insert(1, insert_word)
        return ' '.join(words)

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = self.merge_prompt(instance_prompt, np.random.choice(["a far away", "very small"]))
        elif random_scale > self.size:
            instance_prompt = self.merge_prompt(instance_prompt, np.random.choice(["zoomed in", "close up"]))

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt"] = instance_prompt   # add
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def save_new_embed(text_encoder, modifier_token_id, accelerator, args, output_dir, safe_serialization=True):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        filename = f"{output_dir}/{y}.bin"

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, filename, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, filename)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Custom Diffusion training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/sd-turbo",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default="crossattn_kv",
        choices=["crossattn_kv", "crossattn"],
        help="crossattn to enable fine-tuning of all params in the cross attention",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` (default) and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", action="store_true", default=True, help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        default=True,
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    # new paras for teacher
    parser.add_argument(
        "--teacher_pretrained_model_name_or_path",
        type=str,
        default="sd2-community/stable-diffusion-2-1",
        required=False,
        help="Path to pretrained model or model identifier from teacher lora.",
    )
    parser.add_argument(
        "--train_custom_teacher",
        action="store_true",
        default=True,
        help="whether the teacher is trained or loaded.",
    )
    parser.add_argument(
        "--learning_rate_teacher",
        type=float,
        default=1e-5,
        help="learning_rate_teacher.",
    )
    parser.add_argument(
        "--teacher_path_custom_model",
        type=str,
        default=None,
        # required=True,
        help="Custom layers for teacher model.",
    )
    parser.add_argument(
        "--train_student",
        type=str,
        default="unet",
        choices=["unet_textencoder", "unet"],
        help="crossattn to enable fine-tuning of all params in the cross attention",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help=("The alpha constant of the LoRA update matrices."),
    )
    parser.add_argument(
        "--learning_rate_lora",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use for lora teacher.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        # required=True,
        help="prompt_file for vsd loss.",
    )
    parser.add_argument(
        "--teacher_guidance_scale",
        type=float,
        default=7.5,
        help="guidance_scale for teacher lora.",
    )
    parser.add_argument(
        "--teacher_infer_steps",
        type=int,
        default=25,
        help=("Number of steps when the teachers do inference."),
    )
    parser.add_argument(
        "--student_infer_steps",
        type=int,
        default=1,
        help=("Number of steps when the student do inference."),
    )
    parser.add_argument(
        "--use_loss",
        type=str,
        default="ipadapter_latentmse_gan_MSSWD",
        required=False,
        help="use loss for training.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="CustomDiffusionDatasetV2",
        choices=["PromptDatasetV2", "CustomDiffusionDataset", "CustomDiffusionDatasetV2"],
        help="type of dataset",
    )
    parser.add_argument(
        "--target_samples",
        type=str,
        default="teacher",
        choices=["real", "teacher"],
        help="target_samples for comparing with the output of the student",
    )
    parser.add_argument(
        "--apply_timesteps_weight",
        action="store_true",
        default=True,
        help="apply_timesteps_weight.",
    )
    parser.add_argument(
        "--use_disc",
        action="store_true",
        default=True,
        help="use discriminator.",
    )
    parser.add_argument(
        "--disc_model",
        type=str,
        default="vit_small_patch16_224.dino+vit_large_patch14_dinov2.lvd142m+IPAdapterDisc",
        # choices=["vit_small_patch16_224.dino", "vit_large_patch14_dinov2.lvd142m", "IPAdapterDisc"],
        help="disc_model",
    )
    parser.add_argument(
        "--disc_head_layer",
        type=int,
        default=2,
        help="number of layers in the classification head of the disc.",
    )
    parser.add_argument(
        "--disc_resize_method",
        type=str,
        default="resize",
        choices=["resize", "padding"],
        help="disc_resize_method",
    )
    parser.add_argument(
        "--learning_rate_disc",
        type=float,
        default=1e-4,
        help="learning_rate_disc.",
    )
    parser.add_argument(
        "--disc_samples",
        type=str,
        default="real",
        choices=["real", "teacher"],
        help="samples for the real part of the discriminator",
    )
    parser.add_argument(
        "--use_sampling_timesteps",
        action="store_true",
        help="use_sampling_timesteps.",
    )
    parser.add_argument(
        "--train_all_unet",
        action="store_true",
        help="train_all_unet.",
    )
    parser.add_argument(
        "--feed_teacher_student",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="feed_teacher_student.",
    )
    parser.add_argument(
        "--feed_student_teacher_noise",
        action="store_true",
        help="The student receives the input as the output of the teacher + 999 noise.",
    )
    parser.add_argument("--loss_weight_ipadapter", type=float, default=1.0, help="loss_weight")
    parser.add_argument("--loss_weight_mse", type=float, default=1.0, help="loss_weight")
    parser.add_argument("--loss_weight_MSSWD", type=float, default=0.1, help="loss_weight")
    parser.add_argument("--loss_weight_gan_dinov1", type=float, default=1.0, help="loss_weight")
    parser.add_argument("--loss_weight_gan_dinov2", type=float, default=1.0, help="loss_weight")
    parser.add_argument("--loss_weight_gan_ipadapter", type=float, default=1.0, help="loss_weight")
    parser.add_argument(
        "--ip_adapter_image_encoder_path",
        type=str,
        default=None,
        help="Path to the IP-Adapter CLIP image encoder.",
    )
    parser.add_argument(
        "--ip_adapter_ckpt",
        type=str,
        default=None,
        help="Path to ip-adapter_sd15.bin.",
    )
    parser.add_argument(
        "--ip_adapter_faceid_ckpt",
        type=str,
        default=os.environ.get("IP_ADAPTER_FACEID_CKPT"),
        help="Path to the IP-Adapter FaceID checkpoint, only used by the faceid branch.",
    )

    parser.add_argument(
        "--feed_teacher_student_infer_steps",
        type=int,
        default=0,
        help=(
            "Teacher inference steps for refining the student x0 target. "
            "If <= 0, use the faster random-timestep teacher x0 prediction branch."
        ),
    )
    parser.add_argument(
        "--feed_teacher_student_denoise_steps",
        type=int,
        default=1,
        help=(
            "Maximum teacher denoising steps when --feed_teacher_student_infer_steps > 0. "
            "Ignored when --feed_teacher_student_infer_steps <= 0."
        ),
    )
    parser.add_argument(
        "--not_save_teacher",
        action="store_true",
        help="save_teacher.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def process_teacher_pipeline(pipeline):
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet

    unet.requires_grad_(False)

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
    ########################################################
    ########################################################
    attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor
    )
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            attention_class = CustomDiffusionXFormersAttnProcessor
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    train_kv = True
    train_q_out = False if args.freeze_model == "crossattn_kv" else True
    custom_diffusion_attn_procs = {}

    st = unet.state_dict()
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
            "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
        }
        if train_q_out:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=train_kv,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)

    return pipeline, custom_diffusion_layers



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # Log configuration on every process.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("custom-diffusion", config=vars(args),
                                  init_kwargs={"wandb": {"name": args.output_dir.replace("output_", "", 1)}})

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept["class_data_dir"])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if args.real_prior:
                assert (
                    class_images_dir / "images"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                    len(list((class_images_dir / "images").iterdir())) == args.num_class_images
                ), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                    class_images_dir / "caption.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                    class_images_dir / "images.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
                concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
                args.concepts_list[i] = concept
                accelerator.wait_for_everyone()
            else:
                cur_class_images = len(list(class_images_dir.iterdir()))

                if cur_class_images < args.num_class_images:
                    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                    if args.prior_generation_precision == "fp32":
                        torch_dtype = torch.float32
                    elif args.prior_generation_precision == "fp16":
                        torch_dtype = torch.float16
                    elif args.prior_generation_precision == "bf16":
                        torch_dtype = torch.bfloat16
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                        variant=args.variant,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = args.num_class_images - cur_class_images
                    logger.info(f"Number of class images to sample: {num_new_images}.")

                    sample_dataset = PromptDataset(concept["class_prompt"], num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                    sample_dataloader = accelerator.prepare(sample_dataloader)
                    pipeline.to(accelerator.device)

                    for example in tqdm(
                        sample_dataloader,
                        desc="Generating class images",
                        disable=not accelerator.is_local_main_process,
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = (
                                class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            )
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    """
    Define the teacher
    """
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        args.teacher_pretrained_model_name_or_path
    ).to("cuda")
    if args.train_custom_teacher:
        teacher_pipeline, teacher_custom_diffusion_layers = process_teacher_pipeline(teacher_pipeline)
    else:
        teacher_pipeline.unet.load_attn_procs(args.teacher_path_custom_model, weight_name="pytorch_custom_diffusion_weights.bin")
        teacher_pipeline.load_textual_inversion(args.teacher_path_custom_model, weight_name="<new1>.bin")
    """
    End: Define the teacher
    """
    
    # Load the components
    tokenizer = teacher_pipeline.tokenizer
    vae = teacher_pipeline.vae

    # text encoder
    teacher_text_encoder = teacher_pipeline.text_encoder
    if args.train_student == "unet":
        student_text_encoder = teacher_text_encoder
    elif args.train_student == "unet_textencoder":
        student_text_encoder = copy.deepcopy(teacher_text_encoder)
    else:
        raise NotImplementedError
    
    # unet for teacher
    teacher_unet_fix = teacher_pipeline.unet
    if not args.train_custom_teacher:
        teacher_unet_fix.requires_grad_(False)
        teacher_text_encoder.requires_grad_(False)

    has_tune_teacher = 'vsd' in args.use_loss
    if has_tune_teacher:
        teacher_unet_tune = copy.deepcopy(teacher_unet_fix)
        teacher_custom_diffusion_params = []
        for param_name, param in teacher_unet_tune.named_parameters():
            if 'custom_diffusion' in param_name:
                param.requires_grad = True
                teacher_custom_diffusion_params.append(param)

    # unet for student
    student_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # for discriminator
    def define_disc(disc_model_name):
        if 'dino' in disc_model_name:
            disc = DinoDiscriminator(model_name=disc_model_name, num_classes=2, freeze_dino=True, resize_method=args.disc_resize_method)
        elif 'IPAdapterDisc' in disc_model_name:
            image_encoder_path = args.ip_adapter_image_encoder_path
            ip_ckpt = args.ip_adapter_ckpt
            if image_encoder_path is None or ip_ckpt is None:
                raise ValueError("IPAdapterDisc requires --ip_adapter_image_encoder_path and --ip_adapter_ckpt.")
            disc = IPAdapterDisc(image_encoder_path, ip_ckpt, accelerator.device, num_classes=2, disc_head_layer=args.disc_head_layer)
        else:
            raise NotImplementedError
        return disc

    if args.use_disc:
        disc_list = []
        weight_dict_disc = {
            "vit_small_patch16_224.dino": args.loss_weight_gan_dinov1,
            "vit_large_patch14_dinov2.lvd142m": args.loss_weight_gan_dinov2,
            "IPAdapterDisc": args.loss_weight_gan_ipadapter,
        }
        loss_weight_gan_list = []
        for disc_model_name in args.disc_model.split('+'):
            disc_list.append(define_disc(disc_model_name))
            loss_weight_gan_list.append(weight_dict_disc[disc_model_name])

        adversarial_loss = torch.nn.CrossEntropyLoss().to(accelerator.device)
        if args.apply_timesteps_weight:
            adversarial_loss_noreduc = torch.nn.CrossEntropyLoss(reduction='none').to(accelerator.device)

    # Load scheduler
    # sd 2.1 uses ddim
    teacher_noise_scheduler = DDIMScheduler.from_pretrained(args.teacher_pretrained_model_name_or_path, subfolder="scheduler")
    # sd-turbo uses EulerDiscreteScheduler
    student_noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # student_noise_scheduler = teacher_noise_scheduler

    # find modifier_token_id
    modifier_token_id = []
    if args.modifier_token is not None:
        if not isinstance(args.modifier_token, list):
            args.modifier_token = args.modifier_token.split("+")
        for modifier_token in args.modifier_token:
            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([modifier_token], add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The token must be a single token.")

            modifier_token_id.append(token_ids[0])

        if 'textencoder' in args.train_student:
            # Freeze all parameters except for the token embeddings in text encoder
            params_to_freeze = itertools.chain(
                student_text_encoder.text_model.encoder.parameters(),
                student_text_encoder.text_model.final_layer_norm.parameters(),
                student_text_encoder.text_model.embeddings.position_embedding.parameters(),
            )
            freeze_params(params_to_freeze)
        # else: student_text_encoder is teacher_text_encoder
    ########################################################

    # set requires_grad_
    if args.modifier_token is None:
        student_text_encoder.requires_grad_(False)

    vae.requires_grad_(False)
    student_unet.requires_grad_(False)
    ########################################################

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16" and args.modifier_token is not None:
        student_text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    teacher_text_encoder.to(accelerator.device, dtype=weight_dtype)
    teacher_unet_fix.to(accelerator.device, dtype=weight_dtype)
    if has_tune_teacher:
        teacher_unet_tune.to(accelerator.device, dtype=weight_dtype)
    student_unet.to(accelerator.device, dtype=weight_dtype)
    if args.use_disc:
        for disc in disc_list:
            disc.to(accelerator.device, dtype=weight_dtype)
    
    ########################################################

    attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor
    )
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            attention_class = CustomDiffusionXFormersAttnProcessor
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    train_kv = True
    train_q_out = False if args.freeze_model == "crossattn_kv" else True

    def create_attn_procs(unet, train_kv, train_q_out):
        custom_diffusion_attn_procs = {}
        st = unet.state_dict()
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
                "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
            }
            if train_q_out:
                weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
                weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
                weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
            if cross_attention_dim is not None:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=train_kv,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
                custom_diffusion_attn_procs[name].load_state_dict(weights)
            else:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=False,
                    train_q_out=False,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
        del st
        unet.set_attn_processor(custom_diffusion_attn_procs)
        custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)

        return unet, custom_diffusion_layers

    student_unet, student_custom_diffusion_layers = create_attn_procs(student_unet, train_kv, train_q_out)

    accelerator.register_for_checkpointing(student_custom_diffusion_layers)

    if args.gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()
        teacher_unet_fix.enable_gradient_checkpointing()
        if has_tune_teacher:
            teacher_unet_tune.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            student_text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_lora = (
            args.learning_rate_lora
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0
            raise NotImplementedError

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer_lora_class = optimizer_class

    # Optimizer creation
    if args.train_all_unet:
        student_unet.requires_grad_(True)
        optimizer = optimizer_class(
            student_unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_class(
            itertools.chain(student_text_encoder.get_input_embeddings().parameters(), student_custom_diffusion_layers.parameters())
            if (args.modifier_token is not None and 'textencoder' in args.train_student)
            else student_custom_diffusion_layers.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    if args.train_custom_teacher:
        optimizer_teacher = optimizer_class(
            itertools.chain(teacher_text_encoder.get_input_embeddings().parameters(), teacher_custom_diffusion_layers.parameters())
            if args.modifier_token is not None
            else teacher_custom_diffusion_layers.parameters(),
            lr=args.learning_rate_teacher,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    if has_tune_teacher:
        optimizer_lora = optimizer_lora_class(
            teacher_custom_diffusion_params,
            lr=args.learning_rate_lora,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    if args.use_disc:
        optimizer_class_disc = optimizer_class
        optimizer_disc_list = []
        for disc in disc_list:
            optimizer_disc = optimizer_class_disc(
                disc.classification_head.parameters(),
                lr=args.learning_rate_disc,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
            optimizer_disc_list.append(optimizer_disc)

    # Dataset and DataLoaders creation:
    if args.dataset_type == "PromptDatasetV2":
        train_dataset = PromptDatasetV2(
            args.prompt_file, tokenizer
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            # collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
            num_workers=args.dataloader_num_workers,
        )
    elif args.dataset_type == "CustomDiffusionDataset":
        train_dataset = CustomDiffusionDataset(
            concepts_list=args.concepts_list,
            tokenizer=tokenizer,
            with_prior_preservation=args.with_prior_preservation,
            size=args.resolution,
            mask_size=vae.encode(
                torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
            )
            .latent_dist.sample()
            .size()[-1],
            center_crop=args.center_crop,
            num_class_images=args.num_class_images,
            hflip=args.hflip,
            aug=not args.noaug,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
            num_workers=args.dataloader_num_workers,
        )
    elif args.dataset_type == "CustomDiffusionDatasetV2":
        train_dataset = CustomDiffusionDatasetV2(
            concepts_list=args.concepts_list,
            tokenizer=tokenizer,
            with_prior_preservation=args.with_prior_preservation,
            size=args.resolution,
            mask_size=vae.encode(
                torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
            )
            .latent_dist.sample()
            .size()[-1],
            center_crop=args.center_crop,
            num_class_images=args.num_class_images,
            hflip=args.hflip,
            aug=not args.noaug,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
            num_workers=args.dataloader_num_workers,
        )
    else:
        raise NotImplementedError

    

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    if args.train_custom_teacher:
        lr_scheduler_teacher = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_teacher,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
        optimizer_teacher, lr_scheduler_teacher = accelerator.prepare(optimizer_teacher, lr_scheduler_teacher)
    if has_tune_teacher:
        lr_scheduler_lora = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_lora,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
        optimizer_lora, lr_scheduler_lora = accelerator.prepare(optimizer_lora, lr_scheduler_lora)
    if args.use_disc:
        lr_scheduler_disc_list = []
        for optimizer_disc in optimizer_disc_list:
            lr_scheduler_disc = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer_disc,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
            )
            lr_scheduler_disc_list.append(lr_scheduler_disc)
        optimizer_disc_list = [accelerator.prepare(x) for x in optimizer_disc_list]
        lr_scheduler_disc_list = [accelerator.prepare(x) for x in lr_scheduler_disc_list]

    # Prepare everything with our `accelerator`.
    if args.modifier_token is not None and 'textencoder' in args.train_student:
        student_custom_diffusion_layers, student_text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            student_custom_diffusion_layers, student_text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        student_custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            student_custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
        )
    if args.train_custom_teacher:
        teacher_custom_diffusion_layers, teacher_text_encoder = accelerator.prepare(
            teacher_custom_diffusion_layers, teacher_text_encoder
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # Dec 30, 2024: not checked
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Get alphas cummulative product
    teacher_alphas_cumprod = teacher_noise_scheduler.alphas_cumprod
    teacher_alphas_cumprod = teacher_alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    # Get null-text embedding
    null_dict = encode_prompt([""], teacher_text_encoder, tokenizer)

    # define the ip-adapter as the id loss
    # original ip-adapter
    if 'ipadapter' in args.use_loss:
        args.ipadapter = "ip-adapter_sd15.bin"
        if args.ipadapter == "ip-adapter_sd15.bin":
            image_encoder_path = args.ip_adapter_image_encoder_path
            ip_ckpt = args.ip_adapter_ckpt
            if image_encoder_path is None or ip_ckpt is None:
                raise ValueError("IP-Adapter loss requires --ip_adapter_image_encoder_path and --ip_adapter_ckpt.")
            ip_model = IPAdapter(None, image_encoder_path, ip_ckpt, accelerator.device)
        elif args.ipadapter == "ip-adapter-faceid-plus_sd15.bin":
            # face id
            pass
        else:
            raise NotImplementedError
        ip_model = accelerator.prepare(
                ip_model
            )
    
    if 'lpips' in args.use_loss:
        loss_fn_lpips = lpips.LPIPS(net='alex')
        loss_fn_lpips = accelerator.prepare(loss_fn_lpips)
    if 'MSSWD' in args.use_loss:
        loss_fn_MSSWD = MS_SWD(num_scale=5, num_proj=128)
        loss_fn_MSSWD = accelerator.prepare(loss_fn_MSSWD)
    
    if args.use_sampling_timesteps:
        raise ValueError("--use_sampling_timesteps is an experimental branch and is not included in this release.")

    for epoch in range(first_epoch, args.num_train_epochs):
        if 'vsd' in args.use_loss:
            train_loss_vsd = 0.0
        if has_tune_teacher:
            train_loss_lora = 0.0
        if 'mse' in args.use_loss:
            train_loss_mse = 0.0
        if 'ipadapter' in args.use_loss:
            train_loss_ipadapter = 0.0
        if 'lpips' in args.use_loss:
            train_loss_lpips = 0.0
        if 'MSSWD' in args.use_loss:
            train_loss_MSSWD = 0.0
        if 'gan' in args.use_loss:
            train_loss_gan_list = [0.0] * len(disc_list)
        if args.use_disc:
            train_loss_disc_list = [0.0] * len(disc_list)
        if args.train_custom_teacher:
            train_loss_teacher = 0.0
        
        if args.modifier_token is not None and 'textencoder' in args.train_student:
            student_text_encoder.train()
        
        def loss_log(loss_x):
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss_x = accelerator.gather(loss_x.repeat(args.train_batch_size)).mean()
            return avg_loss_x.item() / args.gradient_accumulation_steps

        for step, batch in enumerate(train_dataloader):
            ############## Validation ##############
            if accelerator.is_main_process:
                images = []

                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    if has_tune_teacher:
                        teacher_unet_tune.eval()
                    student_unet.eval()
                    teacher_unet_fix.eval()
                    teacher_text_encoder.eval()

                    use_prompt = [batch['instance_prompt'][0], args.validation_prompt]
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {use_prompt[0]}."
                        f" {use_prompt[1]}."
                    )

                    if "textencoder" in args.train_student:
                        chosen_text_encoder = accelerator.unwrap_model(student_text_encoder)
                    else:
                        chosen_text_encoder = teacher_text_encoder

                    student_pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(student_unet),
                        text_encoder=chosen_text_encoder,
                        tokenizer=tokenizer,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    student_pipeline = student_pipeline.to(accelerator.device)
                    # pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                    images = [
                        (i, p, student_pipeline(p, num_inference_steps=args.student_infer_steps, generator=generator, guidance_scale=0.0).images[0])
                        for i in range(args.num_validation_images) for p in use_prompt
                    ]
                    # here guidance_scale=0.0 does not make sense, regarded as guidance_scale=1.0

                    # for teacher
                    def generate_images_with_adapter(adapter_enabled):
                        if adapter_enabled:
                            choice = accelerator.unwrap_model(teacher_unet_tune)
                        else:
                            choice = accelerator.unwrap_model(teacher_unet_fix)
                        teacher_pipeline = StableDiffusionPipeline.from_pretrained(
                            args.teacher_pretrained_model_name_or_path,
                            unet=choice,
                            text_encoder=teacher_text_encoder,
                            tokenizer=tokenizer,
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )
                        teacher_pipeline = teacher_pipeline.to(accelerator.device)
                        # teacher_pipeline.set_progress_bar_config(disable=True)
                        teacher_images = [
                            (i, p, teacher_pipeline(p, num_inference_steps=args.teacher_infer_steps, generator=generator, guidance_scale=args.teacher_guidance_scale).images[0])
                            for i in range(args.num_validation_images) for p in use_prompt
                        ]
                        return teacher_images
                    if has_tune_teacher:
                        teacher_images_enabled = generate_images_with_adapter(True)
                    teacher_images_disabled = generate_images_with_adapter(False)

                    for tracker in accelerator.trackers:
                        # if tracker.name == "tensorboard":
                        #     np_images = np.stack([np.asarray(img) for img in images])
                        #     tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            log_dict = {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {p}")
                                        for i, p, image in images
                                    ],
                                    "validation_teacher_images_disabled": [
                                        wandb.Image(image, caption=f"{i}: {p}")
                                        for i, p, image in teacher_images_disabled
                                    ]
                                }
                            if has_tune_teacher:
                                log_dict["validation_teacher_images_enabled"] = [
                                        wandb.Image(image, caption=f"{i}: {p}")
                                        for i, p, image in teacher_images_enabled
                                    ]
                            tracker.log(log_dict, step=global_step)

                    del student_pipeline
                    torch.cuda.empty_cache()
            ############## End Validation ##############

            if has_tune_teacher:
                teacher_unet_tune.train()
            student_unet.train()
            teacher_unet_fix.train()
            teacher_text_encoder.train()
            ############## Loss ##############
            if "textencoder" in args.train_student:
                text_encoder_accumulate = accelerator.accumulate(student_text_encoder)
            else:
                text_encoder_accumulate = nullcontext()
            if has_tune_teacher:
                teacher_unet_tune_accumulate = accelerator.accumulate(teacher_unet_tune)
            else:
                teacher_unet_tune_accumulate = nullcontext()
            with accelerator.accumulate(teacher_text_encoder), accelerator.accumulate(student_unet), teacher_unet_tune_accumulate, text_encoder_accumulate:

                bsz = batch["instance_prompt_ids"].shape[0]

                # for discriminator
                if args.use_disc:
                    target_real = torch.ones((bsz,), dtype=torch.long, device=accelerator.device)
                    target_fake = torch.zeros((bsz,), dtype=torch.long, device=accelerator.device)

                # Sample noise that we'll input into the model
                input_shape = (bsz, 4, args.resolution // 8, args.resolution // 8)
                input_noise = torch.randn(*input_shape, dtype=weight_dtype, device=accelerator.device)

                # Get the text embeddings
                teacher_prompt_embeds = teacher_text_encoder(batch["instance_prompt_ids"])[0]
                if "textencoder" in args.train_student:
                    student_prompt_embeds = student_text_encoder(batch["instance_prompt_ids"])[0]
                else:
                    student_prompt_embeds = teacher_prompt_embeds
                prompt_null_embeds = (
                    null_dict["prompt_embeds"].repeat(bsz, 1, 1).to(accelerator.device, dtype=weight_dtype)
                )

                ####### move train teacher at here 
                def predict_random_step(unet, noise_scheduler, input_latents, timesteps, prompt_embeds, guidance_scale=0.0,
                                negative_prompt_embeds=None, return_x0_latent=False, return_x0_image=True, return_noise_pred=False):
                    latents = input_latents
                    if guidance_scale > 1.0:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                    num_inference_steps = noise_scheduler.config.num_train_timesteps
                    _, _ = retrieve_timesteps(noise_scheduler, num_inference_steps, device=latents.device)

                    if guidance_scale > 1.0:
                        latent_model_input = torch.cat([latents] * 2)
                        timesteps_input = torch.cat([timesteps] * 2)
                    else:
                        latent_model_input = latents
                        timesteps_input = timesteps
                    # latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    noise_pred = unet(
                        latent_model_input,
                        timesteps_input,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    result = {}
                    if return_noise_pred:
                        result['noise_pred'] = noise_pred_text
                    if return_noise_pred and not return_x0_latent and not return_x0_image:
                        return result

                    with torch.no_grad():
                        # compute the previous noisy sample x_t -> x_t-1
                        # return prev_sample (0), pred_original_sample (1),
                        latents_original_sample_list = []
                        for i, t in enumerate(timesteps):
                            latents_original_sample_list.append(noise_scheduler.step(noise_pred[i].unsqueeze(0), t, latents[i].unsqueeze(0)).pred_original_sample)
                        latents_original_sample = torch.cat(latents_original_sample_list, dim=0)
                        
                        image_output = vae.decode(latents_original_sample / vae.config.scaling_factor, return_dict=False)[0]
                    
                    if return_x0_latent:
                        result['x0_latent'] = latents_original_sample
                    if return_x0_image:
                        result['x0_image'] = image_output
                    
                    return result

                if args.train_custom_teacher:
                    optimizer_teacher.zero_grad(set_to_none=args.set_grads_to_none)

                    # Convert images to latent space
                    latents = vae.encode(batch["instance_images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents_gt = latents
                    # Sample noise that we'll add to the latents
                    noise = input_noise
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, teacher_noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = teacher_noise_scheduler.add_noise(latents, noise, timesteps)

                    flag_return_x0 = not (args.use_sampling_timesteps or args.feed_teacher_student)
                    if flag_return_x0:
                        raise NotImplementedError
                    # predict_random_value = predict_random_step(teacher_unet_fix, teacher_noise_scheduler, noisy_latents, timesteps, teacher_prompt_embeds, 
                    #                                         guidance_scale=args.teacher_guidance_scale, negative_prompt_embeds=prompt_null_embeds,
                    #                                         return_x0_latent=flag_return_x0, return_x0_image=flag_return_x0, return_noise_pred=True)
                    # model_pred = predict_random_value['noise_pred']   # this is the prediction of noise from unet
                    # if flag_return_x0:
                    #     teacher_samples = predict_random_value['x0_image'].detach()
                    #     teacher_latent_pred = predict_random_value['x0_latent'].detach()
                    
                    # Predict the noise residual
                    model_pred = teacher_unet_fix(noisy_latents, timesteps, teacher_prompt_embeds).sample

                    # train teacher
                    # Get the target for loss depending on the prediction type
                    if teacher_noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif teacher_noise_scheduler.config.prediction_type == "v_prediction":
                        target = teacher_noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {teacher_noise_scheduler.config.prediction_type}")
                    mask = batch["mask"]
                    loss_teacher = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss_teacher = ((loss_teacher * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    accelerator.backward(loss_teacher)
                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if args.modifier_token is not None:
                        if accelerator.num_processes > 1:
                            grads_text_encoder = teacher_text_encoder.module.get_input_embeddings().weight.grad
                        else:
                            grads_text_encoder = teacher_text_encoder.get_input_embeddings().weight.grad
                        # Get the index for tokens that we want to zero the grads for
                        index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                        for i in range(len(modifier_token_id[1:])):
                            index_grads_to_zero = index_grads_to_zero & (
                                torch.arange(len(tokenizer)) != modifier_token_id[i]
                            )
                        grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                            index_grads_to_zero, :
                        ].fill_(0)

                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(teacher_text_encoder.parameters(), teacher_custom_diffusion_layers.parameters())
                            if args.modifier_token is not None
                            else teacher_custom_diffusion_layers.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_teacher.step()
                    lr_scheduler_teacher.step()
                    # optimizer_teacher.zero_grad(set_to_none=args.set_grads_to_none)

                    train_loss_teacher += loss_log(loss_teacher)
                ####### End: train teacher
                
                # borrow from inference of pipeline
                def predict_original(unet, noise_scheduler, input_noise, prompt_embeds, device, num_inference_steps, guidance_scale=0.0,
                                     negative_prompt_embeds=None, return_latents=True, return_images=True, use_init_noise_sigma=True):
                    latents = input_noise
                    if use_init_noise_sigma:
                        latents = latents * noise_scheduler.init_noise_sigma
                    if guidance_scale > 1.0:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                    timesteps, num_inference_steps = retrieve_timesteps(noise_scheduler, num_inference_steps, device)
                    # t = torch.tensor(999., dtype=torch.float32, device=accelerator.device)
                    for i, t in tqdm(enumerate(timesteps)):
                        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                        # predict the noise residual
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )[0]
                        # perform guidance
                        if guidance_scale > 1.0:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        # compute the previous noisy sample x_t -> x_t-1
                        latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if return_images:
                        images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
                        # images = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                        if return_latents:
                            return latents, images
                        else:
                            return images
                    else:
                        if return_latents:
                            return latents
                        else:
                            raise NotImplementedError

                flag_return_latents = any([x in args.use_loss for x in ['latentmse', 'vsd']])
                if args.feed_teacher_student:
                    flag_return_latents = True
                flag_return_images = any([x in args.use_loss for x in ['mse', 'ipadapter', 'lpips', 'gan', 'MSSWD']])
                if args.train_custom_teacher:
                    student_prompt_embeds = student_prompt_embeds.detach()
                if args.feed_student_teacher_noise:
                    timesteps_999 = torch.full((bsz,), 999, dtype=torch.long, device=accelerator.device)
                    input_for_student = student_noise_scheduler.add_noise(teacher_latent_pred, input_noise, timesteps_999)
                else:
                    input_for_student = input_noise
                pred_original_values = predict_original(student_unet, student_noise_scheduler, input_for_student, student_prompt_embeds, accelerator.device,
                                                         num_inference_steps=args.student_infer_steps, 
                                                         guidance_scale=0.0, negative_prompt_embeds=None, 
                                                         return_latents=flag_return_latents, return_images=flag_return_images,
                                                         use_init_noise_sigma=not args.feed_student_teacher_noise)
                if flag_return_latents and flag_return_images:
                    pred_original_latents, pred_original_images = pred_original_values
                elif flag_return_latents:
                    pred_original_latents = pred_original_values
                elif flag_return_images:
                    pred_original_images = pred_original_values
                else:
                    raise NotImplementedError

                total_loss = 0

                if args.use_disc:
                    for i, disc in enumerate(disc_list):
                        loss_gan = adversarial_loss(disc(pred_original_images), target_real)
                        loss_gan = loss_gan * loss_weight_gan_list[i]
                        total_loss += loss_gan
                        train_loss_gan_list[i] += loss_log(loss_gan)
                
                # for x0 of the teacher
                def predict_original_teacher(unet, noise_scheduler, input_sample, input_noise, prompt_embeds, device, num_inference_steps, max_denoise_steps=1000, guidance_scale=0.0,
                                     negative_prompt_embeds=None, return_latents=True, return_images=True):
                    # latents = input_latent
                    # if use_init_noise_sigma:
                    #     latents = latents * noise_scheduler.init_noise_sigma
                    if guidance_scale > 1.0:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                    timesteps, num_inference_steps = retrieve_timesteps(noise_scheduler, num_inference_steps, device)

                    timestep_index = torch.randint(0, num_inference_steps - 1, (1,)).item()
                    initial_timestep = timesteps[timestep_index]
                    
                    latents = noise_scheduler.add_noise(input_sample, input_noise, timesteps[timestep_index])

                    for i, t in tqdm(enumerate(timesteps[timestep_index:])):
                        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                        # predict the noise residual
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )[0]
                        # perform guidance
                        if guidance_scale > 1.0:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        # compute the previous noisy sample x_t -> x_t-1
                        if i >= max_denoise_steps - 1:
                            latents = noise_scheduler.step(noise_pred, t, latents).pred_original_sample
                            break
                        else:
                            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if return_images:
                        images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
                        # images = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                        if return_latents:
                            return images, latents, initial_timestep
                        else:
                            return images, initial_timestep
                    else:
                        if return_latents:
                            return latents, initial_timestep
                        else:
                            raise NotImplementedError

                
                # image guidance
                if any([x in args.use_loss for x in ['mse', 'ipadapter', 'lpips', 'MSSWD', 'gan']]):
                    if args.dataset_type == "PromptDatasetV2":
                        with torch.no_grad():
                            teacher_samples = predict_original(teacher_unet_fix, teacher_noise_scheduler, input_noise, teacher_prompt_embeds, accelerator.device, 
                                                            num_inference_steps=args.teacher_infer_steps,
                                                            guidance_scale=args.teacher_guidance_scale, negative_prompt_embeds=prompt_null_embeds, 
                                                            return_latents=False)
                    elif args.dataset_type == "CustomDiffusionDataset" or args.target_samples == 'real':
                        teacher_samples = batch["instance_images"]
                    elif args.target_samples == 'teacher' and args.dataset_type == "CustomDiffusionDatasetV2":
                        if not args.train_custom_teacher:
                            # Convert images to latent space
                            latents = vae.encode(batch["instance_images"].to(dtype=weight_dtype)).latent_dist.sample()
                            latents = latents * vae.config.scaling_factor
                            # Sample noise that we'll add to the latents
                            noise = input_noise
                            # Sample a random timestep for each image
                            timesteps = torch.randint(0, teacher_noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                            timesteps = timesteps.long()
                            # Add noise to the latents according to the noise magnitude at each timestep
                            # (this is the forward diffusion process)
                            noisy_latents = teacher_noise_scheduler.add_noise(latents, noise, timesteps)
                            with torch.no_grad():
                                teacher_samples = predict_random_step(teacher_unet_fix, teacher_noise_scheduler, noisy_latents, timesteps, teacher_prompt_embeds, 
                                                                    guidance_scale=args.teacher_guidance_scale, negative_prompt_embeds=prompt_null_embeds,
                                                                    return_x0_latent=False, return_x0_image=True, return_noise_pred=False)['x0_image']
                        else:
                            if args.use_sampling_timesteps:
                                latents = latents_gt
                                noise = input_noise
                                #####
                                timesteps = sample_timesteps(timesteps_prob_dist, bsz)
                                timesteps = timesteps.long()
                                #####
                                noisy_latents = teacher_noise_scheduler.add_noise(latents, noise, timesteps)
                                with torch.no_grad():
                                    teacher_samples = predict_random_step(teacher_unet_fix, teacher_noise_scheduler, noisy_latents, timesteps, teacher_prompt_embeds, 
                                                                        guidance_scale=args.teacher_guidance_scale, negative_prompt_embeds=prompt_null_embeds,
                                                                        return_x0_latent=False, return_x0_image=True, return_noise_pred=False)['x0_image']
                            elif args.feed_teacher_student:
                                if args.feed_teacher_student_infer_steps <= 0:
                                    latents = pred_original_latents.detach()   # ! feed the latent output of the student
                                    noise = input_noise
                                    #####
                                    timesteps = torch.randint(0, teacher_noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                                    timesteps = timesteps.long()
                                    #####
                                    noisy_latents = teacher_noise_scheduler.add_noise(latents, noise, timesteps)
                                    with torch.no_grad():
                                        result_predict = predict_random_step(teacher_unet_fix, teacher_noise_scheduler, noisy_latents, timesteps, teacher_prompt_embeds, 
                                                                            guidance_scale=args.teacher_guidance_scale, negative_prompt_embeds=prompt_null_embeds,
                                                                            return_x0_latent=True, return_x0_image=True, return_noise_pred=False)
                                        teacher_samples = result_predict['x0_image']
                                        teacher_latent_pred = result_predict['x0_latent']
                                else:
                                    input_sample = pred_original_latents.detach()
                                    noise = input_noise
                                    with torch.no_grad():
                                        teacher_samples, teacher_latent_pred, timesteps = predict_original_teacher(teacher_unet_fix, teacher_noise_scheduler, input_sample, noise, teacher_prompt_embeds, accelerator.device, args.feed_teacher_student_infer_steps, max_denoise_steps=args.feed_teacher_student_denoise_steps, guidance_scale=args.teacher_guidance_scale, 
                                        negative_prompt_embeds=prompt_null_embeds, return_latents=True, return_images=True)
                                    timesteps = timesteps.repeat(bsz)
                            else:
                                # nothing to do, keep the output of the teacher.
                                teacher_samples = teacher_samples

                        if args.apply_timesteps_weight:
                            # weight_t = (1.0 - timesteps / 1000.0) * 0.9 + 0.1
                            weight_t = teacher_alphas_cumprod[timesteps]
                    else:
                        raise NotImplementedError

                    if 'mse' in args.use_loss:
                        if 'latentmse' in args.use_loss:
                            loss_mse = 0.5 * F.mse_loss(pred_original_latents.float(), teacher_latent_pred.float(), reduction="none").mean(dim=(1, 2, 3))
                        else:
                            loss_mse = 0.5 * F.mse_loss(pred_original_images.float(), teacher_samples.float(), reduction="none").mean(dim=(1, 2, 3))
                        if args.apply_timesteps_weight:
                            loss_mse = loss_mse * weight_t
                        
                        loss_mse = loss_mse.mean() * args.loss_weight_mse
                        
                        total_loss += loss_mse
                        train_loss_mse += loss_log(loss_mse)

                    if 'ipadapter' in args.use_loss:
                        # clip_processor resize the image to 224 and normalize it. 
                        generated_image = F.interpolate(pred_original_images, size=(224, 224), mode='bilinear', align_corners=False)
                        target_image = F.interpolate(teacher_samples, size=(224, 224), mode='bilinear', align_corners=False)
                        target_features = ip_model.get_image_id_embeds(target_image)
                        generated_features = ip_model.get_image_id_embeds(generated_image)
                        # id loss
                        loss_ipadapter = 1 - F.cosine_similarity(target_features, generated_features, dim=-1).mean(dim=-1)
                        if args.apply_timesteps_weight:
                            loss_ipadapter = loss_ipadapter * weight_t
                        
                        loss_ipadapter = loss_ipadapter.mean() * args.loss_weight_ipadapter
                        
                        total_loss += loss_ipadapter
                        train_loss_ipadapter += loss_log(loss_ipadapter)
                    
                    if 'lpips' in args.use_loss:
                        loss_lpips = loss_fn_lpips(pred_original_images.float(), teacher_samples.float()).squeeze((1,2,3))
                        if args.apply_timesteps_weight:
                            loss_lpips = loss_lpips * weight_t
                        loss_lpips = loss_lpips.mean()

                        total_loss += loss_lpips
                        train_loss_lpips += loss_log(loss_lpips)
                    
                    if 'MSSWD' in args.use_loss:
                        loss_MSSWD = loss_fn_MSSWD(pred_original_images.float(), teacher_samples.float())
                        if args.apply_timesteps_weight:
                            loss_MSSWD = loss_MSSWD * weight_t

                        loss_MSSWD = loss_MSSWD.mean() * args.loss_weight_MSSWD

                        total_loss += loss_MSSWD
                        train_loss_MSSWD += loss_log(loss_MSSWD)

                # VSD loss
                if 'vsd' in args.use_loss:
                # if global_step > -1:

                    # Sample noise that we'll add to the predicted original samples
                    noise = torch.randn_like(pred_original_latents)

                    # Sample a random timestep for each image
                    timesteps_range = torch.tensor([0.02, 0.981]) * teacher_noise_scheduler.config.num_train_timesteps
                    timesteps = torch.randint(*timesteps_range.long(), (bsz,), device=accelerator.device).long()

                    # Add noise to the predicted original samples according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_samples = teacher_noise_scheduler.add_noise(pred_original_latents, noise, timesteps)

                    # Prepare outputs from the teacher
                    with torch.no_grad():
                        teacher_pred_cond = teacher_unet_fix(noisy_samples, timesteps, teacher_prompt_embeds).sample
                        if args.teacher_guidance_scale > 1.0:
                            teacher_pred_uncond = teacher_unet_fix(noisy_samples, timesteps, prompt_null_embeds).sample

                        lora_pred_cond = teacher_unet_tune(noisy_samples, timesteps, teacher_prompt_embeds).sample
                        if args.teacher_guidance_scale > 1.0:
                            lora_pred_uncond = teacher_unet_tune(noisy_samples, timesteps, prompt_null_embeds).sample

                        # Apply classifier-free guidance to the teacher prediction
                        if args.teacher_guidance_scale > 1.0:
                            teacher_pred = teacher_pred_uncond + args.teacher_guidance_scale * (teacher_pred_cond - teacher_pred_uncond)
                            lora_pred = lora_pred_uncond + args.teacher_guidance_scale * (lora_pred_cond - lora_pred_uncond)
                        else:
                            teacher_pred = teacher_pred_cond
                            lora_pred = lora_pred_cond

                    # Compute the score gradient for updating the model
                    sigma_t = ((1 - teacher_alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1, 1)
                    # vsd or sds
                    distance = teacher_pred - lora_pred   # vsd
                    # distance = teacher_pred - noise   # sds
                    score_gradient = torch.nan_to_num(sigma_t**2 * distance)

                    # Compute the VSD loss for the model
                    target = (pred_original_latents - score_gradient).detach()
                    loss_vsd = 0.5 * F.mse_loss(pred_original_latents.float(), target.float(), reduction="mean")

                    total_loss += loss_vsd

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss_vsd = accelerator.gather(loss_vsd.repeat(args.train_batch_size)).mean()
                    train_loss_vsd += avg_loss_vsd.item() / args.gradient_accumulation_steps
                
                # backward all losses
                accelerator.backward(total_loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_token is not None and "textencoder" in args.train_student:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = student_text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = student_text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                    for i in range(len(modifier_token_id[1:])):
                        index_grads_to_zero = index_grads_to_zero & (
                            torch.arange(len(tokenizer)) != modifier_token_id[i]
                        )
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                        index_grads_to_zero, :
                    ].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(student_text_encoder.parameters(), student_custom_diffusion_layers.parameters())
                        if args.modifier_token is not None and "textencoder" in args.train_student
                        else student_custom_diffusion_layers.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                ############## End VSD Loss ##############

                if 'vsd' in args.use_loss:
                    ############## Lora Loss ##############
                    # Sample noise that we'll add to the predicted original samples
                    noise = torch.randn_like(pred_original_latents.detach())

                    # Sample a random timestep for each image
                    timesteps_range = torch.tensor([0, 1]) * teacher_noise_scheduler.config.num_train_timesteps
                    timesteps = torch.randint(*timesteps_range.long(), (bsz,), device=accelerator.device).long()

                    # Add noise to the predicted original samples according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_samples = teacher_noise_scheduler.add_noise(pred_original_latents.detach(), noise, timesteps)

                    # Compute output for updating the LoRA teacher
                    # if args.teacher_guidance_scale > 1.0:
                    #     lora_pred_cond = teacher_unet_tune(noisy_samples, timesteps, teacher_prompt_embeds).sample
                    #     lora_pred_uncond = teacher_unet_tune(noisy_samples, timesteps, prompt_null_embeds).sample
                    #     lora_pred = lora_pred_uncond + args.teacher_guidance_scale * (lora_pred_cond - lora_pred_uncond)
                    # else:
                    encoder_hidden_states = prompt_null_embeds if random.random() < 0.1 else teacher_prompt_embeds
                    lora_pred = teacher_unet_tune(noisy_samples, timesteps, encoder_hidden_states).sample

                    alpha_t = (teacher_alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1, 1)
                    lora_pred = alpha_t * lora_pred
                    target = alpha_t * noise

                    # Compute the loss for LoRA teacher
                    loss_lora = F.mse_loss(lora_pred.float(), target.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss_lora = accelerator.gather(loss_lora.repeat(args.train_batch_size)).mean()
                    train_loss_lora += avg_loss_lora.item() / args.gradient_accumulation_steps

                    # Backpropagate for the LoRA teacher
                    accelerator.backward(loss_lora)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(teacher_unet_tune.parameters(), args.max_grad_norm)
                    optimizer_lora.step()
                    lr_scheduler_lora.step()
                    optimizer_lora.zero_grad(set_to_none=args.set_grads_to_none)
                    ############## End Lora Loss ##############
            
                if args.use_disc:
                    for i, disc in enumerate(disc_list):
                        optimizer_disc = optimizer_disc_list[i]
                        lr_scheduler_disc = lr_scheduler_disc_list[i]

                        # zero grad at first
                        optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                        if args.disc_samples == 'teacher':
                            if args.apply_timesteps_weight:
                                real_loss = adversarial_loss_noreduc(disc(teacher_samples), target_real)
                                real_loss = (real_loss * weight_t).mean()
                            else:
                                real_loss = adversarial_loss(disc(teacher_samples), target_real)
                        elif args.disc_samples == 'real':
                            real_loss = adversarial_loss(disc(batch["instance_images"]), target_real)
                        # r1 regularization? other gan loss?
                        fake_loss = adversarial_loss(disc(pred_original_images.detach()), target_fake)
                        loss_disc = (real_loss + fake_loss) / 2
                        # accelerator.clip_grad_norm_(D.parameters(), 1.0)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(disc.classification_head.parameters(), args.max_grad_norm)
                        accelerator.backward(loss_disc)
                        optimizer_disc.step()
                        lr_scheduler_disc.step()
                        # optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                        train_loss_disc_list[i] += loss_log(loss_disc)

            ############## Loss ##############

            

            ############## Check training images ##############
            if accelerator.is_main_process:
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    def process_images(samples, timesteps, prompts):
                        # Clip the image values to [-1, 1]
                        images = torch.clamp(samples.detach(), -1, 1)

                        return [(timesteps[i], prompts[i], images[i]) for i in range(images.size(0))]
                    if args.target_samples == "teacher":
                        teacher_vis = process_images(teacher_samples, timesteps, batch["instance_prompt"])
                    student_vis = process_images(pred_original_images, torch.zeros((bsz,), dtype=torch.long), batch["instance_prompt"])

                    for tracker in accelerator.trackers:
                        # if tracker.name == "tensorboard":
                        #     np_images = np.stack([np.asarray(img) for img in images])
                        #     tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            log_dict = {
                                    "student_vis": [
                                        wandb.Image(image, caption=f"{i}: {p}")
                                        for i, p, image in student_vis
                                    ]
                                }
                            if args.target_samples == "teacher":
                                log_dict["teacher_vis"] = [
                                        wandb.Image(image, caption=f"{i}: {p}")
                                        for i, p, image in teacher_vis
                                    ]
                            tracker.log(log_dict, step=global_step)
                    torch.cuda.empty_cache()
            ############## End Check training images ##############
            
            logs = {"lr": lr_scheduler.get_last_lr()[0]}
            if 'vsd' in args.use_loss:
                logs["train_loss_vsd"] = train_loss_vsd
                train_loss_vsd = 0.0
            if has_tune_teacher:
                logs["train_loss_lora"] = train_loss_lora
                logs["lr_lora"] = lr_scheduler_lora.get_last_lr()[0]
                train_loss_lora = 0.0
            if 'mse' in args.use_loss:
                logs["train_loss_mse"] = train_loss_mse
                train_loss_mse = 0.0
            if 'ipadapter' in args.use_loss:
                logs["train_loss_ipadapter"] = train_loss_ipadapter
                train_loss_ipadapter = 0.0
            if 'lpips' in args.use_loss:
                logs["train_loss_lpips"] = train_loss_lpips
                train_loss_lpips = 0.0
            if 'MSSWD' in args.use_loss:
                logs["train_loss_MSSWD"] = train_loss_MSSWD
                train_loss_MSSWD = 0.0
            if 'gan' in args.use_loss:
                for i, value in enumerate(train_loss_gan_list):
                    logs[f"train_loss_gan_{i}"] = value
                    train_loss_gan_list[i] = 0.0
            if args.use_disc:
                for i, value in enumerate(train_loss_disc_list):
                    logs[f"train_loss_disc_{i}"] = value
                    train_loss_disc_list[i] = 0.0
            if args.train_custom_teacher:
                logs["train_loss_teacher"] = train_loss_teacher
                train_loss_teacher = 0.0
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            # finish save

            if global_step >= args.max_train_steps:
                break

    # Save the custom diffusion layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if "textencoder" in args.train_student:
            chosen_text_encoder = student_text_encoder
        else:
            chosen_text_encoder = teacher_text_encoder
        def save_custom(save_text_encoder, save_unet, save_folder):
            save_new_embed(
                save_text_encoder,
                modifier_token_id,
                accelerator,
                args,
                save_folder,
                safe_serialization=not args.no_safe_serialization,
            )
            save_unet = save_unet.to(torch.float32)
            save_unet.save_attn_procs(save_folder, safe_serialization=not args.no_safe_serialization)
        save_custom(chosen_text_encoder, student_unet, args.output_dir)
        if args.train_custom_teacher and not args.not_save_teacher:
            teacher_save_path = os.path.join(args.output_dir, 'teacher')
            os.makedirs(teacher_save_path, exist_ok=True)
            save_custom(teacher_text_encoder, teacher_unet_fix, teacher_save_path)

        # Final inference
        # Load previous pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype
        )
        # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        weight_name = (
            "pytorch_custom_diffusion_weights.safetensors"
            if not args.no_safe_serialization
            else "pytorch_custom_diffusion_weights.bin"
        )
        pipeline.unet.load_attn_procs(args.output_dir, weight_name=weight_name)
        for token in args.modifier_token:
            token_weight_name = f"{token}.safetensors" if not args.no_safe_serialization else f"{token}.bin"
            pipeline.load_textual_inversion(args.output_dir, weight_name=token_weight_name)

        # run inference
        if args.validation_prompt and args.num_validation_images > 0:
            use_prompt = [args.instance_prompt, args.validation_prompt]
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            images = [
                        (i, p, pipeline(p, num_inference_steps=args.student_infer_steps, generator=generator, guidance_scale=0.0).images[0])
                        for i in range(args.num_validation_images) for p in use_prompt
                    ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {p}")
                                for i, p, image in images
                            ]
                        }, step=global_step
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
            api = HfApi(token=args.hub_token)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
