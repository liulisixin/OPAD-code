#!/usr/bin/env python3
import argparse
import math
import os

import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline
from PIL import Image


DEFAULT_PROMPTS = [
    "a <new1> dog in the jungle",
    "a <new1> dog in the snow",
    "a <new1> dog on the beach",
    "a <new1> dog with a mountain in the background",
]


def save_grid(images, output_path):
    grid_size = int(math.sqrt(len(images)))
    if grid_size * grid_size < len(images):
        grid_size += 1

    width, height = images[0].size
    grid = Image.new("RGB", (width * grid_size, height * grid_size), color="white")
    for i, image in enumerate(images):
        grid.paste(image, ((i % grid_size) * width, (i // grid_size) * height))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid.save(output_path)
    print("save to:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight OPAD inference.")
    parser.add_argument("--model_path", type=str, default="outputs/opad_dog/dog")
    parser.add_argument("--output_path", type=str, default="outputs/opad_dog/dog/inference/grid.png")
    parser.add_argument("--prompt", type=str, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--seed", type=int, default=77)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.load_attn_procs(args.model_path, weight_name="pytorch_custom_diffusion_weights.bin")
    pipeline.load_textual_inversion(args.model_path, weight_name="<new1>.bin")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = []
    for prompt in args.prompt:
        image = pipeline(
            prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        images.append(image)

    save_grid(images, args.output_path)


if __name__ == "__main__":
    main()
