#!/usr/bin/env python3
import argparse
import csv
import glob
import os

import clip
import torch
# from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm
# from textboost.text_encoder import TextBoostModel


STABLE_DIFFUSION = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21base": "stabilityai/stable-diffusion-2-1-base",
    "sd21": "stabilityai/stable-diffusion-2-1",
}

INSTANCES = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "cat": "cat",
    "cat2": "cat",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal",
}

OBJ_PROMPTS = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} with a wheat field in the background',
    'a {0} with a tree and autumn leaves in the background',
    'a {0} with the Eiffel Tower in the background',
    'a {0} floating on top of water',
    'a {0} floating in an ocean of milk',
    'a {0} on top of green grass with sunflowers around it',
    'a {0} on top of a mirror',
    'a {0} on top of the sidewalk in a crowded street',
    'a {0} on top of a dirt road',
    'a {0} on top of a white rug',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]

LIVE_PROMPTS = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} wearing a red hat',
    'a {0} wearing a santa hat',
    'a {0} wearing a rainbow scarf',
    'a {0} wearing a black top hat and a monocle',
    'a {0} in a chef outfit',
    'a {0} in a firefighter outfit',
    'a {0} in a police outfit',
    'a {0} wearing pink glasses',
    'a {0} wearing a yellow shirt',
    'a {0} in a purple wizard outfit',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to model", default=None)   
    parser.add_argument(
        "--token_format",
        type=str,
        default="<new1> CLS",
        help="Token format for the prompt. Use [sks SUBJECT] for DreamBooth, [<INSTANCE>] for Textual Inversion, or [<INSTANCE> SUBJECT] for Custom Diffusion/TextBoost."
    )
    parser.add_argument("--outdir", type=str, default="./benchmarks")
    # parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--metric", type=str, nargs="+", default=["clip-i", "dino"])
    parser.add_argument("--has_unseen", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dreambooth_path", type=str, default=os.environ.get("DREAMBOOTH_DATA_DIR", "../dreambooth/dataset"))
    # parser.add_argument("--train-dir", type=str, default="./data/dreambooth_n1_train")
    # parser.add_argument("--val-dir", type=str, default="./data/dreambooth_n1_val")
    # parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output-desc", type=str, default=None)


    parser.add_argument("--out_setting", type=str, default="student")

    return parser.parse_args()


def is_live(instance):
    cls = INSTANCES[instance]
    if cls in ("cat", "dog"):
        return True
    return False


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_str=False):
        self.root = root
        self.transform = transform
        self.return_str = return_str
        self.files = glob.glob(f"{root}/*/*.png")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        basename = os.path.basename(path)
        instance = os.path.dirname(path).split("/")[-1]
        prompt = basename.replace(".png", "").replace("_", " ")

        if self.return_str:
            return path, instance, prompt

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, instance, prompt


def load_pipeline(new_model_path, pretrained_model):
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model, torch_dtype=torch.float16,
    ).to("cuda")

    path_to_save_model = new_model_path

    pipeline.unet.load_attn_procs(path_to_save_model, weight_name="pytorch_custom_diffusion_weights.bin")
    pipeline.load_textual_inversion(path_to_save_model, weight_name="<new1>.bin")

    return pipeline


def generate_from_pipeline(
        pipeline,
        instance,
        identifier,
        seed,
        outdir,
        batch_size=8,
        device="cuda",
        out_setting="student",
):
    assert instance in INSTANCES, f"Invalid instance: {instance}"
    prompt_list = LIVE_PROMPTS if is_live(instance) else OBJ_PROMPTS

    if outdir.endswith("/"):
        outdir = outdir[:-1]

    cls = INSTANCES[instance]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    print(f"[seed {seed}")

    i = 0
    while i < len(prompt_list):
        prompts = []
        for _ in range(batch_size):
            prompts.append(prompt_list[i].format(identifier))
            i += 1
            if i >= len(prompt_list):
                break

        print(len(prompts))
        print(prompts)

        if out_setting == 'student':
            images = pipeline(
                prompt=prompts,
                num_inference_steps=1,
                guidance_scale=0.0,  # NOTE: default value.
                generator=generator,
            ).images
        elif out_setting == 'student_2step':
            images = pipeline(
                prompt=prompts,
                num_inference_steps=2,
                guidance_scale=0.0,  # NOTE: default value.
                generator=generator,
            ).images
        elif out_setting == 'student_4step':
            images = pipeline(
                prompt=prompts,
                num_inference_steps=4,
                guidance_scale=0.0,  # NOTE: default value.
                generator=generator,
            ).images
        elif out_setting == 'teacher':
            images = pipeline(
                prompt=prompts,
                num_inference_steps=25,
                guidance_scale=7.5,  # NOTE: default value.
                generator=generator,
            ).images
        elif out_setting == 'student_hypersd':
            raise NotImplementedError("student_hypersd is not included in this release.")
        else:
            raise NotImplementedError

        for prompt, image in zip(prompts, images):
            dst = os.path.join(outdir, f"seed{seed}", instance)
            os.makedirs(dst, exist_ok=True)
            filename = f"{prompt.replace(identifier, cls).replace(' ', '_')}.png"
            image.save(
                os.path.join(dst, filename)
            )
    del pipeline, generator


def generate(args, device):
    if args.instances is not None:
        instances = {}
        for name, cls in INSTANCES.items():
            if name in args.instances:
                instances[name] = cls
    else:
        instances = INSTANCES

        subdirs = os.listdir(args.path)
        subdirs = list(filter(lambda x: os.path.isdir(os.path.join(args.path, x)), subdirs))
        # for instance in INSTANCES.keys():
        #     assert instance in subdirs, f"Missing instance: {instance}"
        # assert len(subdirs) == 30, f"Invalid number of instances: {len(subdirs)}"

    if args.outdir.endswith("/"):
        args.outdir = args.outdir[:-1]
    if args.path.endswith("/"):
        args.path = args.path[:-1]

    basename = os.path.basename(args.path)

    outdir = os.path.join(args.outdir, basename + '_' + args.out_setting)
    if args.skip_gen:
        return outdir


    for instance in tqdm(instances):
        if args.out_setting == 'student_hypersd':
            raise NotImplementedError("student_hypersd is not included in this release.")
        else:
            if args.out_setting in ['student', 'student_2step', 'student_4step']:
                new_model_path = os.path.join(args.path, instance)
                scheduler_name = 'EulerAncestralDiscreteScheduler'
                pretrained_model = "stabilityai/sd-turbo"
            elif args.out_setting == 'teacher':
                new_model_path = os.path.join(args.path, instance, 'teacher')
                scheduler_name = 'DDIMScheduler'
                pretrained_model = "stabilityai/stable-diffusion-2-1"
            else:
                raise NotImplementedError

            pipeline = load_pipeline(new_model_path, pretrained_model)

            if scheduler_name == 'EulerDiscreteScheduler':
                pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            elif scheduler_name == 'EulerAncestralDiscreteScheduler':
                pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
            elif scheduler_name == 'DDIMScheduler':
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            else:
                raise NotImplementedError

        pipeline = pipeline.to(device)
        print(pipeline.tokenizer)

        identifier = args.token_format.replace("CLS", INSTANCES[instance])

        for seed in args.seeds:
            generate_from_pipeline(
                pipeline=pipeline,
                instance=instance,
                identifier=identifier,
                seed=seed,
                outdir=outdir,
                batch_size=4,
                device=device,
                out_setting=args.out_setting,
            )
    return outdir


def clip_score(generated_image_path, device):
    import t2v_metrics
    score = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336', device=device)
    score.eval().requires_grad_(False)

    def _path_to_prompt(path):
        basename = os.path.basename(path)  # prompt.png
        return basename.replace(".png", "").replace("_", " ")
    
    scores = {}
    scores_list = []
    for instance in sorted(os.listdir(generated_image_path)):
        image_paths = sorted(glob.glob(os.path.join(generated_image_path, instance, "*.png")))
        dataset = [
            {"images": [image_path], "texts": [_path_to_prompt(image_path)]}
            for image_path in image_paths
        ]
        score_instance = score.batch_forward(dataset=dataset, batch_size=32)
        scores_list.append(score_instance)
        scores[instance] = score_instance.mean()

    del score
    torch.cuda.empty_cache()

    scores_list = torch.cat(scores_list)
    print(f"Total samples: {scores_list.size(0)}")
    print(f"CLIP-T: {scores_list.mean():.3f} +/- {scores_list.std():.3f}")
    return {"clip_score": scores}


def clip_i_old(args, generated_image_path, device):
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # model, preprocess = clip.load("ViT-B/32", device=device)  # same as Custom Diffusion.
    model.eval().requires_grad_(False)
    preprocess = v2.Compose([
        v2.Resize((512, 512)),
        preprocess,
    ])

    if args.has_unseen:
        train_dir = args.train_dir
        test_dir = args.val_dir
        seen_images = sorted(glob.glob(os.path.join(train_dir, "*/*.*")))
        unseen_images = sorted(glob.glob(os.path.join(test_dir, "*/*.*")))
    else:
        seen_images = sorted(glob.glob(os.path.join(args.dreambooth_path, "*/*.*")))
    instance_to_id = {}

    id = 0
    seen_data = {}
    for image in seen_images:
        instance = os.path.basename(os.path.dirname(image))
        image = Image.open(image).convert("RGB")
        if instance in seen_data:
            seen_data[instance].append(preprocess(image))
        else:
            seen_data[instance] = [preprocess(image)]
            instance_to_id[instance] = id
            id += 1

    if args.has_unseen:
        unseen_data = {}
        for image in unseen_images:
            instance = os.path.basename(os.path.dirname(image))
            image = Image.open(image).convert("RGB")
            if instance in unseen_data:
                unseen_data[instance].append(preprocess(image))
            else:
                unseen_data[instance] = [preprocess(image)]

    seen_scores = []
    if args.has_unseen:
        unseen_scores = []
    n = 0
    for instance in os.listdir(generated_image_path):
        images = sorted(glob.glob(os.path.join(generated_image_path, instance, "*.png")))
        images = torch.stack([
            preprocess(Image.open(image).convert("RGB"))
            for image in images
        ])
        image_features = model.encode_image(images.to(device))  # 25 prompts per instance

        # Compare to seen images.
        train_batch = torch.stack(seen_data[instance])
        seen_feature = model.encode_image(train_batch.to(device))  # num_seen, D
        for seen_feat in seen_feature.unbind(0):
            seen_feat = seen_feat.unsqueeze(0)
            seen_score = torch.cosine_similarity(image_features, seen_feat, dim=1)
            seen_score = torch.maximum(seen_score, torch.zeros_like(seen_score))
            seen_scores.append(seen_score)

        if args.has_unseen:
            # Compare to unseen images.
            test_batch = torch.stack(unseen_data[instance])
            unseen_feature = model.encode_image(test_batch.to(device))  # num_seen, D
            for unseen_feat in unseen_feature.unbind(0):
                unseen_feat = unseen_feat.unsqueeze(0)
                unseen_score = torch.cosine_similarity(image_features, unseen_feat, dim=1)
                unseen_score = torch.maximum(unseen_score, torch.zeros_like(unseen_score))
                unseen_scores.append(unseen_score)

        n += images.shape[0]

    seen_scores = torch.cat(seen_scores)
    if args.has_unseen:
        unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {n}")
    print(f"CLIP-I (seen)  : {seen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    if args.has_unseen:
        print(f"CLIP-I (unseen): {unseen_scores.mean():.3f} +/- {unseen_scores.std():.3f}")
        return {"clip_i": seen_scores, "clip_i_unseen": unseen_scores}
    else:
        return {"clip_i": seen_scores}


def clip_i(args, generated_image_path, device):
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # model, preprocess = clip.load("ViT-B/32", device=device)  # same as Custom Diffusion.
    model.eval().requires_grad_(False)
    preprocess = v2.Compose([
        v2.Resize((512, 512)),
        preprocess,
    ])
    return image_compare(args, generated_image_path, device, model.encode_image, preprocess, print_name="clip_i")


def image_compare(args, generated_image_path, device, model_forward, preprocess, print_name=""):
    if args.has_unseen:
        train_dir = args.train_dir
        test_dir = args.val_dir
        seen_images = sorted(glob.glob(os.path.join(train_dir, "*/*.*")))
        unseen_images = sorted(glob.glob(os.path.join(test_dir, "*/*.*")))
    else:
        seen_images = sorted(glob.glob(os.path.join(args.dreambooth_path, "*/*.*")))
    instance_to_id = {}

    id = 0
    seen_data = {}
    for image in seen_images:
        instance = os.path.basename(os.path.dirname(image))
        image = Image.open(image).convert("RGB")
        if instance in seen_data:
            seen_data[instance].append(preprocess(image))
        else:
            seen_data[instance] = [preprocess(image)]
            instance_to_id[instance] = id
            id += 1

    if args.has_unseen:
        unseen_data = {}
        for image in unseen_images:
            instance = os.path.basename(os.path.dirname(image))
            image = Image.open(image).convert("RGB")
            if instance in unseen_data:
                unseen_data[instance].append(preprocess(image))
            else:
                unseen_data[instance] = [preprocess(image)]

    seen_scores = {}
    seen_scores_list = []
    if args.has_unseen:
        unseen_scores = []
    n = 0
    for instance in sorted(os.listdir(generated_image_path)):
        images = sorted(glob.glob(os.path.join(generated_image_path, instance, "*.png")))
        images = torch.stack([
            preprocess(Image.open(image).convert("RGB"))
            for image in images
        ])
        image_features = model_forward(images.to(device))  # 25 prompts per instance

        # Compare to seen images.
        train_batch = torch.stack(seen_data[instance])
        seen_feature = model_forward(train_batch.to(device))  # num_seen, D
        result_instance = []
        for seen_feat in seen_feature.unbind(0):
            seen_feat = seen_feat.unsqueeze(0)
            seen_score = torch.cosine_similarity(image_features, seen_feat, dim=1)
            seen_score = torch.maximum(seen_score, torch.zeros_like(seen_score))
            result_instance.append(seen_score)
            seen_scores_list.append(seen_score)
        seen_scores[instance] = torch.cat(result_instance).mean()

        if args.has_unseen:
            # Compare to unseen images.
            test_batch = torch.stack(unseen_data[instance])
            unseen_feature = model_forward(test_batch.to(device))  # num_seen, D
            for unseen_feat in unseen_feature.unbind(0):
                unseen_feat = unseen_feat.unsqueeze(0)
                unseen_score = torch.cosine_similarity(image_features, unseen_feat, dim=1)
                unseen_score = torch.maximum(unseen_score, torch.zeros_like(unseen_score))
                unseen_scores.append(unseen_score)

        n += images.shape[0]

    seen_scores_list = torch.cat(seen_scores_list)
    if args.has_unseen:
        unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {n}")
    print(f"{print_name} (seen)  : {seen_scores_list.mean():.3f} +/- {seen_scores_list.std():.3f}")
    if args.has_unseen:
        print(f"{print_name} (unseen): {unseen_scores.mean():.3f} +/- {unseen_scores.std():.3f}")
        return {f"{print_name}": seen_scores, f"{print_name}_unseen": unseen_scores}
    else:
        return {f"{print_name}": seen_scores}


def dino_score(args, generated_image_path, device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval().requires_grad_(False).to(device)
    preprocess = v2.Compose([
        v2.Resize((512, 512)),
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return image_compare(args, generated_image_path, device, model, preprocess, print_name="dino")


def dino_score_old(args, generated_image_path, device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval().requires_grad_(False).to(device)
    preprocess = v2.Compose([
        v2.Resize((512, 512)),
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dreambooth = {}
    dreambooth_images = sorted(glob.glob(os.path.join(args.dreambooth_path, "**/*.*")))
    instance_to_id = {}

    id = 0
    for image in dreambooth_images:
        instance = os.path.basename(os.path.dirname(image))
        image = Image.open(image).convert("RGB")
        if instance in dreambooth:
            dreambooth[instance].append(preprocess(image))
        else:
            dreambooth[instance] = [preprocess(image)]
            instance_to_id[instance] = id
            id += 1

    max_samples = 0
    num_samples = {}
    for instance, images in dreambooth.items():
        id = instance_to_id[instance]
        max_samples = max(max_samples, len(images))
        num_samples[id] = len(images)

    db_batch = torch.zeros(len(dreambooth), max_samples, 3, 224, 224)
    # db_batch = torch.zeros(len(dreambooth), max_samples, 3, 336, 336)
    for instance, images in dreambooth.items():
        id = instance_to_id[instance]
        db_batch[id, :num_samples[id]] = torch.stack(images)


    # N = int(args.path.split("-")[2].split("n")[-1])
    dataloader = torch.utils.data.DataLoader(
        Dataset(generated_image_path, transform=preprocess),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    seen_scores = []
    if args.has_unseen:
        unseen_scores = []
    n = 0
    for image, instance, prompt in dataloader:
        instance = list(map(lambda x: instance_to_id[x], instance))
        instance = torch.as_tensor(instance, dtype=torch.long)
        token = clip.tokenize(prompt)
        image_feature = model(image.to(device))

        if not args.has_unseen:
            N = db_batch[instance].size(1)
        # Compare to seen images.
        train_batch = db_batch[instance][:, :N, :, :, :].to(device)
        seen_i_score = 0.0
        for i, train_image in enumerate(train_batch.unbind(1)):
            seen_i_score += torch.cosine_similarity(image_feature, model(train_image), dim=-1)
        seen_i_score /= N

        if args.has_unseen:
            # Compare to unseen images.
            unseen_batch = db_batch[instance][:, N:, :, :, :].to(device)
            unseen_i_score = 0.0
            num_unseen = torch.as_tensor([num_samples[id.item()]-N for id in instance], device=device)
            for i, unseen_image in enumerate(unseen_batch.unbind(1)):
                sim = torch.cosine_similarity(image_feature, model(unseen_image), dim=-1)
                mask = torch.ones(len(instance), device=device) * i
                mask = mask < torch.as_tensor(num_unseen, device=device)
                sim = mask * sim
                unseen_i_score += sim
            unseen_i_score /= num_unseen

        n += image.shape[0]
        seen_scores.append(seen_i_score)
        if args.has_unseen:
            unseen_scores.append(unseen_i_score)

    seen_scores = torch.cat(seen_scores)
    if args.has_unseen:
        unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {n}")
    print(f"DINO: {seen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    if args.has_unseen:
        print(f"DINO (unseen): {unseen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
        return {"dino": seen_scores, "dino_unseen": unseen_scores}
    else:
        return {"dino": seen_scores}


def vqa_score(args, generated_image_path, device):
    import t2v_metrics
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl', device=device)
    clip_flant5_score.eval().requires_grad_(False)

    def _path_to_prompt(path):
        basename = os.path.basename(path)  # prompt.png
        return basename.replace(".png", "").replace("_", " ")
    
    scores = {}
    scores_list = []
    for instance in sorted(os.listdir(generated_image_path)):
        image_paths = sorted(glob.glob(os.path.join(generated_image_path, instance, "*.png")))
        dataset = [
            {"images": [image_path], "texts": [_path_to_prompt(image_path)]}
            for image_path in image_paths
        ]
        score_instance = clip_flant5_score.batch_forward(dataset=dataset, batch_size=32)
        scores_list.append(score_instance)
        scores[instance] = score_instance.mean()

    del clip_flant5_score
    torch.cuda.empty_cache()

    scores_list = torch.cat(scores_list)
    print(f"Total samples: {scores_list.size(0)}")
    print(f"VQA score: {scores_list.mean():.3f} +/- {scores_list.std():.3f}")
    return {"vqa_score": scores}


@torch.inference_mode()
def main(args):
    if args.path.endswith("/"):
        args.path = args.path[:-1]

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    generated_image_path = generate(args, device)
    if args.instances is not None:
        instance_list = sorted(args.instances)
    else:
        instance_list = sorted(list(INSTANCES.keys()))

    # Save scores to file.
    ckpt = f"_{args.path}"
    desc = f"_{args.out_setting}"
    filename = f"metric{ckpt}{desc}v2.csv"


    score_dict = {
        seed: {
            # Image-text scores
            "clip_score": torch.tensor([0.0]),
            "vqa_score": torch.tensor([0.0]),
            # Image-image scores
            "clip_i": torch.tensor([0.0]),
            "dino": torch.tensor([0.0]),
            **(
            {
                "clip_i_unseen": torch.tensor([0.0]),
                "dino_unseen": torch.tensor([0.0]),
            }
            if args.has_unseen else {}
        ),
        }
        for seed in args.seeds
    }

    for seed in args.seeds:
        path_with_seed = os.path.join(generated_image_path, f"seed{seed}")

        if "clip-t" in args.metric:
            score_dict[seed].update(clip_score(path_with_seed, device))
        if "vqa" in args.metric:
            score_dict[seed].update(vqa_score(args, path_with_seed, device))

        # if "clip-i" in args.metric:
        #     score_dict[seed].update(clip_i_old(args, path_with_seed, device))
        if "clip-i" in args.metric:
            score_dict[seed].update(clip_i(args, path_with_seed, device))
        if "dino" in args.metric:
            score_dict[seed].update(dino_score(args, path_with_seed, device))

    # If not exists, create the file and write header.
    csv_filename = os.path.join(generated_image_path, filename)
    metric_list = list(score_dict[args.seeds[0]].keys())
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        table_list = []
        for seed, s_seed in score_dict.items():
            line = (
                [str(seed)]
                + instance_list
            )
            writer.writerow(line)
            table_seed = []
            for mmm in metric_list:
                s_metric = s_seed[mmm]
                values = [f"{s_metric[x]:.3f}" for x in instance_list]
                line = (
                    [mmm] + values
                )
                table_seed.append(torch.stack([s_metric[x] for x in instance_list]))
                writer.writerow(line)
            table_list.append(torch.stack(table_seed))
    
        line = (
                ['Average']
                + instance_list
            )
        writer.writerow(line)
        
        table_list = torch.stack(table_list)
        mean_table = table_list.mean(0)
        for idx, metric in enumerate(metric_list):
            mean_values = mean_table[idx]
            line = [metric] + [f"{v.item():.3f}" for v in mean_values]
            writer.writerow(line)
        
        writer.writerow(["Average over Instances"])
        for idx, metric in enumerate(metric_list):
            mean_values = mean_table[idx]
            avg_over_instances = mean_values.mean().item()
            line = [metric] + [f"{avg_over_instances:.3f}"]
            writer.writerow(line)
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
