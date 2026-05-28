#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from eval_dreambooth import OBJ_PROMPTS, LIVE_PROMPTS, is_live

# subject_name, class, init_token
INSTANCES = [
    ("backpack", "backpack", "red"),
    ("backpack_dog", "backpack", "color"),
    ("bear_plushie", "stuffed animal", "plushie"),
    ("berry_bowl", "bowl", "white"),
    ("can", "can", "color"),
    ("candle", "candle", "jar"),
    ("cat", "cat", "ginger"),
    ("cat2", "cat", "gray"),
    ("clock", "clock", "yellow"),
    ("colorful_sneaker", "sneaker", "color"),
    ("dog", "dog", "corgi"),
    ("dog2", "dog", "fluffy"),
    ("dog3", "dog", "curly"),
    ("dog5", "dog", "fluffy"),
    ("dog6", "dog", "corgi"),
    ("dog7", "dog", "retriever"),
    ("dog8", "dog", "border collie"),
    ("duck_toy", "toy", "duck"),
    ("fancy_boot", "boot", "white"),
    ("grey_sloth_plushie", "stuffed animal", "brown"),
    ("monster_toy", "toy", "stuffed"),
    ("pink_sunglasses", "glasses", "pink"),
    ("poop_emoji", "toy", "ktn"),
    ("rc_car", "toy", "car"),
    ("red_cartoon", "cartoon", "sketch"),
    ("robot_toy", "toy", "robot"),
    ("shiny_sneaker", "sneaker", "rainbow"),
    ("teapot", "teapot", "brown"),
    ("vase", "vase", "red"),
    ("wolf_plushie", "stuffed animal", "plushie"),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiment')

    parser.add_argument("--output_folder", type=str, default="outputs/opad_dreambooth")
    parser.add_argument("--train_file", type=str, default="train_opad.py")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=30)
    parser.add_argument("--index_list", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DREAMBOOTH_DATA_DIR", "../dreambooth/dataset"))
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--ip_adapter_image_encoder_path", type=str, default=None)
    parser.add_argument("--ip_adapter_ckpt", type=str, default=None)

    args = parser.parse_args()
    return args

def main(args):
    instances = INSTANCES

    outdir = f"{args.output_folder}/"

    os.makedirs(outdir, exist_ok=True)

    data_dir = args.data_dir

    if args.index_list is None:
        instances = instances[args.start_index:args.end_index]
    else:
        use_index = [int(x) for x in args.index_list.split(',')]
        instances = [instances[x] for x in use_index]
    print(instances)

    for name, cls, init_token in instances:
        instance_data_dir = os.path.join(data_dir, name)
        output_dir_ins = os.path.join(outdir, name)
        os.makedirs(output_dir_ins, exist_ok=True)
        instance_prompt = f"<new1> {cls}"
        validation_prompt = LIVE_PROMPTS[0] if is_live(name) else OBJ_PROMPTS[0]
        validation_prompt = validation_prompt.replace('{0}', instance_prompt)
        cmd = [
            "python", 
            args.train_file,
            "--pretrained_model_name_or_path=stabilityai/sd-turbo",
            f"--instance_data_dir={instance_data_dir}",
            f"--output_dir={output_dir_ins}",
            f"--instance_prompt={instance_prompt}",
            "--resolution=512",
            "--train_batch_size=2",
            "--learning_rate=1.e-5",
            "--lr_warmup_steps=0",
            "--max_train_steps=1000",
            "--scale_lr",
            "--hflip",
            "--modifier_token=<new1>",
            f"--validation_prompt={validation_prompt}",
            f"--report_to={args.report_to}",
            "--no_safe_serialization",
            "--validation_steps=100",
            "--teacher_pretrained_model_name_or_path=sd2-community/stable-diffusion-2-1",
            "--train_custom_teacher",
            "--learning_rate_teacher=1.e-5",
            "--train_student=unet",
            "--learning_rate_lora=1.e-5",
            "--teacher_guidance_scale=7.5",
            "--teacher_infer_steps=25",
            "--student_infer_steps=1",
            "--use_loss=ipadapter_latentmse_gan_MSSWD",
            "--dataset_type=CustomDiffusionDatasetV2",
            "--target_samples=teacher",
            "--apply_timesteps_weight",
            "--use_disc",
            "--disc_model=vit_small_patch16_224.dino+vit_large_patch14_dinov2.lvd142m+IPAdapterDisc",
            "--disc_head_layer=2",
            "--learning_rate_disc=1.e-4",
            "--disc_samples=real",
            "--feed_teacher_student",
            "--feed_teacher_student_infer_steps=0",
            "--feed_teacher_student_denoise_steps=1",
            f"--initializer_token={init_token}",
        ]

        if args.ip_adapter_image_encoder_path is not None:
            cmd.append(f"--ip_adapter_image_encoder_path={args.ip_adapter_image_encoder_path}")
        if args.ip_adapter_ckpt is not None:
            cmd.append(f"--ip_adapter_ckpt={args.ip_adapter_ckpt}")

        subprocess.run(cmd, check=True)

        # save cmd as text file
        cmd_txt = "\n".join(cmd)
        cmd_txt_path = os.path.join(output_dir_ins, 'cmd.txt')
        with open(cmd_txt_path, "w") as file:
            file.write(cmd_txt)
        shutil.copy(args.train_file, os.path.join(output_dir_ins, args.train_file))


if __name__ == "__main__":
    args = parse_args()
    main(args)