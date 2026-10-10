# Use the modified diffusers & peft library
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from PIL import Image
import numpy as np
import argparse
import random
from diffusers import FluxFillPipeline as Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="data/logo_product_test/grid",
                    help="Name of the test folder")
parser.add_argument("--output-dir", type=str, default="output/20260730-083648/17000",
                    help="Directory to save the output image")
parser.add_argument("--model-path", type=str, default="black-forest-labs/FLUX.1-Fill-dev",
                    help="Path to the model")
parser.add_argument("--lora-path", type=str, default="checkpoints/pytorch_lora_weights.safetensors",
                    help="Path to the LoRA weights")
parser.add_argument("--gpu", type=str, default="5", help="GPU id to use (e.g., '0' or '0,1')")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

pipe = Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(args.lora_path)
pipe.enable_model_cpu_offload()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

files = os.listdir(args.folder)
images = sorted([
    os.path.join(args.folder, fn)
    for fn in files
    if (fn.lower().endswith('.png') and not fn.endswith('p.png'))
])

mask_decomp = Image.open("data/mask_1x3_011.png")
mask_comp = Image.open("data/mask_1x3_100.png")

for idx, image_path in enumerate(images, start=1):
    print(f"[{idx}/{len(images)}] Processing {image_path}")
    fn = os.path.basename(image_path)
    image = Image.open(image_path).convert("RGB")
    assert image.size == mask_decomp.size, f"Mask size mismatch for {fn}"

    with open(os.path.join(args.folder, fn.replace('.png', '_decomp.txt')), 'r') as f:
        instruction_decomp = f.read().strip()

    with open(os.path.join(args.folder, fn.replace('.png', '_comp.txt')), 'r') as f:
        instruction_comp = f.read().strip()

    width, height = image.size

    # Decomposition step
    result_image_decomp = pipe(
        prompt=instruction_decomp,
        image=image,
        mask_image=mask_decomp,
        height=height,
        width=width,
        guidance_scale=50,
        num_inference_steps=50,
        generator=torch.Generator(device).manual_seed(args.seed)
    ).images[0]
    result_image_decomp.save(os.path.join(output_dir, fn.replace('.png', '_decomp.png')))

    # Composition step
    result_image_decomp_comp = pipe(
        prompt=instruction_comp,
        image=result_image_decomp,
        mask_image=mask_comp,
        height=height,
        width=width,
        guidance_scale=50,
        num_inference_steps=50,
        generator=torch.Generator(device).manual_seed(args.seed)
    ).images[0]
    result_image_decomp_comp.save(os.path.join(output_dir, fn.replace('.png', '_recomp.png')))

print("✅ Inference finished.")
