"""Run one decomposition with the FLUX Fill LoRA."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps


PANEL_SIZE = 512
GRID_SIZE = (PANEL_SIZE * 3, PANEL_SIZE)
DEFAULT_MASK = Path(__file__).resolve().parent / "data" / "mask_1x3_011.png"


def run_inference(
    input_image, prompt, input_format, output_dir, pipeline, mask_path, seed,
    model_path=None, lora_path=None,
):
    import torch

    input_image = Path(input_image)
    output_dir = Path(output_dir)
    if not input_image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_image}")
    if not prompt.strip():
        raise ValueError("Decomposition prompt must not be empty")
    if input_format not in {"grid", "image"}:
        raise ValueError("Input format must be 'grid' or 'image'")

    with Image.open(input_image) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")

    if input_format == "grid":
        if original.size != GRID_SIZE:
            raise ValueError(f"Grid input must be 1536x512 pixels; got {original.size}")
        grid = original.copy()
    else:
        panel = ImageOps.pad(original, (PANEL_SIZE, PANEL_SIZE), color="black")
        grid = Image.new("RGB", GRID_SIZE, "black")
        grid.paste(panel, (0, 0))

    with Image.open(mask_path) as mask_file:
        mask = mask_file.convert("L")
    if mask.size != GRID_SIZE:
        raise ValueError(f"Decomposition mask must be 1536x512 pixels; got {mask.size}")

    run_name = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    suffix = 1
    while True:
        run_dir = output_dir / (run_name if suffix == 1 else f"{run_name}_{suffix}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
    original.save(run_dir / "input.png")
    grid.save(run_dir / "input_grid.png")
    (run_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input_image": str(input_image.resolve()),
                "input_format": input_format,
                "prompt": prompt,
                "model_path": str(model_path) if model_path is not None else None,
                "lora_path": str(lora_path) if lora_path is not None else None,
                "seed": seed,
                "guidance_scale": 50,
                "num_inference_steps": 50,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = pipeline(
        prompt=prompt,
        image=grid,
        mask_image=mask,
        height=GRID_SIZE[1],
        width=GRID_SIZE[0],
        guidance_scale=50,
        num_inference_steps=50,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    result.save(run_dir / "decomp.png")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", type=Path, required=True, help="Path to one input image")
    parser.add_argument("--decomp-prompt", required=True, help="Decomposition instruction text")
    parser.add_argument(
        "--input-format", choices=("grid", "image"), required=True,
        help="Whether the input is already a 1x3 grid or a single image",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model-path", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--lora-path", type=Path, default=Path("checkpoints/pytorch_lora_weights.safetensors"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from diffusers import FluxFillPipeline

    pipeline = FluxFillPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipeline.load_lora_weights(str(args.lora_path))
    pipeline.enable_model_cpu_offload()
    run_dir = run_inference(
        args.input_image, args.decomp_prompt, args.input_format,
        args.output_dir, pipeline, DEFAULT_MASK, args.seed,
        args.model_path, args.lora_path,
    )
    print(f"Saved decomposition to {run_dir}")


if __name__ == "__main__":
    main()
