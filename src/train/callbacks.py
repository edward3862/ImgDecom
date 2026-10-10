import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
# import cv2
import torch
import os
from datetime import datetime

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 1)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")
        self.condition_type = training_config.get("condition_type", "logo_decomposition")

        self.total_steps = 0

    def on_train_start(self, trainer, pl_module):
        print(
                f"Epoch: {trainer.current_epoch}, Steps: {trainer.global_step} - Generating a sample"
            )
        self.generate_a_sample(
            trainer,
            pl_module,
            f"{self.save_path}/{self.run_name}",
            f"lora_{self.total_steps}",
            self.condition_type
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        # And also save PL checkpoint for future resuming
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )
            # print(
            #     f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving Checkpoint"
            # )
            # trainer.save_checkpoint(
            #     f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}/pl_trainer.ckpt"
            # )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}",
                f"lora_{self.total_steps}",
                self.condition_type
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type,
    ):
        print(f"Generating sample for condition type: {condition_type}")

        if condition_type == "cycle":
            grid_width = 3
            self.condition_size = 512
            
            # file_name = [
            #     "data/logo_product_4969_cycle/test/04500.png",
            #     "data/logo_product_4969_cycle/test/04501.png",
            #     "data/logo_product_4969_cycle/test/04502.png",
            #     "data/logo_product_4969_cycle/test/04503.png",
            #     "data/logo_product_4969_cycle/test/04504.png",
            #     "data/logo_product_4969_cycle/test/04505.png",
            #     "data/logo_product_4969_cycle/test/04506.png",
            #     "data/logo_product_4969_cycle/test/04507.png",
            #     "data/logo_product_4969_cycle/test/04508.png",
            #     "data/logo_product_4969_cycle/test/04509.png",
            #     "data/logo_product_4969_cycle/test/04510.png",
            #     "data/logo_product_4969_cycle/test/04511.png",
            #     "data/logo_product_4969_cycle/test/04512.png",
            #     "data/logo_product_4969_cycle/test/04513.png",
            #     "data/logo_product_4969_cycle/test/04514.png",
            #     "data/logo_product_4969_cycle/test/04515.png",
            #     "data/logo_product_4969_cycle/test/04516.png",
            #     "data/logo_product_4969_cycle/test/04517.png",
            #     "data/logo_product_4969_cycle/test/04518.png",
            #     "data/logo_product_4969_cycle/test/04519.png",
            # ]
            test_root = "/mnt/d/guzheng/Experiments/In-Context-LoRA/playground/AssetDropper_papershots/test_grids/011_cycle"
            file_name = sorted([os.path.join(test_root, fn) for fn in os.listdir(test_root) if fn.endswith('.png')])
            
            test_instruction_decomp = []
            for fn in file_name:
                with open(fn.replace('.png', '_decomp.txt'), 'r') as f:
                    instruction = f.read().strip()
                    test_instruction_decomp.append(instruction)
            
            pl_module.flux_fill_pipe.transformer.eval()
            for i, name in enumerate(file_name):
                # combined_image_decomp = Image.open(name.replace('.png', '_decomp.png'))
                combined_image_decomp = Image.open(name)
                mask_decomp = Image.new('L', (self.condition_size * grid_width, self.condition_size), 0)
                draw_decomp = ImageDraw.Draw(mask_decomp)
                draw_decomp.rectangle([self.condition_size, 0, self.condition_size * grid_width, self.condition_size], fill=255)
                prompt_decomp = test_instruction_decomp[i]
                image = pl_module.flux_fill_pipe(
                    prompt=prompt_decomp,
                    image=combined_image_decomp,
                    height=512,
                    width=512 * grid_width,
                    mask_image=mask_decomp,
                    guidance_scale=50,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]
                image_name = f'flux-fill-test-{self.total_steps if self.total_steps is not None else 0}-{i}-{condition_type}-decomp'
                image.save(os.path.join(save_path, f'{image_name}.jpg'))


                # test_instruction_comp = []
                # for fn in file_name:
                #     with open(fn.replace('.png', '_comp.txt'), 'r') as f:
                #         instruction = f.read().strip()
                #         test_instruction_comp.append(instruction)
                # combined_image_comp = Image.open(name.replace('.png', '_comp.png'))
                # mask_comp = Image.new('L', (self.condition_size * grid_width, self.condition_size), 0)
                # draw_comp = ImageDraw.Draw(mask_comp)
                # draw_comp.rectangle([0, 0, self.condition_size, self.condition_size], fill=255)
                # prompt_comp = test_instruction_comp[i]
                # image = pl_module.flux_fill_pipe(
                #     prompt=prompt_comp,
                #     image=combined_image_comp,
                #     height=512,
                #     width=512 * grid_width,
                #     mask_image=mask_comp,
                #     guidance_scale=50,
                #     num_inference_steps=30,
                #     max_sequence_length=512,
                #     generator=torch.Generator("cpu").manual_seed(666)
                # ).images[0]
                # image_name = f'flux-fill-test-{self.total_steps if self.total_steps is not None else 0}-{i}-{condition_type}-comp'
                # image.save(os.path.join(save_path, f'{image_name}.jpg'))
                # if self.use_wandb:
                #     wandb_image = wandb.Image(image, caption=image_name)
                #     wandb.log(
                #         {
                #             image_name: wandb_image
                #         }
                #     )

            pl_module.flux_fill_pipe.transformer.train()

        elif condition_type == "cycle_intrinsic":
            grid_width = 3
            self.condition_size = 512
            
            file_name = [
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_001_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_002_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_003_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_004_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_005_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_006_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_007_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_008_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_009_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_010_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_055_009_cam00_frame0001.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_055_009_cam00_frame0002.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_055_009_cam00_frame0003.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_055_009_cam00_frame0004.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2_cycle/test/ai_055_009_cam00_frame0005.png"
                ]
            
            test_instruction_decomp = []
            for fn in file_name:
                with open(fn.replace('.png', '_decomp.txt'), 'r') as f:
                    instruction = f.read().strip()
                    test_instruction_decomp.append(instruction)
            test_instruction_comp = []
            for fn in file_name:
                with open(fn.replace('.png', '_comp.txt'), 'r') as f:
                    instruction = f.read().strip()
                    test_instruction_comp.append(instruction)
            
            pl_module.flux_fill_pipe.transformer.eval()
            for i, name in enumerate(file_name):
                combined_image_decomp = Image.open(name.replace('.png', '_decomp.png'))
                mask_decomp = Image.new('L', (self.condition_size * grid_width, self.condition_size), 0)
                draw_decomp = ImageDraw.Draw(mask_decomp)
                draw_decomp.rectangle([self.condition_size, 0, self.condition_size * grid_width, self.condition_size], fill=255)
                prompt_decomp = test_instruction_decomp[i]
                image = pl_module.flux_fill_pipe(
                    prompt=prompt_decomp,
                    image=combined_image_decomp,
                    height=512,
                    width=512 * grid_width,
                    mask_image=mask_decomp,
                    guidance_scale=50,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]
                image_name = f'flux-fill-test-{self.total_steps if self.total_steps is not None else 0}-{i}-{condition_type}-decomp'
                image.save(os.path.join(save_path, f'{image_name}.jpg'))


                combined_image_comp = Image.open(name.replace('.png', '_comp.png'))
                mask_comp = Image.new('L', (self.condition_size * grid_width, self.condition_size), 0)
                draw_comp = ImageDraw.Draw(mask_comp)
                draw_comp.rectangle([0, 0, self.condition_size, self.condition_size], fill=255)
                prompt_comp = test_instruction_comp[i]
                image = pl_module.flux_fill_pipe(
                    prompt=prompt_comp,
                    image=combined_image_comp,
                    height=512,
                    width=512 * grid_width,
                    mask_image=mask_comp,
                    guidance_scale=50,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]
                image_name = f'flux-fill-test-{self.total_steps if self.total_steps is not None else 0}-{i}-{condition_type}-comp'
                image.save(os.path.join(save_path, f'{image_name}.jpg'))


            pl_module.flux_fill_pipe.transformer.train()

        elif condition_type == "cycle_foreground_background":
            grid_width = 3
            self.condition_size = 512
            
            file_name = [
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000014.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000019.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000029.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000034.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000048.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000059.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000065.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000074.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000083.png",
                    "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000088.png",
                ]
            
            test_instruction_decomp = []
            for fn in file_name:
                with open(fn.replace('.png', '_decomp.txt'), 'r') as f:
                    instruction = f.read().strip()
                    test_instruction_decomp.append(instruction)
            test_instruction_comp = []
            for fn in file_name:
                with open(fn.replace('.png', '_comp.txt'), 'r') as f:
                    instruction = f.read().strip()
                    test_instruction_comp.append(instruction)
            
            pl_module.flux_fill_pipe.transformer.eval()
            for i, name in enumerate(file_name):
                combined_image_decomp = Image.open(name.replace('.png', '_decomp.png'))
                mask_decomp = Image.new('L', (self.condition_size * grid_width, self.condition_size), 0)
                draw_decomp = ImageDraw.Draw(mask_decomp)
                draw_decomp.rectangle([self.condition_size, 0, self.condition_size * grid_width, self.condition_size], fill=255)
                prompt_decomp = test_instruction_decomp[i]
                image = pl_module.flux_fill_pipe(
                    prompt=prompt_decomp,
                    image=combined_image_decomp,
                    height=512,
                    width=512 * grid_width,
                    mask_image=mask_decomp,
                    guidance_scale=50,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]
                image_name = f'flux-fill-test-{self.total_steps if self.total_steps is not None else 0}-{i}-{condition_type}-decomp'
                image.save(os.path.join(save_path, f'{image_name}.jpg'))


                combined_image_comp = Image.open(name.replace('.png', '_comp.png'))
                mask_comp = Image.new('L', (self.condition_size * grid_width, self.condition_size), 0)
                draw_comp = ImageDraw.Draw(mask_comp)
                draw_comp.rectangle([0, 0, self.condition_size, self.condition_size], fill=255)
                prompt_comp = test_instruction_comp[i]
                image = pl_module.flux_fill_pipe(
                    prompt=prompt_comp,
                    image=combined_image_comp,
                    height=512,
                    width=512 * grid_width,
                    mask_image=mask_comp,
                    guidance_scale=50,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]
                image_name = f'flux-fill-test-{self.total_steps if self.total_steps is not None else 0}-{i}-{condition_type}-comp'
                image.save(os.path.join(save_path, f'{image_name}.jpg'))


            pl_module.flux_fill_pipe.transformer.train()

        else: 

            if condition_type == "logo_decomposition":
                grid_width = 2
            
                file_name = [
                    "data/logo_v4_grid_01/test/00000.png",
                    "data/logo_v4_grid_01/test/00001.png",
                    "data/logo_v4_grid_01/test/00002.png",
                    "data/logo_v4_grid_01/test/00003.png",
                    "data/logo_v4_grid_01/test/00004.png",
                    "data/logo_v4_grid_01/test/00005.png",
                    "data/logo_v4_grid_01/test/00006.png",
                    "data/logo_v4_grid_01/test/00007.png",
                    "data/logo_v4_grid_01/test/00008.png",
                    "data/logo_v4_grid_01/test/00009.png",
                ]
                
                test_instruction = []
                for fn in file_name:
                    with open(fn.replace('.png', '.txt'), 'r') as f:
                        instruction = f.read().strip()
                        test_instruction.append(instruction)

            elif condition_type == "logo_product_decomposition":
                grid_width = 3
                
                file_name = [
                    "data/logo_product_4969/test/04500.png",
                    "data/logo_product_4969/test/04501.png",
                    "data/logo_product_4969/test/04502.png",
                    "data/logo_product_4969/test/04503.png",
                    "data/logo_product_4969/test/04504.png",
                    "data/logo_product_4969/test/04505.png",
                    "data/logo_product_4969/test/04506.png",
                    "data/logo_product_4969/test/04507.png",
                    "data/logo_product_4969/test/04508.png",
                    "data/logo_product_4969/test/04509.png",
                    "data/logo_product_4969/test/04510.png",
                    "data/logo_product_4969/test/04511.png",
                    "data/logo_product_4969/test/04512.png",
                    "data/logo_product_4969/test/04513.png",
                    "data/logo_product_4969/test/04514.png",
                    "data/logo_product_4969/test/04515.png",
                    "data/logo_product_4969/test/04516.png",
                    "data/logo_product_4969/test/04517.png",
                    "data/logo_product_4969/test/04518.png",
                    "data/logo_product_4969/test/04519.png",
                ]
                
                test_instruction = []
                for fn in file_name:
                    with open(fn.replace('.png', '.txt'), 'r') as f:
                        instruction = f.read().strip()
                        test_instruction.append(instruction)
            
            elif condition_type == "logo_product_composition":
                grid_width = 3

                file_name = [
                    "data/logo_product_v4.5_grid_001/test/00010.png",
                    "data/logo_product_v4.5_grid_001/test/00011.png",
                    "data/logo_product_v4.5_grid_001/test/00012.png",
                    "data/logo_product_v4.5_grid_001/test/00013.png",
                    "data/logo_product_v4.5_grid_001/test/00014.png",
                    "data/logo_product_v4.5_grid_001/test/00015.png",
                    "data/logo_product_v4.5_grid_001/test/00016.png",
                    "data/logo_product_v4.5_grid_001/test/00017.png",
                    "data/logo_product_v4.5_grid_001/test/00018.png",
                    "data/logo_product_v4.5_grid_001/test/00019.png",
                ]

                test_instruction = []
                for fn in file_name:
                    with open(fn.replace('.png', '.txt'), 'r') as f:
                        instruction = f.read().strip()
                        test_instruction.append(instruction)
            
            elif condition_type == "viton_decomposition":
                grid_width = 3
                
                file_name = [
                    "data/viton_v1.5_011/test/00000.png",
                    "data/viton_v1.5_011/test/00023.png",
                    "data/viton_v1.5_011/test/00042.png",
                    "data/viton_v1.5_011/test/00043.png",
                    "data/viton_v1.5_011/test/00044.png",
                    "data/viton_v1.5_011/test/00103.png",
                    "data/viton_v1.5_011/test/00113.png",
                    "data/viton_v1.5_011/test/00144.png",
                    "data/viton_v1.5_011/test/00155.png",
                    "data/viton_v1.5_011/test/00200.png",
                ]
                
                test_instruction = []
                for fn in file_name:
                    with open(fn.replace('.png', '.txt'), 'r') as f:
                        instruction = f.read().strip()
                        test_instruction.append(instruction)
            
            elif condition_type == "albedo_lighting_decomposition":
                grid_width = 3
                
                file_name = [
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_001_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_002_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_003_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_004_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_005_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_006_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_007_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_008_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_009_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_010_001_cam00_frame0000.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_055_009_cam00_frame0001.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_055_009_cam00_frame0002.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_055_009_cam00_frame0003.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_055_009_cam00_frame0004.png",
                    "/mnt/d/guzheng/Datasets/hypersim/grids/image_albedo_lighting_v2/test/ai_055_009_cam00_frame0005.png"
                ]
                
                test_instruction = []
                for fn in file_name:
                    with open(fn.replace('.png', '.txt'), 'r') as f:
                        instruction = f.read().strip()
                        test_instruction.append(instruction)
            elif condition_type == "foreground_background_decomposition":
                grid_width = 3
                self.condition_size = 512
                
                file_name = [
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000014.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000019.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000029.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000034.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000048.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000059.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000065.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000074.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000083.png",
                        "/mnt/d/guzheng/Experiments/ICEdit/train/data/foreground_background_v1_cycle/test/00000088.png",
                    ]

                test_instruction = []
                for fn in file_name:
                    with open(fn.replace('.png', '_decomp.txt'), 'r') as f:
                        instruction = f.read().strip()
                        test_instruction.append(instruction)
                           
            else:
                raise ValueError(f"Unknown condition type: {condition_type}")

            pl_module.flux_fill_pipe.transformer.eval()
            for i, name in enumerate(file_name):
                combined_image = Image.open(name)

                mask = Image.new('L', combined_image.size, 0)
                draw = ImageDraw.Draw(mask)
                if condition_type.endswith("decomposition"):
                    draw.rectangle([combined_image.size[0] // grid_width, 0, combined_image.size[0], combined_image.size[1]], fill=255)
                elif condition_type.endswith("composition"):
                    draw.rectangle([combined_image.size[0] // grid_width * 2, 0, combined_image.size[0], combined_image.size[1]], fill=255)
                else:
                    raise ValueError(f"Unknown condition type: {condition_type}")
                prompt_ = test_instruction[i]
                    
                image = pl_module.flux_fill_pipe(
                    prompt=prompt_,
                    image=combined_image,
                    height=512,
                    width=512 * grid_width,
                    mask_image=mask,
                    guidance_scale=50,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666)
                ).images[0]

                image_name = f'flux-fill-test-{self.total_steps if self.total_steps is not None else 0}-{i}-{condition_type}'
                image.save(os.path.join(save_path, f'{image_name}.jpg'))


            pl_module.flux_fill_pipe.transformer.train()
        