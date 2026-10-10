from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import lightning as L
import yaml
import os
import random
import time
import numpy as np
import re
from datasets import load_dataset

from .data import (
    EditDataset,
    FGBGDataset,
    OminiDataset,
    EditDataset_with_Omini,
    LogoDataset,
    LogoProductDataset,
    CycleDataset
)
from .model import CycleOminiModelV3_Comp, CycleOminiModelV3_Decomp, OminiModel, CycleOminiModel, CycleOminiModelV1_5, CycleOminiModelV2, CycleOminiModelV3
from .callbacks import TrainingCallback


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)
        
    if 'use_offset_noise' not in config.keys():
        config['use_offset_noise'] = False

    # Initialize dataset and dataloader
    
    if training_config["dataset"]["type"] == "edit":
        dataset = load_dataset('osunlp/MagicBrush')
        dataset = EditDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "omini":
        dataset = load_dataset(training_config["dataset"]["path"])
        dataset = OminiDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )

    elif training_config["dataset"]["type"] == "edit_with_omini":
        omni = load_dataset("parquet", data_files=os.path.abspath(training_config["dataset"]["path"]), split="train")
        magic = load_dataset('osunlp/MagicBrush')
        dataset = EditDataset_with_Omini(
            magic,
            omni,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "logo":
        dataset = LogoDataset(
            dataset_folder=training_config["dataset"]["path"],
            split=training_config["dataset"].get("split", "train"),
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "logo_product":
        dataset = LogoProductDataset(
            dataset_folder=training_config["dataset"]["path"],
            condition_type=training_config["condition_type"],
            split=training_config["dataset"].get("split", "train"),
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "fgbg":
        dataset = FGBGDataset(
            dataset_folder=training_config["dataset"]["path"],
            condition_type=training_config["condition_type"],
            split=training_config["dataset"].get("split", "train"),
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "cycle":
        dataset = CycleDataset(
            dataset_folder=training_config["dataset"]["path"],
            condition_type=training_config["condition_type"],
            split=training_config["dataset"].get("split", "train"),
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    else:
        raise ValueError(f"Unknown dataset type: {training_config['dataset']['type']}")
          
    print("Dataset length:", len(dataset))
    max_round = training_config["dataset"].get("round_num", -1)
    if max_round > 1:
        round_numbers = []
        for path in dataset.image_paths:
            match = re.search(r"round(\d+)_", os.path.basename(path))
            if match:
                round_num = int(match.group(1))
            else:
                round_num = -1
            round_numbers.append(round_num)

        print(max_round)
        print(max(round_numbers))
        print(len(dataset))
        print(len(round_numbers))

        assert max_round == max(round_numbers)
        assert len(dataset) == len(round_numbers)
        weights = [2. if r == max_round else 1.0 for r in round_numbers]
        print(weights)
        weights = torch.DoubleTensor(weights)

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True
        )

        train_loader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            sampler=sampler,
            num_workers=training_config["dataloader_workers"],
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["dataloader_workers"],
        )

    # Initialize model
    if training_config["dataset"]["type"] == "cycle":
        if config["model"]["model_name"] == "cycle_v1_5":
            trainable_model = CycleOminiModelV1_5(
                flux_fill_id=config["flux_path"],
                lora_path=training_config.get("lora_path", None),
                lora_config=training_config["lora_config"],
                device=f"cuda",
                dtype=getattr(torch, config["dtype"]),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                use_offset_noise=config["use_offset_noise"],
                cycle_loss_weight=training_config.get("cycle_loss_weight", 1.0),
            )
        elif config["model"]["model_name"] == "cycle_v2":
            trainable_model = CycleOminiModelV2(
                flux_fill_id=config["flux_path"],
                lora_config=training_config["lora_config"],
                device=f"cuda",
                dtype=getattr(torch, config["dtype"]),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                use_offset_noise=config["use_offset_noise"],
                cycle_loss_weight=training_config.get("cycle_loss_weight", 1.0),
                x_0_loss_weight=training_config.get("x_0_loss_weight", 1.0)
            )
        elif config["model"]["model_name"] == "cycle_v3":
            trainable_model = CycleOminiModelV3(
                flux_fill_id=config["flux_path"],
                lora_path=training_config.get("lora_path", None),
                lora_config=training_config["lora_config"],
                device=f"cuda",
                dtype=getattr(torch, config["dtype"]),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                use_offset_noise=config["use_offset_noise"],
                cycle_loss_weight=training_config.get("cycle_loss_weight", 1.0),
            )
        elif config["model"]["model_name"] == "cycle_v3_decomp":
            trainable_model = CycleOminiModelV3_Decomp(
                flux_fill_id=config["flux_path"],
                lora_config=training_config["lora_config"],
                device=f"cuda",
                dtype=getattr(torch, config["dtype"]),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                use_offset_noise=config["use_offset_noise"],
                cycle_loss_weight=training_config.get("cycle_loss_weight", 1.0),
            )
        elif config["model"]["model_name"] == "cycle_v3_comp":
            trainable_model = CycleOminiModelV3_Comp(
                flux_fill_id=config["flux_path"],
                lora_config=training_config["lora_config"],
                device=f"cuda",
                dtype=getattr(torch, config["dtype"]),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                use_offset_noise=config["use_offset_noise"],
                cycle_loss_weight=training_config.get("cycle_loss_weight", 1.0),
            )
        else:
            trainable_model = CycleOminiModel(
                flux_fill_id=config["flux_path"],
                lora_config=training_config["lora_config"],
                device=f"cuda",
                dtype=getattr(torch, config["dtype"]),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                use_offset_noise=config["use_offset_noise"],
                cycle_loss_weight=training_config.get("cycle_loss_weight", 1.0)
            )
    else:
        trainable_model = OminiModel(
            flux_fill_id=config["flux_path"],
            lora_config=training_config["lora_config"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=config.get("model", {}),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            use_offset_noise=config["use_offset_noise"],
        )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    ckpt_path = training_config.get("ckpt_path", None)
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
        trainer.fit(trainable_model, train_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
