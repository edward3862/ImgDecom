import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
import torch
from peft import PeftModel, LoraConfig, get_peft_model_state_dict
import os
import prodigyopt

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, encode_images_fill, prepare_text_input


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
            
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)
                
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss


class CycleOminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        cycle_loss_weight: float = 0.01,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.cycle_loss_weight = cycle_loss_weight

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss, losses_dict = self.cycle_step(batch)
        self.losses_dict = losses_dict
        
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss
    
    def get_noise_prediction(self, batch, t, x_0=None, x_1=None):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        prompts = batch["description"]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            if x_0 is None:
                x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            else:
                # todo
                _, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            
            if x_1 is None:
                x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        target = (x_1 - x_0)

        x_0_pred = x_t - t_ * pred

        return pred, target, x_0_pred, x_0, x_1


    def cycle_step(self, batch):

        t = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        
        pred_decomp, target_decomp, x_0_pred_decomp, x_0_decomp, x_1_decomp = self.get_noise_prediction(batch["decomp"], t)
        pred_comp, target_comp, _, _, _ = self.get_noise_prediction(batch["comp"], t, x_0=x_0_decomp, x_1=x_1_decomp)
        decomp_loss = torch.nn.functional.mse_loss(pred_decomp, target_decomp, reduction="mean")
        comp_loss = torch.nn.functional.mse_loss(pred_comp, target_comp, reduction="mean")

        pred_comp_cyc, _, _, _, _ = self.get_noise_prediction(batch["comp"], t, x_0=x_0_pred_decomp, x_1=x_1_decomp)
        cycle_loss = torch.nn.functional.mse_loss(pred_comp_cyc, pred_comp, reduction="mean")
        
        total_loss = (decomp_loss + comp_loss) / 2 + self.cycle_loss_weight * cycle_loss
        
        losses_dict = {
            "total_loss": total_loss,
            "decomp_loss": decomp_loss,
            "comp_loss": comp_loss,
            "cycle_loss": cycle_loss
        }
        self.last_t = t.mean().item()
        
        return total_loss, losses_dict


class CycleOminiModelV1_5(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        cycle_loss_weight: float = 1.0,
        x_0_loss_weight: float = 1.0,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.cycle_loss_weight = cycle_loss_weight
        self.x_0_loss_weight = x_0_loss_weight

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            # raise NotImplementedError        

            print(f"[LoRA] Loading pretrained LoRA from: {lora_path}")
            
            self.flux_fill_pipe.load_lora_weights(lora_path, adapter_name="default")
            self.flux_fill_pipe.set_adapters(["default"])

            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
            
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss, losses_dict = self.cycle_step(batch)
        self.losses_dict = losses_dict
        
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss
    
    def get_noise_prediction(self, batch, t, x_0=None, x_1=None):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        prompts = batch["description"]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            if x_0 is None:
                x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            else:
                # todo
                _, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            
            if x_1 is None:
                x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        target = (x_1 - x_0)

        x_0_pred = x_t - t_ * pred

        return pred, target, x_0_pred, x_0, x_1

    def cycle_step(self, batch):

        t1 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        t2 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        
        pred_decomp, target_decomp, x_0_pred_decomp, x_0_decomp, x_1_decomp = self.get_noise_prediction(batch["decomp"], t1)
        pred_comp, target_comp, x_0_pred_comp, x_0_comp, x_1_comp = self.get_noise_prediction(batch["comp"], t2)

        decomp_loss = torch.nn.functional.mse_loss(pred_decomp, target_decomp, reduction="mean")
        comp_loss = torch.nn.functional.mse_loss(pred_comp, target_comp, reduction="mean")

        pred_comp_cyc, target_comp_cyc, x_0_pred_comp_cyc, x_0_comp_cyc, x_1_comp_cyc = self.get_noise_prediction(batch["comp"], t2, x_0=x_0_pred_decomp, x_1=x_1_comp)
        pred_decomp_cyc, target_decomp_cyc, x_0_pred_decomp_cyc, x_0_decomp_cyc, x_1_decomp_cyc = self.get_noise_prediction(batch["decomp"], t1, x_0=x_0_pred_comp, x_1=x_1_decomp)

        cycle_loss = (torch.nn.functional.mse_loss(pred_decomp, pred_decomp_cyc, reduction="mean") + torch.nn.functional.mse_loss(pred_comp, pred_comp_cyc, reduction="mean")) / 2

        total_loss = (decomp_loss + comp_loss) / 2 + self.cycle_loss_weight * cycle_loss

        losses_dict = {
            "total_loss": total_loss,
            "decomp_loss": decomp_loss,
            "comp_loss": comp_loss,
            "cycle_loss": cycle_loss
        }
        self.last_t = (t1.mean().item() + t2.mean().item()) / 2
        
        return total_loss, losses_dict


class CycleOminiModelV2(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        cycle_loss_weight: float = 1.0,
        x_0_loss_weight: float = 1.0,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.cycle_loss_weight = cycle_loss_weight
        self.x_0_loss_weight = x_0_loss_weight

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss, losses_dict = self.cycle_step(batch)
        self.losses_dict = losses_dict
        
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss
    
    def get_noise_prediction(self, batch, t, x_0=None, x_1=None):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        prompts = batch["description"]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            if x_0 is None:
                x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            else:
                # todo
                _, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            
            if x_1 is None:
                x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        target = (x_1 - x_0)

        x_0_pred = x_t - t_ * pred

        return pred, target, x_0_pred, x_0, x_1

    def cycle_step(self, batch):

        t1 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        t2 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        
        pred_decomp, target_decomp, x_0_pred_decomp, x_0_decomp, x_1_decomp = self.get_noise_prediction(batch["decomp"], t1)
        pred_comp, target_comp, x_0_pred_comp, x_0_comp, x_1_comp = self.get_noise_prediction(batch["comp"], t2)

        decomp_loss = torch.nn.functional.mse_loss(pred_decomp, target_decomp, reduction="mean")
        comp_loss = torch.nn.functional.mse_loss(pred_comp, target_comp, reduction="mean")

        pred_comp_cyc, target_comp_cyc, x_0_pred_comp_cyc, x_0_comp_cyc, x_1_comp_cyc = self.get_noise_prediction(batch["comp"], t2, x_0=x_0_pred_decomp, x_1=x_1_comp)
        pred_decomp_cyc, target_decomp_cyc, x_0_pred_decomp_cyc, x_0_decomp_cyc, x_1_decomp_cyc = self.get_noise_prediction(batch["decomp"], t1, x_0=x_0_pred_comp, x_1=x_1_decomp)

        x_0_loss = torch.nn.functional.mse_loss(x_0_pred_decomp, x_0_pred_comp, reduction="mean")
        cycle_loss = (torch.nn.functional.mse_loss(pred_decomp, pred_decomp_cyc, reduction="mean") + torch.nn.functional.mse_loss(pred_comp, pred_comp_cyc, reduction="mean")) / 2

        total_loss = (decomp_loss + comp_loss) / 2 + self.cycle_loss_weight * cycle_loss + self.x_0_loss_weight * x_0_loss

        losses_dict = {
            "total_loss": total_loss,
            "decomp_loss": decomp_loss,
            "comp_loss": comp_loss,
            "x_0_loss": x_0_loss,
            "cycle_loss": cycle_loss
        }
        self.last_t = (t1.mean().item() + t2.mean().item()) / 2
        
        return total_loss, losses_dict


class CycleOminiModelV3(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        cycle_loss_weight: float = 1.0,
        x_0_loss_weight: float = 1.0,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.cycle_loss_weight = cycle_loss_weight
        self.x_0_loss_weight = x_0_loss_weight

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss, losses_dict = self.cycle_step(batch)
        self.losses_dict = losses_dict
        
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss
    
    def get_noise_prediction(self, batch, t, x_0=None, x_1=None):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        prompts = batch["description"]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            if x_0 is None:
                x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            else:
                # todo
                _, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            
            if x_1 is None:
                x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        target = (x_1 - x_0)

        x_0_pred = x_t - t_ * pred

        return pred, target, x_0_pred, x_0, x_1

    def cycle_step(self, batch):

        t1 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        t2 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        
        pred_decomp, target_decomp, x_0_pred_decomp, x_0_decomp, x_1_decomp = self.get_noise_prediction(batch["decomp"], t1)
        pred_comp, target_comp, x_0_pred_comp, x_0_comp, x_1_comp = self.get_noise_prediction(batch["comp"], t2)

        decomp_loss = torch.nn.functional.mse_loss(pred_decomp, target_decomp, reduction="mean")
        comp_loss = torch.nn.functional.mse_loss(pred_comp, target_comp, reduction="mean")

        pred_comp_cyc, target_comp_cyc, x_0_pred_comp_cyc, x_0_comp_cyc, x_1_comp_cyc = self.get_noise_prediction(batch["comp"], t2, x_0=x_0_pred_decomp, x_1=x_1_comp)
        pred_decomp_cyc, target_decomp_cyc, x_0_pred_decomp_cyc, x_0_decomp_cyc, x_1_decomp_cyc = self.get_noise_prediction(batch["decomp"], t1, x_0=x_0_pred_comp, x_1=x_1_decomp)

        cycle_loss = (torch.nn.functional.mse_loss(pred_decomp_cyc, target_decomp, reduction="mean") + torch.nn.functional.mse_loss(pred_comp_cyc, target_comp, reduction="mean")) / 2

        total_loss = (decomp_loss + comp_loss) / 2 + self.cycle_loss_weight * cycle_loss

        losses_dict = {
            "total_loss": total_loss,
            "decomp_loss": decomp_loss,
            "comp_loss": comp_loss,
            "cycle_loss": cycle_loss
        }
        self.last_t = (t1.mean().item() + t2.mean().item()) / 2
        
        return total_loss, losses_dict
    

class CycleOminiModelV3_Decomp(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        cycle_loss_weight: float = 1.0,
        x_0_loss_weight: float = 1.0,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.cycle_loss_weight = cycle_loss_weight
        self.x_0_loss_weight = x_0_loss_weight

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss, losses_dict = self.cycle_step(batch)
        self.losses_dict = losses_dict
        
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss
    
    def get_noise_prediction(self, batch, t, x_0=None, x_1=None):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        prompts = batch["description"]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            if x_0 is None:
                x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            else:
                # todo
                _, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            
            if x_1 is None:
                x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        target = (x_1 - x_0)

        x_0_pred = x_t - t_ * pred

        return pred, target, x_0_pred, x_0, x_1

    def cycle_step(self, batch):

        t1 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        pred_decomp, target_decomp, x_0_pred_decomp, x_0_decomp, x_1_decomp = self.get_noise_prediction(batch["decomp"], t1)
        decomp_loss = torch.nn.functional.mse_loss(pred_decomp, target_decomp, reduction="mean")

        total_loss = decomp_loss
        losses_dict = {
            "total_loss": total_loss,
            "decomp_loss": decomp_loss,
        }
        self.last_t = t1.mean().item()
        
        return total_loss, losses_dict
    

class CycleOminiModelV3_Comp(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        cycle_loss_weight: float = 1.0,
        x_0_loss_weight: float = 1.0,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.cycle_loss_weight = cycle_loss_weight
        self.x_0_loss_weight = x_0_loss_weight

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss, losses_dict = self.cycle_step(batch)
        self.losses_dict = losses_dict
        
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss
    
    def get_noise_prediction(self, batch, t, x_0=None, x_1=None):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        prompts = batch["description"]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            if x_0 is None:
                x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            else:
                # todo
                _, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)
            
            if x_1 is None:
                x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        target = (x_1 - x_0)

        x_0_pred = x_t - t_ * pred

        return pred, target, x_0_pred, x_0, x_1

    def cycle_step(self, batch):

        t2 = torch.sigmoid(torch.randn((batch["decomp"]["image"].shape[0],), device=self.device))
        pred_comp, target_comp, x_0_pred_comp, x_0_comp, x_1_comp = self.get_noise_prediction(batch["comp"], t2)
        comp_loss = torch.nn.functional.mse_loss(pred_comp, target_comp, reduction="mean")
        total_loss = comp_loss

        losses_dict = {
            "total_loss": total_loss,
            "comp_loss": comp_loss,
        }
        self.last_t = t2.mean().item()
        
        return total_loss, losses_dict