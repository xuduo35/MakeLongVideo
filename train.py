import argparse
import datetime
import logging
import inspect
import math
import random
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from makelongvideo.models.unet import UNet3DConditionModel
from makelongvideo.data.dataset import MakeLongVideoDataset
from makelongvideo.pipelines.pipeline_makelongvideo import MakeLongVideoPipeline
from makelongvideo.util import save_videos_grid, ddim_inversion
from einops import rearrange

from torch.cuda.amp import autocast

from accelerate import DistributedDataParallelKwargs

import torch.nn.functional as F

import webdataset as wds
from functools import partial

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

#logger = get_logger(__name__, log_level="INFO")
logger = get_logger(__name__)

def identity(x):
    return x

def dotokenize(prompt, tokenizer):
    if random.random() < 0.5:
        prompt = "{} ...0x".format(prompt)

    return tokenizer(
                    prompt,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                    ).input_ids[0]

class Laion400MImageLoader:
    def __init__(self, dataset, width=256, height=256, num_workers=8, batch_size=96):
        self.loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=batch_size)
        self.iter = iter(self.loader)
        self.width = width
        self.height = height

    def getnext(self):
        try:
            sample = next(self.iter)
        except:
            self.iter = iter(self.loader)
            sample = next(self.iter)

        images = sample[0]*2.-1.
        prompt_ids = sample[1]

        images = rearrange(images, "(b f) h w c -> b (f c) h w", f=1)
        images = F.interpolate(images, size=(self.height,self.width))
        images = rearrange(images, "b (f c) h w -> b f c h w", f=1)
        #images = rearrange(images, "(b f) h w c -> b f c h w", f=1)

        example = {
            "pixel_values": images,
            "prompt_ids": prompt_ids
        }

        return example

# adapt offset noise from https://github.com/ExponentialML/Text-To-Video-Finetuning
# theory: https://www.crosslabs.org/blog/diffusion-with-offset-noise

def sample_noise(latents, noise_strength, use_offset_noise):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "temporal_conv",
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
        "rel_pos_bias",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    train_image: Optional[Dict] = None,
    use_offset_noise: bool = False,
    offset_noise_strength: float = 0.1,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    logging_dir = Path(output_dir, "0", args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        #kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)],
    )

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    if resume_from_checkpoint is not None or args.unwrap is not None:
        ckptpath = resume_from_checkpoint
        if args.unwrap is not None:
            ckptpath = args.unwrap
        ckpt = torch.load(os.path.join(ckptpath, 'pytorch_model.bin'), map_location='cpu')
        unet.load_state_dict(ckpt)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    highlr_params = []
    lowlr_params =[]

    unet.requires_grad_(False)

    for name, param in unet.named_parameters():
        found = False
        for m in trainable_modules:
            if name.find(m) > 0:
                found = True
                break
        if found:
            highlr_params.append(param)
            param.requires_grad = True
        #else:
        elif False:
            lowlr_params.append(param)
            param.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        #unet.parameters(),
        [{
            "params": highlr_params, 
            "lr": learning_rate,
        }, {
            "params": lowlr_params,
            "lr": learning_rate*0.001
        }],
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the training dataset
    train_dataset = MakeLongVideoDataset(**train_data, tokenizer=tokenizer)

    # Preprocessing the dataset
    #train_dataset.prompt_ids = tokenizer(
    #    train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, num_workers=8
    )

    train_list = ["video"]

    if train_image is not None:
        train_list.append("image")

        tokenizeit = partial(dotokenize, tokenizer=tokenizer)

        image_dataset = wds.WebDataset(
                train_image['image_files'], nodesplitter=wds.split_by_node
                ).shuffle(1000).decode("rgb").to_tuple("jpg", "txt").map_tuple(identity, tokenizeit)
        image_loader = Laion400MImageLoader(
                image_dataset,
                width=train_image['width'],
                height=train_image['height'],
                batch_size=train_image['train_batch_size']
                )

    # Get the validation pipeline
    validation_pipeline = MakeLongVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if train_image is None:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, image_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, image_loader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint is not None or args.unwrap is not None:
        if args.unwrap is not None:
            resume_from_checkpoint = args.unwrap

        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        #accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

        if args.unwrap:
            unet = accelerator.unwrap_model(unet)
            pipeline = MakeLongVideoPipeline.from_pretrained(
                pretrained_model_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
            )
            pipeline.save_pretrained(output_dir)

            sys.exit(0)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    #torch.autograd.set_detect_anomaly(True)
    #with autocast():
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        train_loss_img = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
              for loop in train_list:
                if loop == "image":
                    batch = image_loader.getnext()
                    batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    batch["prompt_ids"] = batch["prompt_ids"].to(accelerator.device)

                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                #b, f, c, h, w = pixel_values.shape
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                #if pixel_values.shape[-1] > 256 and random.random() < 0.3:
                #    pixel_values = F.interpolate(pixel_values, (256,256))
                latents = vae.encode(pixel_values).latent_dist.sample()

                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                #pixel_values = F.interpolate(pixel_values, size=(32,32))
                #pixel_values = F.interpolate(pixel_values, size=(h,w))
                #blurred_latents = vae.encode(pixel_values).latent_dist.sample()
                #blurred_latents = rearrange(blurred_latents, "(b f) c h w -> b c f h w", f=video_length)
                #blurred_latents = blurred_latents * 0.18215

                # Sample noise that we'll add to the latents
                #noise = torch.randn_like(latents)
                noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                #noisy_latents = noise_scheduler.add_noise(blurred_latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                #with accelerator.autocast():
                if True:
                    #model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()

                if loop == "video":
                    train_loss += avg_loss.item() / gradient_accumulation_steps
                else:
                    train_loss_img += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                #with torch.autograd.detect_anomaly():
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                '''
                print("\n")
                for name, p in unet.named_parameters():
                    if p.requires_grad:
                        print(name, p.grad)
                print("\n")
                '''
                optimizer.zero_grad() 

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                #print("\n", unet.state_dict()['up_blocks.2.attentions.2.transformer_blocks.0.attn_temp.to_q.weight'], "\n")
                #print("\n", unet.state_dict()['conv_in.spatial_conv.weight'], "\n")
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "train_loss_img": train_loss_img}, step=global_step)
                train_loss = 0.0
                train_loss_img = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                  with accelerator.autocast():
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)

                        ddim_inv_latent = None

                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")

                            validh,validw = (validation_data.height, validation_data.width)
                            latents = vae.encode(
                                F.interpolate(pixel_values[:video_length,:,:,:], (validh, validw))
                                ).latent_dist.sample()
                            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                            latents = latents * 0.18215

                            ddim_inv_latent = ddim_inversion(
                                validation_pipeline, ddim_inv_scheduler, video_latent=latents[:1,:,:,:,:],
                                num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)

                            torch.save(ddim_inv_latent, inv_latents_path)

                        for idx, prompt in enumerate(validation_data.prompts):
                            sample = validation_pipeline(
                                    prompt, generator=generator, latents=ddim_inv_latent, **validation_data
                                    ).videos
                            save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                            samples.append(sample)
                        #samples = torch.concat(samples)
                        samples = torch.cat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                        save_videos_grid(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = MakeLongVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unwrap", type=str, default=None)
    parser.add_argument("--config", type=str, default="./configs/makelongvideo.yaml")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
