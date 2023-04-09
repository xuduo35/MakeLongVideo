import os
import sys
import random
import argparse
from makelongvideo.pipelines.pipeline_makelongvideo import MakeLongVideoPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from makelongvideo.models.unet import UNet3DConditionModel
from makelongvideo.util import save_videos_grid, ddim_inversion
import torch
import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
import torch.nn.functional as F

def randstr(l=16):
  s =''
  chars ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'

  for i in range(l):
    s += chars[random.randint(0, len(chars)-1)]

  return  s

parser = argparse.ArgumentParser(description='Make Long Video')
parser.add_argument('--prompt', type=str, default=None, required=True, help='prompt')
parser.add_argument('--negprompt', type=str, default=None, help='negtive prompt')
parser.add_argument('--speed', type=int, default=None, help='playback speed')
parser.add_argument('--inv_latent_path', type=str, default=None, help='inversion latent path')
parser.add_argument('--sample_video_path', type=str, default=None, help='sample video path')
parser.add_argument('--guidance_scale', type=float, default=12.5, help='guidance scale')
parser.add_argument('--save', action='store_true', default=False, help='save parameters')
parser.add_argument('--width', type=int, default=512, help='width')
parser.add_argument('--height', type=int, default=512, help='height')

args = parser.parse_args()

pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
my_model_path = "./outputs/makelongvideo"

unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
state_dict = unet.state_dict()

#print(state_dict['up_blocks.2.attentions.0.transformer_blocks.0.temporal_rel_pos_bias.net.2.weight'])
#print(state_dict['up_blocks.2.attentions.2.transformer_blocks.0.attn_temp.to_q.weight'])

if args.save:
    print(state_dict)
    sys.exit(0)

pipeline = MakeLongVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_vae_slicing()

ddim_inv_latent = None

if args.sample_video_path is not None:
    noise_scheduler = DDPMScheduler.from_pretrained(my_model_path, subfolder="scheduler")

    ddim_inv_scheduler = DDIMScheduler.from_pretrained(my_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(50)

    sample_start_idx = 0
    sample_frame_rate = 2
    n_sample_frames = 24
    vr = decord.VideoReader(args.sample_video_path, width=args.width, height=args.height)
    framelst = list(range(sample_start_idx, len(vr), sample_frame_rate))
    sample_index = framelst[0:n_sample_frames]
    video = vr.get_batch(sample_index)
    pixel_values = rearrange(video, "(b f) h w c -> b f c h w", f=n_sample_frames) / 127.5 - 1.0
    b, f, c, h, w = pixel_values.shape
    video_length = pixel_values.shape[1]
    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    ###
    #pixel_values = F.interpolate(pixel_values, size=(32,32))
    #pixel_values = F.interpolate(pixel_values, size=(h,w))
    ###
    pixel_values = pixel_values.to('cuda', dtype=torch.float16)
    with torch.no_grad():
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    '''
    ###
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each video
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    latents = noise_scheduler.add_noise(latents, noise, timesteps)
    ###
    '''
    ddim_inv_latent = ddim_inversion(
        pipeline, ddim_inv_scheduler, video_latent=latents, num_inv_steps=50, prompt=""
        )[-1].to(torch.float16)
elif args.inv_latent_path is not None:
    ddim_inv_latent = torch.load(args.inv_latent_path).to(torch.float16)
#else:
elif False:
    ddim_inv_latent = torch.randn([1, 4, 24, 64, 64]).to(torch.float16)
    #ddim_inv_latent = torch.randn([1, 4, 1, 64, 64]).repeat_interleave(24,dim=2)

prompt = "{} ...{}x".format(args.prompt, args.speed) if args.speed is not None else args.prompt

print('prompt:', prompt)

video = pipeline(prompt, latents=ddim_inv_latent, video_length=24, height=args.height, width=args.width, num_inference_steps=50, guidance_scale=args.guidance_scale, negative_prompt=args.negprompt).videos

if not os.path.exists("./outputs/results"):
    os.mkdir("./outputs/results")

fps = 24//args.speed if args.speed is not None else 12

if args.speed is not None:
    resultfile = f"./outputs/results/{args.prompt[:16]}-{randstr(6)}-{args.speed}x.gif"
else:
    resultfile = f"./outputs/results/{args.prompt[:16]}-{randstr(6)}.gif"

save_videos_grid(video, resultfile, fps=fps)
