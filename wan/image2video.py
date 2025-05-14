# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import torchvision.io as io

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            21,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 80, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)

            # ... existing code ...
            for i, t in enumerate(tqdm(timesteps)):
                


                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

 
                if i == sampling_steps - 1:
           
                    step_index = sample_scheduler._step_index
                    lower_order_nums = sample_scheduler.lower_order_nums


                    def generate_latent(scale, latent):
      
                        sample_scheduler._step_index = step_index
                        sample_scheduler.lower_order_nums = lower_order_nums


                        noise_pred_cond = self.model(
                            latent_model_input, t=timestep, **arg_c)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred = noise_pred_uncond + scale * (
                            noise_pred_cond - noise_pred_uncond)

                        latent = latent.to(
                            torch.device('cpu') if offload_model else self.device)

                        
                        temp_x = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latent.unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        return temp_x.squeeze(0)

                    latent_x1 = generate_latent(guide_scale, latent)
                    latent_x2 = generate_latent(guide_scale+20, latent)


                    test_video = self.vae.decode([latent_x1])



                    cache_video(
                        tensor=test_video[0][None],
                        save_file='video_cover/i2v.mp4',
                        fps=16,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
                    my_video, _, _ = io.read_video('video_cover/i2v.mp4', pts_unit='pts')

                    my_video = my_video.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()
                    my_video = (my_video / 127.5) - 1  
                    
                    




                    test_rev = self.vae.encode(my_video)


                    if isinstance(test_rev, list):
                        test_rev = test_rev[0]

                   
                    mask1 = torch.abs(latent_x1 - test_rev) < 0.2
                    mask2 = torch.abs(latent_x1 - latent_x2) > 0.2
                    mask = mask1 * mask2

                    print(mask1.sum().item())
                    print(mask2.sum().item())

                    n = mask.sum().item()

                    print(n)

  
                    message = torch.randint(0, 2, (n,),  device=self.device)


                    save_dir = "message"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(message, os.path.join(save_dir, "message.pt"))


                    location = torch.zeros_like(latent_x1, dtype=torch.long)


                    indices = torch.nonzero(mask, as_tuple=True)


                    location[indices] = message

                    mask_flat = mask.flatten()
                    location_flat = location.flatten()
                    for i in range(1, len(mask_flat)):
                        if not mask_flat[i]:
                            location_flat[i] = location_flat[i - 1]

                    location = location_flat.view(latent_x1.shape)


                    latent_mod = latent_x1 - (latent_x2 - latent_x1)


                    



                    latents = torch.where(location == 0, latent_mod, latent_x2)
                    x0 = [latents.to(self.device)]
                    del latent_model_input, timestep

                    
                    break  
                
                

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep


                

            
           
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            videos = self.vae.decode(x0)
        
       

        del noise, latents, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        cache_video(
            tensor=videos[0][None],
            save_file='video_steg/i2v.mp4',
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        steg_video, _, _ = io.read_video('video_steg/i2v.mp4', pts_unit='pts')

        steg_video = steg_video.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()
        steg_video = (steg_video / 127.5) - 1
        
                    




        x_prime = self.vae.encode(steg_video)



        if isinstance(x_prime, list):
            x_prime = x_prime[0]

        dist_to_x1 = torch.abs(x_prime - latent_mod)
        dist_to_x2 = torch.abs(x_prime - latent_x2)

        recovered_message_flat = torch.where(dist_to_x1 < dist_to_x2, 0, 1)


        message_recovered = recovered_message_flat[mask]
        return message_recovered


        