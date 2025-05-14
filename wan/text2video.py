import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import numpy as np
import cv2

import tempfile
import subprocess

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torchvision.io as io
from wan.utils.utils import cache_video
import datetime



def add_gaussian_noise(img, std=0.05):
    noise = np.random.normal(0, std * 127, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def add_pepper_salt_noise(img, prob=0.01):
    output = np.copy(img)
    probs = np.random.rand(*img.shape[:2])
    output[probs < (prob / 2)] = 0      # pepper
    output[probs > 1 - (prob / 2)] = 255  # salt
    return output

def adjust_brightness(img, delta=0.1):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + delta), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)





class WanT2V:

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
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

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
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
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
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

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
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 val_mask1=0.32,
                 val_mask2=0.98,
                 add_cfg=16
                 ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

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

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

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
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # ... existing code ...
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)

                # message embedding in the last step
                if i == sampling_steps - 1:
                    # save the current state
                    step_index = sample_scheduler._step_index
                    lower_order_nums = sample_scheduler.lower_order_nums


                    def generate_latent(scale):
                        # revert to the saved state
                        sample_scheduler._step_index = step_index
                        sample_scheduler.lower_order_nums = lower_order_nums

                        noise_pred_cond = self.model(
                            latent_model_input, t=timestep, **arg_c)[0]
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0]
                        noise_pred = noise_pred_uncond + scale * (
                            noise_pred_cond - noise_pred_uncond)
                        temp_x = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        return temp_x.squeeze(0)

                    latent_x1 = generate_latent(guide_scale)
                    latent_x2 = generate_latent(guide_scale+add_cfg)


                    test_video = self.vae.decode([latent_x1])



                    current_time = datetime.datetime.now()
                    time_str = current_time.strftime("%m_%d_%H_%M")
                    #name the video by input_prompt
                    first_word = input_prompt.split()[0] if input_prompt else ""
                    
                    video_file_name = f'video_cover/{first_word}_{time_str}.mp4'

                    cache_video(
                        tensor=test_video[0][None],
                        save_file=video_file_name,
                        fps=16,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))

                    

                    my_video, _, _ = io.read_video(video_file_name, pts_unit='pts')

                    # # ======
                    # my_video = my_video.numpy()  # Tensor -> NumPy
                    # if my_video.max() <= 1:
                    #     my_video = (my_video * 255).astype(np.uint8)

                    # perturbed_frames = []
                    # for frame in my_video:
                    #     # frame = add_gaussian_noise(frame, std=0.05)
                    #     # frame = add_pepper_salt_noise(frame, prob=0.05)
                    #     # frame = adjust_brightness(frame, delta=0.1)
                    #     # frame = adjust_contrast(frame, contrast_factor=0.5)
                    #     perturbed_frames.append(frame)

                    # my_video = torch.from_numpy(np.stack(perturbed_frames))  # back to Tensor
                    # # ======




                    my_video = my_video.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()  #[C, T, H, W]
                    #
                    my_video = (my_video / 127.5) - 1  # (-1, 1)
                    
                    




                    test_rev = self.vae.encode(my_video)
                    

                    if isinstance(test_rev, list):
                        test_rev = test_rev[0]

                   
                    abs_diff1 = torch.abs(latent_x1 - test_rev)
                    # val_mask1 = 0.32(\tau_1)
                    threshold1 = torch.quantile(abs_diff1.flatten(), val_mask1)
                    mask1 = abs_diff1 < threshold1



                    abs_diff = torch.abs(latent_x1 - latent_x2)
                    # val_mask2 = 0.98(\tau_2)
                    threshold = torch.quantile(abs_diff.flatten(), val_mask2)
                    mask2 = abs_diff > threshold
                    mask = mask1 * mask2

                    print(mask1.sum().item())
                    print(mask2.sum().item())


                    n = mask.sum().item()

                    print(n)
                    # torch.manual_seed(99)

                    message = torch.randint(0, 2, (n,),  device=self.device)

                    # save message
                    save_dir = "message"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(message, os.path.join(save_dir, "message.pt"))

                    # Matrix H
                    location = torch.zeros_like(latent_x1, dtype=torch.long)

                    indices = torch.nonzero(mask, as_tuple=True)

                    location[indices] = message

                
                    mask_inverse = ~mask
                    location[mask_inverse] = 2


                    latent_mod = latent_x1 - (latent_x2 - latent_x1)


                    # stego latents
                    latents = torch.where(location == 0, latent_mod, 
                                        torch.where(location == 2, latent_x1, latent_x2))
                    latents = [latents]


                    break  
                
                

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                
            
            

            x0 = latents

            

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)
        
        

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0], time_str, first_word


 



    def receive(self, video_path, input_prompt, size=(1280, 720), frame_num=81, shift=5.0,
                sample_solver='unipc', sampling_steps=50, guide_scale=5.0, n_prompt="", seed=-1, offload_model=True, val_mask1=0.32,
                 val_mask2=0.98,
                 add_cfg=16):
   
        
        video, _, _ = io.read_video(video_path, pts_unit='pts')


        # # ======
                    # my_video = my_video.numpy()  # Tensor -> NumPy
                    # if my_video.max() <= 1:
                    #     my_video = (my_video * 255).astype(np.uint8)

                    # perturbed_frames = []
                    # for frame in my_video:
                    #     # frame = add_gaussian_noise(frame, std=0.05)
                    #     # frame = add_pepper_salt_noise(frame, prob=0.05)
                    #     # frame = adjust_brightness(frame, delta=0.1)
                    #     # frame = adjust_contrast(frame, contrast_factor=0.5)
                    #     perturbed_frames.append(frame)

                    # my_video = torch.from_numpy(np.stack(perturbed_frames))  # back to Tensor
                    # # ======

        video = video.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()
        video = (video / 127.5) - 1  #(-1, 1)
        
        x_prime = self.vae.encode(video)

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

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

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

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
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

           # ... existing code ...
            for i, t in enumerate(tqdm(timesteps)):

                

                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)


                if i == sampling_steps - 1:

                    step_index = sample_scheduler._step_index
                    lower_order_nums = sample_scheduler.lower_order_nums


                    def generate_latent(scale):

                        sample_scheduler._step_index = step_index
                        sample_scheduler.lower_order_nums = lower_order_nums

                        noise_pred_cond = self.model(
                            latent_model_input, t=timestep, **arg_c)[0]
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0]
                        noise_pred = noise_pred_uncond + scale * (
                            noise_pred_cond - noise_pred_uncond)
                        temp_x = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        return temp_x.squeeze(0)

                    latent_x1 = generate_latent(guide_scale)
                    latent_x2 = generate_latent(guide_scale+add_cfg)
                    break 

                
                    
                
                

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]





        if isinstance(x_prime, list):
            x_prime = x_prime[0]

 
        my_video, _, _ = io.read_video('video_cover/test.mp4', pts_unit='pts')

        my_video = my_video.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()
        my_video = (my_video / 127.5) - 1  # (-1, 1)
                    
                    


        test_rev = self.vae.encode(my_video)


                    
                    

        if isinstance(test_rev, list):
            test_rev = test_rev[0]

                   
        abs_diff1 = torch.abs(latent_x1 - test_rev)
        # val_mask1 = 0.32(\tau_1)
        threshold1 = torch.quantile(abs_diff1.flatten(), val_mask1)
        mask1 = abs_diff1 < threshold1



        abs_diff = torch.abs(latent_x1 - latent_x2)
        # val_mask2 = 0.98(1-\tau_2)
        threshold = torch.quantile(abs_diff.flatten(), val_mask2)
        mask2 = abs_diff > threshold
        mask = mask1 * mask2

        latent_mod = latent_x1 - (latent_x2 - latent_x1)


       
        dist_to_x1 = torch.abs(x_prime - latent_mod)
        dist_to_x2 = torch.abs(x_prime - latent_x2)


        recovered_message_flat = torch.where(dist_to_x1 < dist_to_x2, 0, 1)


        message_recovered = recovered_message_flat[mask]            


        
       
        # save_path = 'output.txt'
        # with open(save_path, 'w') as f:
        #     # x_prime[0,0,:,:]
        #     for row in x_prime[0, 0].cpu().numpy():
        #         line = ' '.join(map(str, row))
        #         f.write(line + '\n')
        #     f.write('\n') 

        #     # latent_x1[0,0,:,:]
        #     for row in latent_x1[0, 0].cpu().numpy():
        #         line = ' '.join(map(str, row))
        #         f.write(line + '\n')
        #     f.write('\n') 

        #     # latent_x2[0,0,:,:]
        #     for row in latent_x2[0, 0].cpu().numpy():
        #         line = ' '.join(map(str, row))
        #         f.write(line + '\n')


        return message_recovered