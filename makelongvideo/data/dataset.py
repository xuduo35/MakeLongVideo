import os
import decord
decord.bridge.set_bridge('torch')
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from einops import rearrange

import random

class MakeLongVideoDataset(Dataset):
    def __init__(
            self,
            video_dir: str,
            train_list: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            sample_frame_rates: str = None,
            tokenizer=None
    ):
        self.video_dir = video_dir
        f = open(os.path.join(self.video_dir, train_list))
        self.videolist = [line.strip() for line in f.readlines()]
        random.shuffle(self.videolist)
        f.close()
        self.tokenizer = tokenizer

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rates = [sample_frame_rate]

        if sample_frame_rates is not None:
            self.sample_frame_rates = [int(n) for n in sample_frame_rates.split(',')]

    def __len__(self):
        return len(self.videolist)

    def getvr(self, index, sample_frame_rate):
        vr = None

        line = self.videolist[index]
        idx = line.find(' ')

        if idx > 0:
            video_path = os.path.join(self.video_dir, line[0:idx])
            prompt = line[idx+1:]
        else:
            video_path = os.path.join(self.video_dir, line)
            prompt = ""

        sample_frame_rates = self.sample_frame_rates
        sample_frame_rate = sample_frame_rates[random.randint(0,len(sample_frame_rates)-1)]

        # load and sample video frames
        try:
            #vr = decord.VideoReader(video_path, width=self.width, height=self.height)
            vr = decord.VideoReader(video_path)

            # assume every video has length>=n_sample_frames 
            min_frame_rate = max(len(vr)//self.n_sample_frames, sample_frame_rates[0])
            sample_frame_rate = min(min_frame_rate, sample_frame_rate)
        except Exception as e:
            print("\n")
            print("illegal file", video_path)
            print("\n")

        if sample_frame_rate == sample_frame_rates[0] and random.random() < 0.5:
            return vr, prompt, sample_frame_rate

        return vr, "{} ...{}x".format(prompt, sample_frame_rate), sample_frame_rate

    def __getitem__(self, index):
        idx = index
        sample_frame_rate = 2

        while True:
            vr, prompt, sample_frame_rate = self.getvr(idx, sample_frame_rate)

            if vr is None or len(vr) < self.sample_start_idx+1 \
                    or ((len(vr)-self.sample_start_idx)//sample_frame_rate) < self.n_sample_frames+3:
                idx = random.randint(0, len(self.videolist)-1)
                continue

            break

        ###
        framelst = list(range(self.sample_start_idx, len(vr), sample_frame_rate))
        firstidx = random.randint(0,len(framelst)-self.n_sample_frames)
        sample_index = framelst[firstidx:firstidx+self.n_sample_frames]
        ###

        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        f, c, h, w = video.shape
        neww = (self.height*w)//h

        if neww <= self.width:
            video = F.interpolate(video, (self.height,self.width), mode='bilinear')
        else:
            video = F.interpolate(video, (self.height,neww), mode='bilinear')
            startpos = random.randint(0,neww-self.width-1)
            video = video[:,:,:,startpos:startpos+self.width]

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.tokenizer(
                                prompt,
                                max_length=self.tokenizer.model_max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                            ).input_ids[0]
        }

        return example
