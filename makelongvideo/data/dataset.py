import os
import decord
decord.bridge.set_bridge('torch')

import torch

from torch.utils.data import Dataset
from einops import rearrange

import random


class MakeLongVideoDataset(Dataset):
    def __init__(
            self,
            video_dir: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            sample_frame_rates: str = None,
            tokenizer=None
    ):
        self.video_dir = video_dir
        f = open(os.path.join(self.video_dir, "train.txt"))
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

        # load and sample video frames
        try:
            vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        except Exception as e:
            print("\n")
            print("illegal file", video_path)
            print("\n")

        if sample_frame_rate == self.sample_frame_rates[0] and random.random() < 0.5:
            return vr, prompt

        return vr, "SSFFRR_{} {}".format(sample_frame_rate, prompt)

    def __getitem__(self, index):
        sample_frame_rate = self.sample_frame_rates[random.randint(0,len(self.sample_frame_rates))-1]

        idx = index

        while True:
            vr, prompt = self.getvr(idx, sample_frame_rate)

            if vr is None or (len(vr)//sample_frame_rate) < self.n_sample_frames+3:
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
