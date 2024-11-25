# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import numpy as np
import clip
import torch
import json
from PIL import Image
import pytorch_lightning as pl
from torchvision.transforms import Normalize
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection


class Clip4CLIPModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super(Clip4CLIPModel, self).__init__()
        print("Loading clip4clip model ...")

        self.flag_freeze = kwargs.pop("freeze", True)

        self.text_model = CLIPTextModelWithProjection.from_pretrained("/mnt/nfs/projects/usense/data/clip4clip-webvid150k")
        self.tokenizer = CLIPTokenizer.from_pretrained("/mnt/nfs/projects/usense/data/clip4clip-webvid150k")
        self.video_model = CLIPVisionModelWithProjection.from_pretrained("/mnt/nfs/projects/usense/data/clip4clip-webvid150k")

        self.video_model.eval()
        self.text_model.eval()


        if self.flag_freeze:
            self.eval()
            self.freeze()

    def get_text_embeddings(self, text: List[str], device: Optional[str] = None):

        device = self.device if device is None else device

        # Compute text features
        tokens = self.tokenizer(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.text_model(input_ids=tokens['input_ids'].cuda(), attention_mask=tokens['attention_mask'].cuda())['text_embeds']        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


    def get_video_embeddings(self, video, device: Optional[str] = None):

        # This is a forward pass if features are precomputed
        if len(video.shape) == 2:
            return video

        # video: [batch_size x n_frames x grid x grid x 3]
        batch_size, n_frames, _, grid, _ = video.shape
        video = video.reshape(batch_size * n_frames, 3, grid, grid) # [batch_size * n_frames x 3 x grid x grid] to parallelize
        visual_output_raw = self.video_model(video)
        video_features = visual_output_raw["image_embeds"]
        video_features = video_features.reshape(batch_size, n_frames, -1)

        # average over frames
        video_features = video_features.mean(dim=1)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features
