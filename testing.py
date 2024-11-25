# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import random
import json
from datetime import datetime
import torch
import pytorch_lightning as pl
from lib.imu_models import MW2StackRNNPooling
from lib.classification_head import Head, ZeroShotClassification
from lib.clip_model import ClipPLModel
from lib.train_modules import ClassificationModule
from lib.data_modules import SupervisedEgoExo4dDataModule
from argparse import ArgumentParser
import yaml
from dataset.egoexo4d.dataloader import EgoExo4dDataset, collate_wrapper
from tqdm import tqdm
import clip
from dataset.egoexo4d.utils.utils import display_animation, get_video_frames, get_ego4d_metadata
import os
import glob

DATA_PATH = "/mnt/nfs/projects/usense/data/egoexo/"

def train_downstream(configs):

    random.seed(1234)

    # Load Model Parameters
    model_hparams = configs.get("model_hparams", {})
    model_name = model_hparams.get("model_name")
    model_suffix = model_hparams.get("model_suffix", "")
    imu_encoder_name = model_hparams.get("imu_encoder_name")
    window_sec = model_hparams.get("window_sec")
    target_fps = model_hparams.get("target_fps")

    # Params for the trainer
    train_hparams = configs.get("train_hparams", {})
    list_modalities = train_hparams.get("list_modalities")
    path_load_pretrained_imu_encoder = train_hparams.get(
        "path_load_pretrained_imu_encoder"
    )

    # Paths, etc.
    path_root_save_dir = f"./saved/{model_name}"
    # print(list_modalities)
    list_modalities = ['imu', 'video', 'text']
    list_modalities.sort()
    str_modality_initials = "".join([m[0] for m in list_modalities])
    model_identifier = (
        f"{model_name}_{str_modality_initials}_ie_{imu_encoder_name}_w_{window_sec}"
    )
    if model_suffix != "":
        model_identifier += "_" + model_suffix
    else:
        model_identifier += "_%d" % (int(datetime.now().timestamp() % 10000))
    path_save_checkpoint = f"{path_root_save_dir}/{model_identifier}.ckpt"
    result_path = f"./results/{model_identifier}"

    # Initialize the data module
    dataset_params = {
        "window_sec": window_sec,
        "target_fps": target_fps,
        "list_modalities": list_modalities,
    }


    dataset = EgoExo4dDataset(
        window_sec=dataset_params["window_sec"],
        video=True,
        imu=True,
        narr=True,
        audio=False,
        return_tuple=False,
        target_frames_in_window=dataset_params["target_fps"],
        split='training',
    )

    collate_fn = lambda data: collate_wrapper(
        data, list_modalities
    )

    # crete dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=20,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    encoder = MW2StackRNNPooling(size_embeddings=512).cuda()
    # print the last layer parameters of encoder

    if path_load_pretrained_imu_encoder:
        # Load the parameters
        dic = torch.load(path_load_pretrained_imu_encoder)
        dic = {k[12:]: v for k, v in dic["state_dict"].items() if "imu_encoder" in k}
        encoder.load_state_dict(dic)
        print("loaded pretrained imu model")



    meta_video = get_ego4d_metadata("video")

    encoder.eval()
    with torch.no_grad():
        outputs_all = []
        texts_all = []
        frames_all = []
        for item in tqdm(dataloader):
            signal = item['imu'].float().cuda()
            outputs = encoder(signal).detach().cpu()
            outputs_all.append(outputs)
            texts_all.extend(item['narration'])
            frames_all.extend(item['video_metadata'])
            torch.cuda.empty_cache()





    outputs_all = torch.cat(outputs_all, dim=0)
    # normalize the outputs
    outputs_all = outputs_all / outputs_all.norm(dim=-1, keepdim=True)
    assert outputs_all.shape[0] == len(texts_all)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_encoder, preprocess = clip.load("ViT-B/16", device=device)    
    prompts = ["admnistering a medical test", "cooking a meal", "fixing a bicycle", "playing a musical instrument", "jumping"]
    for text in prompts:


        text_tokens = clip.tokenize([text]).to(device)

        with torch.no_grad():
            text_features = clip_encoder.encode_text(text_tokens)

        # Normalize the text embedding
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()
        # Find the indices in the outputs_all that are closest to the text_features
        similarity = (outputs_all.cuda() @ text_features.T).squeeze()
        # print(similarity.shape)
        # Rank the indices based on similarity, and get the top 5
        # topk = torch.argsort(similarity, descending=True)[:5]


        # Get the top 5 most similar indices and values
        topk = similarity.topk(5)
        indices = topk.indices
        print(topk.values)

        # # topk = similarity.topk(5)

        # check if f'./qual_results_training/prompt=\"{text}\" and all parents exist 
        # if not os.path.exists(f'./qual_results_training/prompt=\"{text}\"'):
        #     os.makedirs(f'./qual_results_training/prompt=\"{text}\"')
        # print(f"Captions for the most similar IMU embeddings to \"{text}\"")
        # print("-"*10)
        # print(topk)
        # for rank, i in enumerate(topk):
        #     print(texts_all[i])
        #     metadata = frames_all[i]
        #     files = glob.glob(os.path.join(DATA_PATH, f"{meta_video[metadata['video_uid']]['root_dir']}/frame_aligned_videos/*_214-1.mp4"))
        #     path = files[0]
        #     frames = get_video_frames(
        #             video_fn=path,
        #             video_start_sec=metadata['window_start'],
        #             video_end_sec=metadata['window_end'],
        #             target_frames_in_window=10,
        #         )["frames"].permute(1, 2, 3, 0).numpy()

        #     display_animation(frames, texts_all[i], f'./qual_results_training/prompt=\"{text}\"/{rank}.gif')
            
        # print("="*10)
        # print("\n")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Main parameters are defined in a YAML file
    parser.add_argument(
        "--path_configs", default="./configs/train_downstream/default.yaml"
    )

    # Override-params for a quick resource allocation adjustment or for debugging purposes
    # If it is *not* None, the values in args override the values in the YAML file.
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num_workers_for_dm", default=None)
    parser.add_argument("--path_load_pretrained_imu_encoder", default='./saved/i2c/i2c_s_i_t_tv_se_mw2_w_2.5_master-epoch=05-val_loss=8.79.ckpt')
    args = parser.parse_args()

    # Load the YAML file
    with open(args.path_configs) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Override the configs with args, if requested
    if args.gpus is not None:
        configs["train_hparams"]["gpus"] = int(args.gpus)
    if args.num_workers_for_dm is not None:
        configs["train_hparams"]["num_workers_for_dm"] = int(args.num_workers_for_dm)
    if args.path_load_pretrained_imu_encoder is not None:
        configs["train_hparams"][
            "path_load_pretrained_imu_encoder"
        ] = args.path_load_pretrained_imu_encoder

    print(configs)
    train_downstream(configs)




# load csv 
path = "/mnt/nfs/projects/usense/data/egoexo/egoexo4d_metadata.csv"
import pandas as pd

df = pd.read_csv(path)
print(df.head())
