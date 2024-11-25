# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.


import copy
import json
import math
import random
import os
import string
import numpy as np
from tqdm import tqdm
import torch
from dataset.egoexo4d.utils.utils import (
    get_egoexo4d_metadata,
    load_json,
    modality_checker,
    get_video_frames,
    get_imu_frames,
    index_narrations,
    display_animation
)
from typing import Callable, Dict, List, Optional
import glob
import pickle as pkl

# random.seed(1234)
DATA_PATH = "/path/to/egoexo/"
IMU_CACHE_DIR = "./tmp/imu/"
VIDEO_CACHE_DIR = "/path/to/tmp/video/cache/"
PATH_EGO_META = "/path/to/egoexo/takes.json"

def clean_narration_text(narration_text: str) -> str:
    return (
        narration_text.replace("#C C ", "")
        .replace("C", "")
        .replace("#unsure", "something")
        .strip()
        .strip(string.punctuation)
        .lower()[:128]
    )


class EgoExo4dDataset(torch.utils.data.Dataset):
    """
    Datasets class for the ego4d Dataset
    This dataloder loops over (Audio, Video, IMU) windows
    of n-sec.
    """

    def __init__(
        self,
        video=True,
        audio=True,
        imu=True,
        narr=False,
        window_sec: float = 1.0,
        target_frames_in_window: int = 10,
        return_tuple: bool = True,
        cache_imu: bool = False,
        clean_narration_func: Callable[[str], str] = clean_narration_text,
        window_sample_rate: float = 1.0,
        imu_sampling_rate: int = 200,
        max_n_windows_per_video: Optional[int] = None,
        split: str = "training",
        supervised: bool = False,
        num_shots: Optional[int] = None,
        seed=1234,
        create_cache_mode: bool = False,
    ):
    
        self.return_tuple = return_tuple
        self.cache_imu = {"cache": cache_imu, "path": IMU_CACHE_DIR}
        if cache_imu and not os.path.exists(self.cache_imu["path"]):
            os.makedirs(self.cache_imu["path"], exist_ok=True)
        self.window_sec = window_sec
        self.target_frames_in_window = target_frames_in_window
        self.imu_sampling_rate = imu_sampling_rate
        self.meta_video = get_egoexo4d_metadata()
        self.supervised = supervised
        self.num_shots = num_shots
        self.create_cache_mode = create_cache_mode
        if self.supervised:
            self.labels = []
            with open("./dataset/egoexo4d/class2video.pkl", "rb") as f:
                class2video = pkl.load(f)
            self.class_dict = {}
            for k,v in class2video.items():
                for pth in v:
                    self.class_dict[pth] = k

        # load bad_imu.txt as a list
        with open("./dataset/egoexo4d/bad_imu.txt", "r") as f:
            lines = f.readlines()
            bad_imus = [line.strip() for line in lines]

        bad_imus = set(bad_imus)

        # bools to check which modality is requested
        self.video = video
        self.imu = imu
        self.narr = narr
        
        print("Loading narrations")
        narration_dict, _ = index_narrations(split) #

        # Loop through unique video_uids
        self.bad_set = []
        for uid, value in self.meta_video.items():
            files = glob.glob(os.path.join(DATA_PATH, f"{self.meta_video[uid]['root_dir']}/*.vrs"))
            if len(files) == 0:
                self.bad_set.append(uid)

        self.bad_set = set(self.bad_set)
        print(f"There are {len(self.bad_set)} videos with missing .vrs files!")


        if self.supervised and self.num_shots is not None:
            cls_counts = {}
            class_idx = {}

        self.window_idx = []
        for video_uid, narrations in tqdm(narration_dict.items()):

            imu_only = False

            # file = os.path.join(DATA_PATH, f"{self.meta_video[uid]['root_dir']}")
            if self.supervised:
                label = self.class_dict[os.path.join(DATA_PATH, f"{self.meta_video[video_uid]['root_dir']}")]

            # SKIP if the video_uid is not in the metadata
            if video_uid not in self.meta_video:
                continue

            if video_uid in self.bad_set:
                continue

            video_duration = self.meta_video[video_uid]["duration_sec"]
            n_windows_per_video = 0

            if max_n_windows_per_video is not None:
                random.shuffle(narrations)

            """
            We are going to loop over the narrations and create windows of data
            around the narration timestamps. The window will be 2*window_sec seconds long.
            We only sample a single window per narration point!!
            """

            # for (start_time, end_time, text, a_uid) in narrations:
            for (timestamp, text) in narrations:
                # if not filter_narration_func(text):
                #     continue
                if window_sample_rate != 1.0 and random.random() > window_sample_rate: # how many windows do we actually want to sample
                    continue
                """
                ATOMIC
                """
                # check if it's the timestamp is at the very beginning
                if timestamp <= window_sec * 2:
                    w_s = 0.0
                    w_e = window_sec * 2
                # check if it's the time stamp is at the very end
                elif timestamp + window_sec * 2 >= video_duration:
                    w_s = video_duration - window_sec * 2
                    w_e = video_duration
                else:
                    w_s = timestamp - window_sec
                    w_e = timestamp + window_sec

                w_s = int(math.floor(w_s))
                w_e = int(math.floor(w_e))

                if w_s < 0 or w_e > video_duration:
                    continue

                if f"{video_uid}_{w_s}_{w_e}" in bad_imus:
                    continue


                # video_cache_name = os.path.join(VIDEO_CACHE_DIR, f"{video_uid}_{w_s}_{w_e}")
                # if video_cache_name in seen_paths:
                #     print(f"Duplicate path found: {video_cache_name}")
                #     continue

                if w_e - w_s == window_sec * 2:
                    # print("Processing window")
                    if self.supervised:
                        self.labels.append(label)

                    # print(cls_counts)
                    # print(self.labels)
                    input_dict = {
                        "window_start": w_s,
                        "window_end": w_e,
                        "video_uid": video_uid,
                        "text": clean_narration_func(text),
                        "imu_only": imu_only
                    } 

                    # video_cache_name = os.path.join(VIDEO_CACHE_DIR, f"{video_uid}_{w_s}_{w_e}_embedding.pt")

                    # if os.path.exists(video_cache_name):
                    #     continue
                    # else:
                    n_windows_per_video += 1

                    if max_n_windows_per_video is not None and n_windows_per_video >= max_n_windows_per_video:
                        break

                    self.window_idx.append(input_dict)
                    
                    if self.supervised and self.num_shots is not None:
                        if label in class_idx:
                            class_idx[label].append(len(self.window_idx) - 1)
                        else:
                            class_idx[label] = [len(self.window_idx) - 1]

                else:
                    print("Skipping window")
                    continue


        # set numpy seed 
        np.random.seed(seed)
        if num_shots is not None:
            idx = []
            for k, v in class_idx.items():
                # select num_shots items without replacement from v
                if len(v) > num_shots:
                    # shuffle v and pick num_shots
                    v_shuffled = np.random.permutation(v)
                    selected = v_shuffled[:num_shots]
                    # selected = np.random.choice(v, num_shots, replace=False)
                    idx.extend(selected)
                else:
                    idx.extend(v)
        
            self.window_idx = [self.window_idx[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

        print(f"There are {len(self.window_idx)} windows to process.")


        # count how many windows there are with imu_only = True
        cnt = 0
        for window in self.window_idx:
            if window["imu_only"]:
                cnt += 1

        print(f"There are {cnt} windows with imu_only = True, out of {len(self.window_idx)} windows.")

    def check_modality_clip_uid(self, video_uid):
        """
        Check which modality is avalibale in the clip based on the request input
        """
        has_imu, _ = modality_checker(self.meta_video[video_uid])
        if self.imu and not has_imu:
            return False

        return True

    def __len__(self):
        # return 100 # DEBUGGING
        return len(self.window_idx)

    def __getitem__(self, idx):
            
        # while True:
            # dict_out = copy.deepcopy(self.window_idx[idx])
            # uid = dict_out["video_uid"]
            # w_s = dict_out["window_start"]
            # w_e = dict_out["window_end"]
            # text = dict_out["text"]
            # video_cache_name = os.path.join(VIDEO_CACHE_DIR, f"{uid}_{w_s}_{w_e}_embedding.pt")
            # if not os.path.exists(video_cache_name):
            #     break 

            # else:
            #     idx = random.randint(0, len(self.window_idx) - 1)

        dict_out = copy.deepcopy(self.window_idx[idx])
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        text = dict_out["text"]
        imu_only = dict_out["imu_only"]

        if self.video and not imu_only:
            # if cache exists
            video_cache_name = os.path.join(VIDEO_CACHE_DIR, f"{uid}_{w_s}_{w_e}_embedding.pt")
            if os.path.exists(video_cache_name):
                
                dict_out["video"] = torch.load(video_cache_name, map_location="cpu")
                if abs(torch.mean(torch.abs(dict_out["video"])).item()) < 1e-6:
                    print(video_cache_name, "EMPTY VECTOR!!")

            else:
                # print("No cached file found...")
                files = glob.glob(os.path.join(DATA_PATH, f"{self.meta_video[uid]['root_dir']}/frame_aligned_videos/*_214-1.mp4"))
                if files:
                    path = files[0]
                else:
                    raise FileNotFoundError(f"Video path for {uid} does not exist")
                dict_out["video"] = get_video_frames(
                    video_fn=path,
                    video_start_sec=w_s,
                    video_end_sec=w_e,
                    target_frames_in_window=self.target_frames_in_window,
                )
                dict_out["video_cache_name"] = video_cache_name
                # print(dict_out["video"]["frames"].shape)


        if self.supervised:
            dict_out["label"] = self.labels[idx]

        # @TODO: implement this for our case
        if self.imu:
            dict_out["imu"] = get_imu_frames(
                uid=uid,
                video_start_sec=w_s,
                video_end_sec=w_e,
                cache=self.cache_imu,
                data_source_file=os.path.join(DATA_PATH, self.meta_video[uid]['root_dir'], "processed_imu.pkl"),
                sampling_rate=self.imu_sampling_rate
            )

            # @QUESTION: Do we want IMU to be replaced by zeros when it is None?
            if dict_out["imu"] is None:
                print("BAD IMU shouldn't be here")

                # write to a text file named bad IMU
                with open("./dataset/egoexo4d/bad_imu.txt", "a") as f:
                    f.write(f"{uid}_{w_s}_{w_e}\n")
                
                dict_out["imu"] = {
                    "signal": torch.zeros(6, int(self.window_sec * 2) * 200)
                }

        if self.narr and not imu_only:
            dict_out["narration"] = text

        dict_out["imu_only"] = imu_only
        # # Create a temporary directory
        # if not os.path.exists(f"temp/{uid}"):
        #     os.makedirs(f"temp/{uid}")

        # if self.video:
        #     video = dict_out["video"]["frames"].permute(1, 2, 3, 0).numpy()
        #     display_animation(video, text, f"temp/{uid}/vid.gif")
        

        if self.return_tuple:
            tuple_out = ()

            if self.video:
                tuple_out = tuple_out + (dict_out["video"]["frames"],)
            if self.imu:
                tuple_out = tuple_out + (dict_out["imu"]["signal"],)
            if self.narr:
                tuple_out = tuple_out + (text,)

            return tuple_out


        # import ipdb; ipdb.set_trace()

        return dict_out


def collate_wrapper(data, list_modalities, create_cache_mode=False):

    has_imu = "imu" in list_modalities
    has_video = "video" in list_modalities
    has_text = "text" in list_modalities

    if has_imu:
        input_tensor_IMU = []
    if has_video:
        input_tensor_video = []
        input_caches_video = []
        input_video_metadata = []
    if has_text:
        input_tensor_NARRATION = []

    supervised = False
    if "label" in data[0]:
        supervised = True
        input_tensor_LABEL = []


    for d in data:

        if has_video:
            if "video" not in d and not d["imu_only"]:
                continue
            else:
                if d["imu_only"]:
                    input_tensor_video.append(torch.zeros(512))
                    input_video_metadata.append({'window_start': 0, 'window_end': 0, 'video_uid': "None"})
                else:
                    if isinstance(d["video"], torch.Tensor):
                        if create_cache_mode:
                            continue
                        else:
                            input_tensor_video.append(d["video"])
                    else:
                        input_tensor_video.append(d["video"]["frames"])
                    if "video_cache_name" in d:
                        input_caches_video.append(d["video_cache_name"])

                    input_video_metadata.append({'window_start': d['window_start'], 'window_end': d['window_end'], 'video_uid': d['video_uid']})

        if has_imu:
            input_tensor_IMU.append(d["imu"]["signal"])
        if supervised: 
            input_tensor_LABEL.append(d["label"])
        if has_text:
            if d["imu_only"]:
                input_tensor_NARRATION.append("#None")
            else:
                input_tensor_NARRATION.append(d["narration"])

    dict_output = {}
    if has_imu:
        dict_output["imu"] = torch.stack(input_tensor_IMU).float()
    if has_video:
        dict_output["video"] = torch.stack(input_tensor_video).float()
        if len(input_caches_video) > 0:
            dict_output["video_cache_name"] = input_caches_video
        dict_output["video_metadata"] = input_video_metadata
    if supervised:
        dict_output["labels"] = torch.tensor(input_tensor_LABEL)
    if has_text:
        dict_output["narration"] = input_tensor_NARRATION

    dict_output["imu_only"] = [d["imu_only"] for d in data] 

    return dict_output


def filter_narration(narration_text: str) -> bool:
    if "#c" in narration_text.lower():
        return True
    return False


class Ego4dDatasetSupervised(torch.utils.data.Dataset):
    """
    Datasets class for the ego4d Dataset
    This dataloder loops over (Audio, Video, IMU) windows
    of n-sec with labels
    """

    def __init__(
        self,
        video=False,
        audio=False,
        imu=False,
        window_sec: float = 1.0, # @TODO: what is this? is every window the same?
        target_frames_in_window: int = 10,
        return_tuple: bool = True,
        cache_imu: bool = False,
        window_set: List = [],
        class_dict: Dict = {},
    ):
        self.return_tuple = return_tuple
        self.cache_imu = {"cache": cache_imu, "path": IMU_CACHE_DIR}
        if cache_imu and not os.path.exists(self.cache_imu["path"]):
            os.makedirs(self.cache_imu["path"], exist_ok=True)
        self.window_sec = window_sec
        self.target_frames_in_window = target_frames_in_window

        self.meta_video = get_egoexo4d_metadata("video")
        self.video = video
        self.audio = audio
        self.imu = imu
        self.class_dict = class_dict
        # load narration
        # print("Loading narrations")

        self.window_idx = []
        for window_dict in tqdm(window_set):
            if not self.check_modality_clip_uid(window_dict["video_uid"]):
                continue
            self.window_idx.append(window_dict)
        print(f"There are {len(self.window_idx)} windows to process.")

    def check_modality_clip_uid(self, video_uid):
        """
        Check which modality is available in the clip based on the request input
        """
        has_imu, has_audio = modality_checker(self.meta_video[video_uid])
        if self.imu and not has_imu:
            return False
        if self.audio and (
            not has_audio
            or not os.path.exists(
                os.path.join(DATA_PATH, f"processed_audios/{video_uid}.wav")
            )
        ):
            return False
        return True

    def __len__(self):
        return len(self.window_idx)

    def __getitem__(self, idx):
        dict_out = copy.deepcopy(self.window_idx[idx])
        uid = dict_out["video_uid"]
        w_s = int(dict_out["window_start"])
        w_e = int(dict_out["window_end"])
        text = dict_out["label"]

        if self.video:
            print("Getting video frames")
            dict_out["video"] = get_video_frames(
                video_fn=os.path.join(DATA_PATH, f"processed_videos/{uid}.mp4"),
                video_start_sec=w_s,
                video_end_sec=w_e,
                target_frames_in_window=self.target_frames_in_window,
            )

        if self.imu:
            # @FIXME: get_imu_frames now requires a data_source_file, which is the path of the converted data, e.g. /egoexo/takes/cmu_bike01_2/processed_imu.pkl
            dict_out["imu"] = get_imu_frames(
                uid=uid,
                video_start_sec=w_s,
                video_end_sec=w_e,
                cache=self.cache_imu,
            )

        dict_out["label"] = self.class_dict[text]

        if self.return_tuple:
            tuple_out = ()

            if self.video:
                tuple_out = tuple_out + (dict_out["video"]["frames"].float(),)
            if self.audio:
                tuple_out = tuple_out + (dict_out["audio"]["signal"].float(),)
            if self.imu:
                tuple_out = tuple_out + (dict_out["imu"]["signal"].float(),)
            tuple_out = tuple_out + (self.class_dict[text],)

            return tuple_out

        return dict_out
