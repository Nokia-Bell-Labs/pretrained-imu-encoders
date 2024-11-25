# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
from bisect import bisect_left
from collections import defaultdict
import math
import os
import csv
import json
from typing import Any, List, Optional
import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from PIL import Image


import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import pickle

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

PATH_EGO_META = "/mnt/nfs/projects/usense/data/egoexo/takes.json"
DATA_PATH = "../../checkpoint/clips"
TOLERANCE_MS = 100


def load_json(json_path: str):
    """
    Load a json file
    """
    with open(json_path, "r", encoding="utf-8") as f_name:
        data = json.load(f_name)
    return data


def save_json(json_path: str, data_obj: Any):
    """
    Save a json file
    """
    with open(json_path, "w", encoding="utf-8") as f_name:
        json.dump(data_obj, f_name, indent=4)


def load_csv(csv_path: str):
    """
    Load a CSV file
    """
    with open(csv_path, "r", encoding="utf-8") as f_name:
        reader = csv.DictReader(f_name)
        data = []
        for row in reader:
            data.append(row)
    return data


def load_npy(npy_path: str):
    """
    Load a json file
    """
    with open(npy_path, "rb") as f_name:
        data = np.load(f_name)
    return data


def save_npy(npy_path: str, np_array: np.ndarray):
    """
    Load a json file
    """
    with open(npy_path, "wb") as f_name:
        np.save(f_name, np_array)


# def get_ego4d_metadata(types: str = "clip"):
#     """
#     Get ego4d metadata
#     """
#     return {
#         clip[f"{types}_uid"]: clip for clip in load_json(PATH_EGO_META)[f"{types}s"]
#     }
        
def get_egoexo4d_metadata(types: str = "clip"):
    """
    Get ego4d metadata
    """
    return {item['take_uid']: item for item in load_json(PATH_EGO_META)}


def modality_checker(meta_video: dict):
    """
    Give the video metadata return which modality is available
    """
    has_imu = meta_video["has_imu"]
    has_audio = (
        False if meta_video["video_metadata"]["audio_start_sec"] is None else True
    )
    return has_imu, has_audio


def get_windows_in_clip(s_time: float, e_time: float, window_sec: float, stride: float):
    """
    Given start and end time, return windows of size window_sec.
    If stride!=window_sec, convolve with stride.
    """
    windows = []
    for window_start, window_end in zip(
        np.arange(s_time, e_time, stride),
        np.arange(
            s_time + window_sec,
            e_time,
            stride,
        ),
    ):
        windows.append([window_start, window_end])
    return windows


def resample(
    signals: np.ndarray,
    timestamps: np.ndarray,
    original_sample_rate: int,
    resample_rate: int,
):
    """
    Resamples data to new sample rate
    """
    signals = torch.as_tensor(signals)
    timestamps = torch.from_numpy(timestamps).unsqueeze(-1)
    signals = torchaudio.functional.resample(
        waveform=signals.data.T,
        orig_freq=original_sample_rate,
        new_freq=resample_rate,
    ).T.numpy()

    nsamples = len(signals)

    period = 1 / resample_rate

    # timestamps are expected to be shape (N, 1)
    initital_seconds = timestamps[0] / 1e3

    ntimes = (torch.arange(nsamples) * period).view(-1, 1) + initital_seconds

    timestamps = (ntimes * 1e3).squeeze().numpy()
    return signals, timestamps


def delta(first_num: float, second_num: float):
    """Compute the absolute value of the difference of two numbers"""
    return abs(first_num - second_num)


def padIMU(signal, duration_sec, sampling_rate=200):
    """
    Pad the signal if necessary
    """
    expected_elements = round(duration_sec) * sampling_rate

    if signal.shape[0] > expected_elements:
        signal = signal[:expected_elements, :]
    elif signal.shape[0] < expected_elements:
        padding = expected_elements - signal.shape[0]
        padded_zeros = np.zeros((padding, 6))
        signal = np.concatenate([signal, padded_zeros], 0)
        # signal = signal[:expected_elements, :]
    return signal


def padAudio(signal, duration_sec, sr):
    """
    Pad the audio signal if necessary
    """
    expected_elements = round(duration_sec * int(sr))
    if signal.shape[1] < expected_elements:
        pad = (0, expected_elements - signal.shape[1])
        signal = torch.nn.functional.pad(signal, pad)
    return signal


def padVIDEO(frames, fps, duration_sec):
    """
    Pad the video frames if necessary
    """
    expected_elements = round(duration_sec) * int(fps)

    if frames.shape[0] > expected_elements:
        frames = frames[:expected_elements, :, :, :]
    elif frames.shape[0] < expected_elements:
        padding = expected_elements - frames.shape[0]
        padded_zeros = np.zeros(
            (padding, frames.shape[1], frames.shape[2], frames.shape[3])
        )
        frames = np.concatenate([frames, padded_zeros], 0)
    return frames

def index_narrations(split: str = "training"):
    # narration_raw = load_json("/datasets01/ego4d_track2/v1/annotations/narration.json")
    if split == "training":
        # narration_raw = load_json("/mnt/nfs/projects/usense/data/egoexo/annotations/keystep_train.json")['annotations']
        narration_raw = load_json("/mnt/nfs/projects/usense/data/egoexo/annotations/atomic_descriptions_train.json")['annotations']
    elif split == "validation":
        narration_raw = load_json("/mnt/nfs/projects/usense/data/egoexo/annotations/atomic_descriptions_val.json")['annotations']
    

    elif "custom_train" in split:
        print("Loading custom train")
        narration_raw = load_json("./dataset/egoexo4d/atomic_descriptions_custom_train.json")['annotations']
    elif "custom_val" in split:
        print("Loading custom val")
        narration_raw = load_json("./dataset/egoexo4d/atomic_descriptions_custom_val.json")['annotations']
    elif "custom_test" in split:
        print("Loading custom test")
        narration_raw = load_json("./dataset/egoexo4d/atomic_descriptions_custom_test.json")['annotations']

    narration_dict = defaultdict(list)
    avg_len = []

    print(f"Processing {len(narration_raw.keys())} videos...")

    # Loop over v_id, narr pairs
    for v_id, narr in narration_raw.items():
        """
        - For a particular video indexed by v_id each narr contains text summaries 
          for the whole video and (dense) narrations for time segments.
        """

        """ 
        IF USING KEYSTEP
        """ 
    #     if 'segments' not in narr.keys():
    #         continue

    #     segments = narr['segments']
    #     assert v_id == narr['take_uid']

    #     if len(segments) > 0:
    #         narration_dict[v_id] = [
    #             (
    #                 float(seg["start_time"]),
    #                 float(seg["end_time"]),
    #                 seg["step_description"],
    #                 seg["step_id"])
    #             for seg in segments
    #         ]
    #         avg_len.append(len(narration_dict[v_id]))
    #     else:
    #         narration_dict[v_id] = []

    # print(f"Avg. narration length {np.mean(avg_len)}")
    # return narration_dict, None

        """
        IF USING ATOMIC DESCRIPTIONS 
        """
        segments = narr[0]['descriptions']

        if len(segments) > 0:
            narration_dict[v_id] = [
                (
                    float(seg["timestamp"]),
                    seg["text"]
                )
                for seg in segments
            ]
            avg_len.append(len(narration_dict[v_id]))
        else:
            narration_dict[v_id] = []

    print(f"Avg. narration length {np.mean(avg_len)}")
    return narration_dict, None

### OLD FUNCTION
# def index_narrations():
#     print("loading narration_raw")
#     # narration_raw = load_json("/datasets01/ego4d_track2/v1/annotations/narration.json")
#     narration_raw = load_json("/mnt/nfs/projects/usense/data/egoexo/annotations/keystep_train.json")

#     narration_dict = defaultdict(list)
#     summary_dict = defaultdict(list)
#     avg_len = []

#     # Loop over v_id, narr pairs
#     for v_id, narr in narration_raw.items():
#         narr_list = []
#         summ_list = []

#         """
#         - For a particular video indexed by v_id each narr contains text summaries 
#           for the whole video and (dense) narrations for time segments.
#         - @TODO: why are there two passes? why do we concatenate them together?
#         """
#         if "narration_pass_1" in narr:
#             narr_list += narr["narration_pass_1"]["narrations"]
#             summ_list += narr["narration_pass_1"]["summaries"]
#         if "narration_pass_2" in narr:
#             narr_list += narr["narration_pass_2"]["narrations"]
#             summ_list += narr["narration_pass_2"]["summaries"]

#         if len(narr_list) > 0:
#             narration_dict[v_id] = [
#                 (
#                     float(n_t["timestamp_sec"]),
#                     n_t["narration_text"],
#                     n_t["annotation_uid"],
#                     n_t["timestamp_frame"],
#                 )
#                 for n_t in narr_list
#             ]
#             avg_len.append(len(narration_dict[v_id]))
#         else:
#             narration_dict[v_id] = []
#         if len(summ_list) > 0:
#             summary_dict[v_id] = [
#                 (
#                     float(s_t["start_sec"]),
#                     float(s_t["end_sec"]),
#                     s_t["summary_text"],
#                 )
#                 for s_t in summ_list
#             ]
#         else:
#             summary_dict[v_id] = []
#     # print(f"Number of Videos with narration {len(narration_dict)}")
#     # print(f"Avg. narration length {np.mean(avg_len)}")
#     # print(f"Number of Videos with summaries {len(summary_dict)}")


#     ###### only the narration_dict is used!!!!
#     return narration_dict, summary_dict


def resampleIMU(signal, timestamps, resample_rate=200):
    sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
    # resample all to 200hz
    if sampling_rate != resample_rate:
        signal, timestamps = resample(signal, timestamps, sampling_rate, resample_rate)
    return signal, timestamps


def tosec(value):
    return value / 1000


def toms(value):
    return value * 1000


# def downsample_video(
#     frames: torch.Tensor = torch.zeros(3, 10, 224, 224), targer_frames: int = 5
# ):
#     """
#     Downsample video to target number of frame. For example from [3,10,224,224] to [3,5,224,224]
#     """
#     temporal_dim = 1
#     num_frames_sampled = frames.size(temporal_dim)
#     # -1 because index starts from 0. linspace includes both ends in the sampled list
#     selected_frame_indices = torch.linspace(
#         0, num_frames_sampled - 1, targer_frames
#     ).long()
#     return torch.index_select(frames, temporal_dim, selected_frame_indices)

def downsample_video(
    frames: torch.Tensor = torch.zeros(3, 10, 224, 224), targer_frames: int = 5
):
    """
    Downsample video to target number of frame. For example from [3,10,224,224] to [3,5,224,224]
    """
    temporal_dim = 0
    num_frames_sampled = frames.size(temporal_dim)
    # -1 because index starts from 0. linspace includes both ends in the sampled list
    selected_frame_indices = torch.linspace(
        0, num_frames_sampled - 1, targer_frames
    ).long()
    return torch.index_select(frames, temporal_dim, selected_frame_indices)


def get_video_frames(video_fn, target_frames_in_window=1, video_start_sec=10, video_end_sec=25):

    size = 224
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),            
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)
    
    cap = cv2.VideoCapture(video_fn)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps < 1:
        frames = torch.zeros([3, size, size], dtype=np.float32) 
        print("ERROR: problem reading video file: ", video_fn)
    else:
        if video_end_sec is None:
            video_end_sec = frameCount / fps
        if video_start_sec < 0:
            video_start_sec = 0
        if video_end_sec > frameCount / fps:
            video_end_sec = frameCount / fps
        if video_start_sec >= video_end_sec:
            print("ERROR: video_start_sec should be less than video_end_sec")
            return None

        start_frame = int(video_start_sec * fps)
        end_frame = int(video_end_sec * fps)
        time_depth = end_frame - start_frame

        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        n_frames_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = torch.FloatTensor(time_depth, 3, 224, 224)
        n_frames = 0
        for f in range(min(time_depth, n_frames_available)):
            ret, frame = cap.read()
            if not ret:
                print(f"ERROR: Bad frame, {video_fn}, {start_frame}, {n_frames}, {f}")
                return {
                    "frames": torch.zeros(target_frames_in_window*time_depth, 3, 224, 224),
                    "meta": {"video_fps": target_frames_in_window*time_depth},
                }
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[f,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))

            last_frame = f

        # interval = fps / fps
        # frames_idx = np.floor(np.arange(start_frame, end_frame, interval)).astype(int)
        
        # images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)
        # ret = True     

        # for i, idx in enumerate(frames_idx):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        #     ret, frame = cap.read()    
        #     if not ret: break
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             
        #     last_frame = i
        #     images[i,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))
            
        frames = frames[:last_frame+1]
    


    cap.release()
    video_frames = frames #torch.tensor(frames)
    # print(f"Video frames shape: {video_frames.shape}")
    if target_frames_in_window != fps:
        video_frames = downsample_video(video_frames, target_frames_in_window*(video_end_sec - video_start_sec))

    # print(f"Video frames shape: {video_frames.shape}")
    # print("+---------------------------------+")
    return {"frames": video_frames, "meta": {"video_fps": target_frames_in_window}}

# def get_video_frames(
#     video_fn: str,
#     video_start_sec: float,
#     video_end_sec: float,
#     target_frames_in_window: int = 10,
# ):
    
#     target_frames_in_window = target_frames_in_window * (video_end_sec - video_start_sec)

#     channels = 3
#     height = 224 # 1408
#     width = 224 # 1408

#     cap = cv2.VideoCapture(video_fn) # read video file
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     start_frame = int(math.floor(video_start_sec * fps))
#     stop_frame = int(math.floor(video_end_sec * fps))
#     time_depth = stop_frame - start_frame

#     if start_frame:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#     n_frames_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames = torch.FloatTensor(channels, time_depth, height, width)
#     n_frames = 0
#     for f in range(min(time_depth, n_frames_available)):
#         ret, frame = cap.read()
#         if not ret:
#             print(f"ERROR: Bad frame, {video_fn}, {start_frame}, {n_frames}, {f}")
#             return {
#                 "frames": torch.zeros(channels, target_frames_in_window, height, width),
#                 "meta": {"video_fps": target_frames_in_window},
#             }

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Central crop
#         # height, width, _ = frame.shape
#         # y = (height - height) // 2
#         # x = (width - width) // 2
#         # frame = frame[y : y + height, x : x + width]
#         frame = cv2.resize(frame, (width, height))
#         frame = torch.from_numpy(frame)

#         # HWC 2 CHW
#         frame = frame.permute(2, 0, 1)
#         frames[:, f, :, :] = frame
#         n_frames += 1
#         if stop_frame and start_frame and stop_frame - start_frame + 1 == n_frames:
#             break

#     if target_frames_in_window != frames.size(1):
#         frames = downsample_video(frames, target_frames_in_window)
#     frames = frames / 255.0

    return {"frames": frames, "meta": {"video_fps": target_frames_in_window}}


def check_window_signal(info_t, w_s, w_e):
    length = w_e - w_s
    frame_offset = int(w_s * info_t.sample_rate)
    num_frames = int(length * info_t.sample_rate)
    if frame_offset + num_frames >= info_t.num_frames:
        return False
    else:
        return True

def print_stat_signal(signal, timestamps):
    print(f"Timestamps:{timestamps.shape}")
    print(f"Signal:{signal.shape}")
    sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
    print(f"Sampling Rate: {sampling_rate}")


def get_imu_frames(
    uid: str,
    video_start_sec: float,
    video_end_sec: float,
    cache: dict = {"cache": False, "path": "/tmp/imu"},
    data_source_file="./egoexo/takes/cmu_bike01_2/processed_imu.pkl", # run dataset/ego4d/preprocessing_scripts/convert_imu_vrs_egoexo.py to generate these files
    sampling_rate=200
):
    
    """
    Given a IMU signal return the frames between video_start_sec and video_end_sec
    """

    # video_cache_name = os.path.join(VIDEO_CACHE_DIR, f"{uid}_{w_s}_{w_e}_embedding.pt")
    video_end_sec_orig = video_end_sec
    cache_path = os.path.join(cache["path"], f"{uid}_{video_start_sec}_{video_end_sec}_{sampling_rate}Hz_imu.pkl")

    if cache["cache"] and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            imu_sample = pickle.load(f)

        return imu_sample

    if not os.path.exists(data_source_file):
        print(f"1 file not found {data_source_file}")
        return None
    
    with open(data_source_file, "rb") as f:
        # keys in pickle file: refer to dataset/ego4d/preprocessing_scripts/convert_imu_vrs_egoexo.py
        all_data = pickle.load(f)
    
    timestamps = all_data["timestamps"] # Timestamps are in nanoseconds
    timestamps = timestamps / 1000000 # Conver to millisecond
    timestamps -= timestamps[0] # shift all timestamp to start at 0 (for alignment with video and text)
    signal = all_data["data"].transpose()

    # signal, timestamps = signal[video_start_sec * 1000 : video_end_sec * 1000][::5], timestamps[video_start_sec * 1000 : video_end_sec * 1000][::5] - timestamps[video_start_sec * 1000]

    if toms(video_start_sec) > timestamps[-1]:
        print("2 start or end out of bound", toms(video_start_sec), toms(video_end_sec), timestamps[-1])
        return None
    

    # If the end point is not too far off, then we just zero pad the signal
    if toms(video_end_sec) > timestamps[-1]:
        if abs(toms(video_end_sec) - timestamps[-1]) > TOLERANCE_MS:
            print("2.1 start or end out of bound", toms(video_start_sec), toms(video_end_sec), timestamps[-1])
            return None
        
        else:
            video_end_sec = timestamps[-1]//1000 # @FIXME: THIS IS VERY HARDCODED

        # print("2.1 start or end out of bound", toms(video_start_sec), toms(video_end_sec), timestamps[-1])
        # return None

    start_id = bisect_left(timestamps, toms(video_start_sec))
    end_id = bisect_left(timestamps, toms(video_end_sec))

    # print(start_id, end_id, timestamps[start_id], timestamps[end_id])
    # make sure the retrieved window interval are correct by a max of 1 sec margin
    
    if (
        delta(video_start_sec, tosec(timestamps[start_id])) > 4
        or delta(video_end_sec, tosec(timestamps[end_id])) > 4
    ):
        print("3 video timestamp too far from imu time", video_start_sec, tosec(timestamps[start_id]), start_id,  video_end_sec, tosec(timestamps[end_id], end_id))
        return None

    # get the window
    if start_id == end_id:
        start_id -= 1
        end_id += 1
    signal, timestamps = signal[start_id:end_id], timestamps[start_id:end_id]
    
    if len(signal) < 10 or len(timestamps) < 10:
        print("4 window too short", len(signal))
        return None

    # resample the signal at 200hz if necessary
    signal, timestamps = resampleIMU(signal, timestamps, resample_rate=sampling_rate)

    # pad  the signal if necessary
    signal = padIMU(signal, video_end_sec_orig - video_start_sec, sampling_rate=sampling_rate)

    imu_sample = {
        "timestamp": timestamps,
        "signal": torch.tensor(signal.T),
        "sampling_rate": sampling_rate,
    }

    if cache["cache"]:
        with open(cache_path, "wb") as f:
            pickle.dump(imu_sample, f)

    return imu_sample


def display_image_list(
    images: np.array,
    title: Optional[List[str]] = None,
    columns: Optional[int] = 5,
    width: Optional[int] = 20,
    height: Optional[int] = 8,
    max_images: Optional[int] = 20,
    label_font_size: Optional[int] = 10,
    save_path_img: str = "",
) -> None:
    """
    Util function to plot a set of images with, and save it into
    manifold. If the labels are provided, they will be added as
    title to each of the image.

    Args:
        images: (numpy.ndarray of shape (batch_size, color, hight, width)) - batch of
                images

        labels: (List[str], optional) —  List of strings to be used a title for each img.
        columns: (int, optional) — Number of columns in the grid. Raws are compute accordingly.
        width: (int, optional) — Figure width.
        height: (int, optional) — Figure height.
        max_images: (int, optional) — Maximum number of figure in the grid.
        label_font_size: (int, optional) - font size of the lable in the figure
        save_path_img: (str, ) - path to the manifold to save the figure.

    Example:

        >>> img = torch.rand(2, 3, 224, 224)
        >>> lab = ["a cat", "a dog"]
        >>> display_image_list(
                img,
                lab,
                save_path_img="path_name.png",
            )
    """
    plt.rcParams["axes.grid"] = False

    if len(images) > max_images:
        images = images[0:max_images, :, :, :]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))
    for i in range(len(images)):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        # plt.imshow(transforms.ToPILImage()(images[i]).convert("RGB"))
        plt.imshow(images[i])
        plt.axis("off")

        if title:
            plt.title(title, fontsize=label_font_size)

    with open(save_path_img, "wb") as f_name:
        plt.savefig(fname=f_name, dpi=400)
    plt.close()


# def display_animation(frames, title, save_path_gif):
#     fig, ax = plt.subplots()
#     frames = [[ax.imshow(frames[i])] for i in range(len(frames))]
#     plt.title(title)
#     ani = animation.ArtistAnimation(fig, frames)
#     ani.save(save_path_gif, writer="imagemagick")
#     plt.close()


def display_animation(frames, title, save_path_gif, fps=10):
    fig, ax = plt.subplots()
    ims = [[ax.imshow(frames[i])] for i in range(len(frames))]
    plt.title(title)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(save_path_gif, writer=PillowWriter(fps=fps))
    plt.close()


def display_animation_imu(frames, imu, title, save_path_gif):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_title(title)
    ax2.set_title("Acc.")
    ax3.set_title("Gyro.")
    frames = [[ax1.imshow(frames[i])] for i in range(len(frames))]
    ani = animation.ArtistAnimation(fig, frames)

    ax2.plot(imu[0].cpu().numpy(), color="red")
    ax2.plot(imu[1].cpu().numpy(), color="blue")
    ax2.plot(imu[2].cpu().numpy(), color="green")
    ax3.plot(imu[3].cpu().numpy(), color="red")
    ax3.plot(imu[4].cpu().numpy(), color="blue")
    ax3.plot(imu[5].cpu().numpy(), color="green")
    plt.tight_layout()
    ani.save(save_path_gif, writer="imagemagick")
    plt.close()
