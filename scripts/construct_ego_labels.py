import json
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import glob
import pickle as pkl
from tqdm import tqdm



label_map = {0: 'Car mechanic',
            1: 'Carpenter',
            2: 'Cleaning / laundry',
            3: 'Clothes, other shopping',
            4: 'Cooking',
            5: 'Crafting/knitting/sewing/drawing/painting',
            6: 'Doing yardwork / shoveling snow',
            7: 'Eating',
            8: 'Farmer',
            9: 'Grocery shopping indoors', 
            10: 'Household management - caring for kids',
            11: 'Playing cards', 
            12: 'Practicing a musical instrument', 
            13: 'Walking on street',
            14: 'jobs related to construction/renovation company\n(Director of work, tiler, plumber, Electrician, Handyman, etc)'
            }

label_map_inv = {v: k for k, v in label_map.items()}
             


with open('/mnt/nfs/projects/usense/data/ego4d/ego4d.json', 'r') as f:
    data = json.load(f)


class2video = {}
for vid in tqdm(data['videos']):
    scn = vid['scenarios']
    if len(scn) == 1:
        if scn[0] in label_map_inv:
            class2video.setdefault(label_map_inv[scn[0]], []).append(vid['video_uid'])
        else:
            continue
    else:
        continue

with open("../dataset/ego4d/class2video.pkl", 'wb') as f:
    pkl.dump(class2video, f)


    
    # print(type(vid['scenarios']))
    # print(vid['scenarios'])
    # break
    # scenes.extend(vid['scenarios'])

# keep only the scenes that have more than 5 occurences
# scenes = np.array(scenes)
# scenes, counts = np.unique(scenes, return_counts=True)
# scenes = scenes[counts > 100]
# print(scenes)
# print(len(scenes))

# print(len(scenes))
# import ipdb; ipdb.set_trace()