import json
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import glob
import pickle as pkl
from tqdm import tqdm


with open('/mnt/nfs/projects/usense/data/ego4d/ego4d.json', 'r') as f:
    data = json.load(f)


scenes = []
for item in data['videos']:
    if type(item['scenarios']) == list:
        if len(item['scenarios']) == 1:
            scenes.append(item['scenarios'][0])
        else:
            continue
    else:
        scenes.append(item['scenarios'])



# count number of scenes


print(len(set(scenes)))