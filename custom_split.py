import json 
import numpy as np
import torch
import pickle as pkl


with open("/mnt/nfs/projects/usense/data/egoexo/annotations/atomic_descriptions_val.json", "rb") as f:
    dic = json.load(f)

# split the data into test and val
train = {'ds': dic['ds'], 'take_cam_id_map': dic['take_cam_id_map'], 'annotations': {}}
val = {'ds': dic['ds'], 'take_cam_id_map': dic['take_cam_id_map'], 'annotations': {}}
test = {'ds': dic['ds'], 'take_cam_id_map': dic['take_cam_id_map'], 'annotations': {}}
for elem, subdic in dic['annotations'].items():
    rand_val = np.random.rand()
    if rand_val < 0.5:
        train['annotations'][elem] = subdic
    elif rand_val < 0.75:
        val['annotations'][elem] = subdic 
    else:
        test['annotations'][elem] = subdic


print(len(train['annotations']), len(val['annotations']), len(test['annotations']))
with open("./dataset/egoexo4d/atomic_descriptions_custom_train.json", "w", encoding='utf-8') as f:
    json.dump(train, f)

with open("./dataset/egoexo4d/atomic_descriptions_custom_val.json", "w", encoding='utf-8') as f:
    json.dump(val, f)

with open("./dataset/egoexo4d/atomic_descriptions_custom_test.json", "w", encoding='utf-8') as f:
    json.dump(test, f)

