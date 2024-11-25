import json
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import glob
import pickle as pkl

# json_path = "/mnt/nfs/projects/usense/data/egoexo/annotations/keystep_train.json"

# with open(json_path, 'r') as f:
#     data = json.load(f)

# taxonomy = data['taxonomy']

# # Find all step_ids for each key in the taxonomy
# step_ids = {}
# for key in taxonomy:
#     step_ids[key] = []
#     for k,dic in taxonomy[key].items():
#         if 'unique_id' in dic:
#             step_ids[key].append(dic['unique_id'])


# print(step_ids)

# label_map = {0: ['Covid-19 Rapid Antigen Test'], 
#              1: ['Fix a Flat Tire - Replace a Bike Tube', 'Remove a Wheel', 'Install a Wheel', 'Clean and Lubricate the Chain'],
#              2: ['First Aid - CPR'],
#              3: ['Making Coffee latte', 'Making Cucumber & Tomato Salad', 'Cooking Scrambled Eggs', 'Cooking an Omelet', 'Making Milk Tea', 'Making Sesame-Ginger Asian Salad', 'Cooking Tomato & Eggs', 'Making Chai Tea', 'Cooking Noodles', 'Cooking Sushi Rolls', 'Cooking Pasta']
#              }



# label2stepid = {}
# for key, classes in label_map.items():
#     label2stepid[key] = []
#     for cl in classes:
#         label2stepid[key].extend(step_ids[cl])
    

# # Make sure that no step_id is present in more than one class
# for key, classes in label2stepid.items():
#     print(key, classes)
#     for cl in classes:
#         for key2, classes2 in label2stepid.items():
#             if key != key2 and cl != 10000:
#                 assert cl not in classes2, f"Step ID {cl} is present in more than one class {key} and {key2}"



# # print("===== step_ids =====")
# # for k,v in step_ids.items():
# #     print(k, len(v))

# # print("===== label2stepid =====")
# # for k,v in label2stepid.items():
# #     print(k, len(v))
    

# # print(step_ids)


path = "/mnt/nfs/projects/usense/data/egoexo/takes/"

# # list paths in path
# list_paths = glob.glob(path + "*")
# list_paths = [p.split('/')[-1].split('_') for p in list_paths]


# actions = []
# for p in list_paths:
#     actions.extend(p)

# # remove numbers
# actions = [a for a in actions if not any(char.isdigit() for char in a)]

# # remove duplicates
# actions = list(set(actions))
# print(actions)


# actions = ['cpr', 'covidtest', 'Cooking', 'Duet', 'music', 'soccer', 'Dance', 'piano', 'dance', 'rockclimbing', 'cooking', 'pcr', 'minnesota', 'omelet', 'Violin', 'sushi', 'basketball', 'bike', 'salad', 'Guitar', 'covid', 'bouldering', 'guitar', 'Piano']
list_paths = glob.glob(path + "*")

# for path in list_paths:
#     flag = False
#     for action in actions:
#         if action in path:
#             flag = True
#             if action == "Partner":
#                 print(action, path)
#             elif action == "Dance" or action == "dance":
#                 print(action, path)
#             break
#     if not flag:
#         print(f"{path} does not contain an action in the list")


class2action = {
    0: ['covidtest', 'pcr', 'covid'],
    1: ['bike'],
    2: ['bouldering', 'rockclimbing'],
    3: ['soccer', 'basketball'],
    4: ['cpr'],
    5: ['Cooking', 'cooking', 'omelet', 'salad', 'sushi'],
    6: ['Dance', 'dance'],
    7: ['music', 'Guitar', 'guitar', 'Piano', 'piano', 'Violin', 'violin']
}

class2video = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: []
}

list_paths = glob.glob(path + "*")
# list_paths = ['/mnt/nfs/projects/usense/data/egoexo/takes/iiith_cooking_98_2', '/mnt/nfs/projects/usense/data/egoexo/takes/iiith_cooking_109_4']
for pth in list_paths:
    flag = False
    for cls,actions in class2action.items():
        for action in actions:

            # check if action is a substring of path
            

            # print(action, pth, action in pth)
            if action in pth:
                class2video[cls].append(pth)
                flag = True
                break 
        if flag:
            break

    if not flag:
        print(f"{pth} does not contain an action in the list")

total = 0
for k,v in class2video.items():
    total += len(v)

assert total == len(list_paths), f"Total number of videos in class2video {total} does not match total number of videos {len(list_paths)}"

# Save class2video to a pkl file
with open("./dataset/egoexo4d/class2video.pkl", 'wb') as f:
    pkl.dump(class2video, f)