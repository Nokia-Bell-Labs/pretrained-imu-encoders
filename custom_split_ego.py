import json 
import numpy as np
import torch
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split

# with open("./dataset/ego4d/class2video.pkl", "rb") as f:
#     class2video = pkl.load(f)


# class2dict = {}
# for k, v in class2video.items():
#     for elem in v:
#         class2dict[elem] = k

# X = []
# y = []
# for k, v in class2dict.items():
#     X.append(k)
#     y.append(v)


# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# # save X_train as a json
# with open("./dataset/ego4d/X_training.json", "w", encoding='utf-8') as f:
#     json.dump(X_train, f)

# with open("./dataset/ego4d/X_testing.json", "w", encoding='utf-8') as f:
#     json.dump(X_test, f)

# with open("./dataset/ego4d/X_validation.json", "w", encoding='utf-8') as f:
#     json.dump(X_val, f)




# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))
# print(np.unique(y_val, return_counts=True))

# with open("./dataset/ego4d/X_train.json", "r", encoding="utf-8") as f:
#     X = json.load(f)

# classes = [class2dict[elem] for elem in X]
# print(np.unique(classes, return_counts=True))

with open("./dataset/ego4d/class2video.pkl", "rb") as f:
    class2video = pkl.load(f)


class2dict = {}
for k, v in class2video.items():
    for elem in v:
        class2dict[elem] = k

# trim class2dict
new_class2dict = {}
for k, v in class2dict.items():
    if v in [0,3,6,9,11]:
        pass
    else:
        new_class2dict[k] = v 

print(len(new_class2dict), len(class2dict))


new_class_mapping = {
    1: 0,
    2: 1,
    4: 2,
    5: 3,
    7: 4,
    8: 5,
    10: 6,
    12: 7,
    13: 8,
    14: 9,
}

new_class2dict = {k: new_class_mapping[v] for k, v in new_class2dict.items()}
print(new_class2dict)
labels = [new_class2dict[elem] for elem in new_class2dict.keys()]
print(np.unique(labels, return_counts=True))


# save new_class2dict as a pkl
with open("./dataset/ego4d/new_class2dict.pkl", "wb") as f:
    pkl.dump(new_class2dict, f)