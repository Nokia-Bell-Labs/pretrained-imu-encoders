## PRIMUS 
This is the code for [PRIMUS], a novel pre-training approach to learn effective Inertial Measurement Unit (IMU) motion sensor representations with multimodal and self-supervised learning. This code is built off of the original repo developed
by [IMU2CLIP] (https://arxiv.org/abs/2210.14395).

<ADD A FIGURE>

# Installation
```
conda create -n primus python=3.8
conda activate primus
pip install pytorch_lightning
pip install torchvision
pip install git+https://github.com/openai/CLIP.git # replace with clip4clip
pip install opencv-python
pip install matplotlib
pip install ffmpeg-python
pip install pandas
pip install transformers
```

@TODO: Update this with EgoExo4D installation instructions

# Experiments

**To create a cache where the video frames are preprocessed run:**
```
python pretraining.py --path_configs ./configs/train_contrastive/ego4d_imu2text+video_mw2_sampling_rate=50.yaml --dataset "egoexo4d" --multihead 
```

**To run an example train loop with all three loss terms in the paper:**
```
python pretraining.py --path_configs ./configs/train_contrastive/ego4d_imu2text+video_mw2_sampling_rate=50.yaml --dataset "egoexo4d" --ssl_coeff 0.5 --transform_list 2 4 --multihead --nnclr
```

**To run a pretrained model in downstream task**
```
python downstream.py
```
