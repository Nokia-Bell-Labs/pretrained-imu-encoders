# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import random
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import WandbLogger, CSVLogger
from utils import get_zeroshot_classifier
from modeling import FewShotModel
import sys
import os
import wandb 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from dataset.egoexo4d.dataloader import EgoExo4dDataset, collate_wrapper
from lib.imu_models import MW2StackRNNPooling, MW2StackRNNPoolingMultihead


# text_prompts = {
#     0: ["admnistering a medical test", "taking a medical test"],
#     1: ["repairing a bicycle", "fixing a bicycle"],
#     2: ["rock climbing", "bouldering"],
#     3: ["playing soccer", "playing basketball"], 
#     4: ["performing CPR", "administering CPR"],
#     5: ["preparing food", "cooking a meal"],
#     6: ["dancing", "performing a dance"],
#     7: ["playing music", "playing a musical instrument"]
# }
DATA_PATH = "/mnt/nfs/projects/usense/data/egoexo/"
TEXT_PROMPTS = {
    0: ["COVID-19 test", "Taking a PCR test", "Receiving a COVID-19 vaccine", "Waiting in line for a COVID-19 test", "Reading a COVID-19 test result", "Medical staff administering a COVID-19 test"],
    1: ["repairing a bicycle", "Fixing a flat tire", "Adjusting bicycle brakes", "Lubricating the bicycle chain", "Replacing a bicycle chain", "Aligning bicycle wheels"],
    2: ["rock climbing", "bouldering", "Climbing an indoor rock wall", "Using climbing gear (harness, carabiners)", "Planning a climbing route", "Reaching the summit of a climb", "Practicing bouldering techniques"],
    3: ["playing soccer", "playing basketball", "Dribbling a soccer ball", "Shooting a soccer goal", "Passing a basketball", "Performing a slam dunk", "Playing a soccer match", "Engaging in a basketball game"], 
    4: ["performing CPR", "Checking for a pulse", "Giving chest compressions", "Using an AED (Automated External Defibrillator)", "Rescuing breathing", "Calling emergency services"],
    5: ["cooking", "Chopping vegetables", "Baking a cake", "Stir-frying in a wok", "Following a recipe book"],
    6: ["dancing", "Ballet dancing", "Salsa dancing", "Hip-hop dancing", "Ballroom dancing", "Practicing dance moves in a studio"],
    7: ["playing a musical instrument", "Playing the piano", "Strumming a guitar", "Playing the violin", "Practicing with a band", "drumming"]
}
NUM_CLASSES = 8
EMBED_DIM = 512
BATCH_SIZE = 256
NUM_EPOCHS = 20
IMU_SAMPLING_RATE=50

def eval_downstream(configs, args):

    random.seed(args.seed)



    if args.num_shots is not None:
        args.num_shots = int(args.num_shots)


    # Load Model Parameters
    model_hparams = configs.get("model_hparams", {})
    window_sec = model_hparams.get("window_sec")
    target_fps = model_hparams.get("target_fps")

    # Params for the trainer
    train_hparams = configs.get("train_hparams", {})
    list_modalities = train_hparams.get("list_modalities")
    path_load_pretrained_imu_encoder = train_hparams.get(
        "path_load_pretrained_imu_encoder"
    )

    list_modalities = ['imu', 'video', 'text']
    list_modalities.sort()

    # Initialize the data module
    dataset_params = {
        "window_sec": window_sec,
        "target_fps": target_fps,
        "list_modalities": list_modalities,
    }

    print("Instantiating Training Set...")
    train_dataset = EgoExo4dDataset(
        window_sec=dataset_params["window_sec"],
        video=True,
        imu=True,
        narr=True,
        audio=False,
        return_tuple=False,
        target_frames_in_window=dataset_params["target_fps"],
        split='custom_train',
        supervised=True,
        num_shots=args.num_shots, # Number of videos per class
        max_n_windows_per_video=20, # Number of windows per video
        imu_sampling_rate = IMU_SAMPLING_RATE, 
    )

    print("Instantiating Val Set...")
    val_dataset = EgoExo4dDataset(
        window_sec=dataset_params["window_sec"],
        video=True,
        imu=True,
        narr=True,
        audio=False,
        return_tuple=False,
        target_frames_in_window=dataset_params["target_fps"],
        split='custom_val',
        supervised=True,
        num_shots=None,
        imu_sampling_rate = IMU_SAMPLING_RATE,
    )

    print("Instantiating Test Set...")
    test_dataset = EgoExo4dDataset(
        window_sec=dataset_params["window_sec"],
        video=True,
        imu=True,
        narr=True,
        audio=False,
        return_tuple=False,
        target_frames_in_window=dataset_params["target_fps"],
        split='custom_test',
        supervised=True,
        num_shots=None,
        imu_sampling_rate = IMU_SAMPLING_RATE
    )

    collate_fn = lambda data: collate_wrapper(
        data, list_modalities
    )

    # crete dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=10
    ) 


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=10
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=10
    )

    if args.multihead:
        encoder = MW2StackRNNPoolingMultihead(size_embeddings=EMBED_DIM).cuda()
    else:
        encoder = MW2StackRNNPooling(size_embeddings=EMBED_DIM).cuda()

    if path_load_pretrained_imu_encoder:
        # Load the parameters
        print("Loaded Pretrained IMU Encoder")
        dic = torch.load(path_load_pretrained_imu_encoder)
        dic = {k[12:]: v for k, v in dic["state_dict"].items() if "imu_encoder" in k}
        encoder.load_state_dict(dic)
        clf = get_zeroshot_classifier(text_prompts=TEXT_PROMPTS, embed_dim=EMBED_DIM).cuda()
    
    else:
        print("Ranodmly Initialized IMU Encoder...")
        clf = get_zeroshot_classifier(text_prompts=None, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES).cuda()
        

    name = args.wandb_run_name if args.wandb_run_name is not None else f"Num Shots: {args.num_shots}"
    wandb_logger = WandbLogger(project=args.wandb_project_name, entity=args.wandb_entity, name=name) if args.wandb else CSVLogger("downstream_clf_logs_realworld", name=name)
    fewshot_model = FewShotModel(imu_encoder=encoder, clf=clf, T_max=int(max(len(train_dataset) / BATCH_SIZE, 1) * NUM_EPOCHS), finetune=args.finetune, num_classes=NUM_CLASSES, emb_type=args.emb_type)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath="./downstream_clf_models/",
        filename=f"num_shots={args.num_shots}_seed={args.seed}"+"-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator='auto', num_sanity_val_steps=0, check_val_every_n_epoch=5, log_every_n_steps=1, logger=wandb_logger, enable_checkpointing=True, callbacks=[checkpoint_callback],)  # Adjust parameters as needed
    if trainer.global_rank == 0 and args.wandb:
        wandb_logger.experiment.config.update({**configs, "num_shots": args.num_shots})



    zs_acc = trainer.test(fewshot_model, dataloaders=test_loader, verbose=True) # zero-shot 
    trainer.fit(fewshot_model, train_loader, val_loader)
    best_acc = trainer.test(fewshot_model, dataloaders=test_loader, ckpt_path='best', verbose=True)

    if trainer.global_rank == 0:
        print("+="*50)
        print("Zero-shot accuracy: ", zs_acc, "Best accuracy after training: ", best_acc)


    # wandb.log({"best_test_acc": best_acc})
    # wandb.finish()

    return best_acc

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
    parser.add_argument("--path_load_pretrained_imu_encoder", default=None) #'./saved/i2c/i2c_s_i_t_tv_se_mw2_w_2.5_master-epoch=05-val_loss=8.79.ckpt')
    parser.add_argument("--num_shots", default=None)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project_name", default="evaluation_downstream")
    parser.add_argument("--wandb_entity", default="arnavmdas")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--finetune", default=False, action="store_true")
    parser.add_argument("--multihead", default=False, action="store_true", help="Use multihead model")
    parser.add_argument("--emb_type", default="mmcl", help="Embedding type for multihead model")
    parser.add_argument("--dataset_name", default="egoexo", type=int)
    args = parser.parse_args()

    # Load the YAML file
    with open(args.path_configs) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Override the configs with args, if requested
    if args.num_workers_for_dm is not None:
        configs["train_hparams"]["num_workers_for_dm"] = int(args.num_workers_for_dm)
    if args.path_load_pretrained_imu_encoder is not None:
        configs["train_hparams"][
            "path_load_pretrained_imu_encoder"
        ] = args.path_load_pretrained_imu_encoder

    configs["finetune"] = args.finetune
    configs["seed"] = args.seed
    


    acc = eval_downstream(configs, args) #num_shots=args.num_shots, seed=args.seed)
    print(f"Num_shots={args.num_shots} Acc: {acc}")
