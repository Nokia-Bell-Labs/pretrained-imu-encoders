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
from utils import get_zeroshot_classifier, get_zeroshot_classifier_imagebind
from modeling import FewShotModel
import sys
import os
import wandb 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from lib.data_modules import Ego4dDataModule
from lib.imu_models import MW2StackRNNPooling, MW2StackRNNPoolingMultihead
from dataset.ego4d.dataloader import filter_narration, clean_narration_text

import torch
# from imagebind.models.imagebind_model import imagebind_huge



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
DATA_PATH = "/mnt/nfs/projects/usense/data/ego4d/"
# TEXT_PROMPTS = {
#     0: ["Fixing a car", "changing a tire", "checking the engine", "repairing the brakes", "replacing car battery", "tuning up the engine", "fixing car lights", "oil change", "cleaning car interior"],
#     1: ["Carpentry / woodworking", "building a table", "sanding wood", "cutting wood", "assembling furniture", "measuring and marking wood", "applying wood stain", "drilling holes", "carving wood", "repairing wooden furniture"],
#     2: ["Cleaning / laundry", "vacuuming the floor", "dusting furniture", "washing dishes", "mopping the floor", "doing laundry", "folding clothes", "cleaning windows", "scrubbing the bathroom", "taking out the trash"],
#     3: ["Clothes, other shopping", "shopping for clothes", "trying on clothes", "browsing in a store", "buying accessories", "looking at shoes", "checking out sales", "returning items", "shopping for gifts"],
#     4: ["Cooking", "chopping vegetables", "stirring a pot", "baking a cake", "frying food", "grilling meat", "seasoning dishes", "boiling water", "making a salad", "preparing ingredients"],
#     5: ["Crafting/knitting/sewing/drawing/painting", "knitting a scarf", "sewing clothes", "drawing a picture", "painting a canvas", "making jewelry", "crafting with paper", "pottery", "cross-stitching", "wood burning"],
#     6: ["Doing yardwork / shoveling snow", "mowing the lawn", "raking leaves", "planting flowers", "trimming hedges", "watering plants", "pulling weeds", "shoveling snow from driveway", "spreading mulch", "cleaning gutters"],
#     7: ["Eating", "having a meal", "snacking", "drinking coffee", "eating dessert", "having a picnic", "dining out", "eating at home", "enjoying a buffet"],
#     8: ["Farmer", "planting seeds", "harvesting crops", "feeding livestock", "milking cows", "driving a tractor", "fixing farm equipment", "watering fields", "collecting eggs", "baling hay"],
#     9: ["Grocery shopping indoors", "pushing a shopping cart", "choosing fruits and vegetables", "reading food labels", "weighing produce", "buying dairy products", "checking off a shopping list", "standing in line at checkout", "using a self-checkout machine"],
#     10: ["Household management - caring for kids", "feeding children", "changing diapers", "helping with homework", "playing with kids", "reading bedtime stories", "preparing school lunches", "bathing children", "organizing playdates"],
#     11: ["Playing cards", "shuffling cards", "dealing cards", "playing poker", "playing bridge", "playing solitaire", "playing blackjack", "organizing a card game", "counting points"],
#     12: ["Practicing a musical instrument", "playing the piano", "strumming a guitar", "playing the violin", "practicing the flute", "drumming", "playing the trumpet", "tuning an instrument", "reading sheet music"],
#     13: ["Walking on street", "strolling down the street", "walking with a friend", "walking a dog", "window shopping", "walking to work", "walking in a park", "crossing the street", "walking briskly"],
#     14: ["jobs related to construction/renovation company (Director of work, tiler, plumber, Electrician, Handyman, etc)", "directing construction work", "laying tiles", "installing plumbing", "wiring electrical systems", "repairing pipes", "painting walls", "installing flooring", "fixing leaks", "building scaffolding"]
# }
# NUM_CLASSES = 15

TEXT_PROMPTS = {
    0: ["Carpentry / woodworking", "building a table", "sanding wood", "cutting wood", "assembling furniture", "measuring and marking wood", "applying wood stain", "drilling holes", "carving wood", "repairing wooden furniture"],
    1: ["Cleaning / laundry", "vacuuming the floor", "dusting furniture", "washing dishes", "mopping the floor", "doing laundry", "folding clothes", "cleaning windows", "scrubbing the bathroom", "taking out the trash"],
    2: ["Cooking", "chopping vegetables", "stirring a pot", "baking a cake", "frying food", "grilling meat", "seasoning dishes", "boiling water", "making a salad", "preparing ingredients"],
    3: ["Crafting/knitting/sewing/drawing/painting", "knitting a scarf", "sewing clothes", "drawing a picture", "painting a canvas", "making jewelry", "crafting with paper", "pottery", "cross-stitching", "wood burning"],
    4: ["Eating", "having a meal", "snacking", "drinking coffee", "eating dessert", "having a picnic", "dining out", "eating at home", "enjoying a buffet"],
    5: ["Farmer", "planting seeds", "harvesting crops", "feeding livestock", "milking cows", "driving a tractor", "fixing farm equipment", "watering fields", "collecting eggs", "baling hay"],
    6: ["Household management - caring for kids", "feeding children", "changing diapers", "helping with homework", "playing with kids", "reading bedtime stories", "preparing school lunches", "bathing children", "organizing playdates"],
    7: ["Practicing a musical instrument", "playing the piano", "strumming a guitar", "playing the violin", "practicing the flute", "drumming", "playing the trumpet", "tuning an instrument", "reading sheet music"],
    8: ["Walking on street", "strolling down the street", "walking with a friend", "walking a dog", "window shopping", "walking to work", "walking in a park", "crossing the street", "walking briskly"],
    9: ["jobs related to construction/renovation company (Director of work, tiler, plumber, Electrician, Handyman, etc)", "directing construction work", "laying tiles", "installing plumbing", "wiring electrical systems", "repairing pipes", "painting walls", "installing flooring", "fixing leaks", "building scaffolding"]
}
NUM_CLASSES = 10 #5
EMBED_DIM = 512
BATCH_SIZE = 256
NUM_EPOCHS = 20
IMU_SAMPLING_RATE=50

def eval_downstream(configs, args):

    global TEXT_PROMPTS, NUM_CLASSES, EMBED_DIM, BATCH_SIZE, NUM_EPOCHS, IMU_SAMPLING_RATE

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

    list_modalities = ['imu', 'text']
    list_modalities.sort()

    if args.slip:
        encoder = MW2StackRNNPoolingMultihead(size_embeddings=EMBED_DIM).cuda()
    elif args.imagebind:
        encoder = imagebind_huge(pretrained='./imagebind_huge.pth').cuda()
        IMU_SAMPLING_RATE = 400
        EMBED_DIM = 1024
    else:
        encoder = MW2StackRNNPooling(size_embeddings=EMBED_DIM).cuda()


    # Initialize the data module
    dataset_params = {
        "window_sec": window_sec,
        "target_fps": target_fps,
        "list_modalities": list_modalities,
        "clean_narration_func": clean_narration_text,
        "filter_narration_func": filter_narration,
        "imu_sampling_rate": IMU_SAMPLING_RATE,
    }

    datamodule = Ego4dDataModule(
        batch_size=BATCH_SIZE,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
        dataset_params=dataset_params,
        supervised=True,
        num_shots=args.num_shots,
        seed=args.seed,
    )

    if path_load_pretrained_imu_encoder:
        # Load the parameters
        print("Loaded Pretrained IMU Encoder")
        dic = torch.load(path_load_pretrained_imu_encoder)
        dic = {k[12:]: v for k, v in dic["state_dict"].items() if "imu_encoder" in k}
        encoder.load_state_dict(dic)
        clf = get_zeroshot_classifier(text_prompts=TEXT_PROMPTS, embed_dim=EMBED_DIM).cuda()
    
    elif not args.imagebind:
        print("Ranodmly Initialized IMU Encoder...")
        clf = get_zeroshot_classifier(text_prompts=None, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES).cuda()
        
    elif args.imagebind:
        clf = get_zeroshot_classifier_imagebind(encoder,text_prompts=TEXT_PROMPTS).cuda()
        
    name = args.wandb_run_name if args.wandb_run_name is not None else f"Num Shots: {args.num_shots}"
    wandb_logger = WandbLogger(project=args.wandb_project_name, entity=args.wandb_entity, name=name) if args.wandb else CSVLogger("downstream_clf_logs_ego4d", name=name)
    fewshot_model = FewShotModel(imu_encoder=encoder, clf=clf, T_max=int(max(len(datamodule.get_dataset('training')) / BATCH_SIZE, 1) * NUM_EPOCHS), finetune=args.finetune, num_classes=NUM_CLASSES, emb_type=args.emb_type, imagebind=args.imagebind)     #fewshot_model = FewShotModel(imu_encoder=encoder, clf=clf, T_max=int(max(len(train_dataset) / BATCH_SIZE, 1) * NUM_EPOCHS), finetune=args.finetune, num_classes=NUM_CLASSES, emb_type=args.emb_type

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=None,
        dirpath="./downstream_clf_models/",
        filename=f"ego4d_num_shots={args.num_shots}_seed={args.seed}"+"-{epoch:02d}",
        every_n_epochs=5,
        save_last=True
    )

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator='auto', num_sanity_val_steps=0, check_val_every_n_epoch=25, log_every_n_steps=1, logger=wandb_logger, enable_checkpointing=True) #, callbacks=[checkpoint_callback],)  # Adjust parameters as needed
    if trainer.global_rank == 0 and args.wandb:
        wandb_logger.experiment.config.update({**configs, "num_shots": args.num_shots})



    zs_acc = trainer.test(fewshot_model, dataloaders=datamodule, verbose=True) # zero-shot 
    trainer.fit(fewshot_model, datamodule=datamodule) #train_loader, val_loader)
    best_acc = trainer.test(fewshot_model, dataloaders=datamodule, verbose=True)

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
    parser.add_argument("--slip", default=False, action="store_true", help="Use slip model")
    parser.add_argument("--emb_type", default="mmcl", help="Embedding type for multihead model")
    parser.add_argument("--imagebind", default=False, action="store_true", help="Use imagebind model")
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


    # Open a csv file and write a line to it
    with open("./final_ego4d2_results_redo.csv", "a") as f:
        if args.path_load_pretrained_imu_encoder is None:
            f.write(f"Finetuned,{args.seed},{args.num_shots},{acc}\n")
        else:
            f.write(f"{args.path_load_pretrained_imu_encoder.split('/')[1]},{args.seed},{args.num_shots},{acc}\n")