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
from utils import get_zeroshot_classifier, set_random_seed
from modeling import FewShotModel
import sys
import os
import wandb 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from dataset.realworld.realworld_dataset import get_realworld_dataset
from lib.imu_models import MW2StackRNNPooling, MW2StackRNNPoolingMultihead


import torch.nn as nn

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
# TEXT_PROMPTS = {
#     0: ["climbing down"],
#     1: ["climbing up"],
#     2: ["jumping"],
#     3: ["lying down"], 
#     4: ["running"],
#     5: ["sitting"],
#     6: ["standing"],
#     7: ["walking"]
# }
TEXT_PROMPTS = {
    0: ["climbing down", "descending a steep slope", "scaling down a rocky cliff", "lowering themselves from a tree branch", "moving down a ladder", "carefully stepping down a hillside", "clambering down a pile of rubble", "making their way down a narrow stairway", "edging down a steep embankment"],
    1: ["climbing up", "ascending a tall ladder", "climbing a steep hill", "hoisting themselves up a wall", "scrambling up a rocky path", "pulling themselves onto a rooftop", "climbing up a tree trunk", "scaling a vertical rock face", "making their way up a flight of stairs"],
    2: ["jumping", "leaping over a puddle", "jumping from one rock to another", "springing into the air", "hopping across stepping stones", "vaulting over a fence", "bouncing on a trampoline", "jumping off a low wall", "taking a leap of faith"],
    3: ["lying down", "resting on a soft bed of grass", "sprawled out on a beach towel", "lying on a comfortable couch", "relaxing on a picnic blanket", "stretched out under the shade of a tree", "lounging on a hammock", "reclining on a park bench", "lying flat on the ground, staring at the sky"],
    4: ["running", "sprinting across a finish line", "jogging through a park", "dashing down a city street", "running away from a pursuer", "racing along a beach", "chasing after a runaway pet", "running in a marathon", "fleeing from danger"],
    5: ["sitting", "perched on a barstool", "sitting cross-legged on the floor", "relaxing in an armchair", "sitting on a park bench", "perched on a windowsill", "sitting on the edge of a dock", "seated at a dining table", "sitting on a swing"],
    6: ["standing", "standing tall on a mountaintop", "waiting in line", "standing at attention", "leaning against a wall", "standing on a podium", "waiting for a bus", "standing in a field of flowers", "standing in a shallow stream"],
    7: ["walking", "strolling through a forest", "walking along a beach", "meandering through a marketplace", "hiking on a mountain trail", "walking a dog in the park", "wandering through a museum", "walking in the rain with an umbrella", "walking across a bridge"]
}

NUM_CLASSES = 8
EMBED_DIM = 512
BATCH_SIZE = 256
NUM_EPOCHS = 50
IMU_SAMPLING_RATE=50

def eval_downstream(configs, args):

    global TEXT_PROMPTS, NUM_CLASSES, EMBED_DIM, BATCH_SIZE, NUM_EPOCHS, IMU_SAMPLING_RATE

    set_random_seed(args.seed)


    if args.num_shots is not None:
        args.num_shots = int(args.num_shots)


    # Load Model Parameters
    model_hparams = configs.get("model_hparams", {})
    window_sec = 2*model_hparams.get("window_sec")


    if args.multihead:
        encoder = MW2StackRNNPoolingMultihead(size_embeddings=EMBED_DIM).cuda()
    else:
        encoder = MW2StackRNNPooling(size_embeddings=EMBED_DIM).cuda()

    # Params for the trainer
    train_hparams = configs.get("train_hparams", {})
    path_load_pretrained_imu_encoder = train_hparams.get(
        "path_load_pretrained_imu_encoder"
    )


    print("Instantiating Training Set...")
    train_dataset = get_realworld_dataset(split='training', window_sec=window_sec, num_shots=args.num_shots, imu_sampling_rate=IMU_SAMPLING_RATE)
    val_dataset = get_realworld_dataset(split='validation', window_sec=window_sec, imu_sampling_rate=IMU_SAMPLING_RATE)
    test_dataset = get_realworld_dataset(split='testing', window_sec=window_sec, imu_sampling_rate=IMU_SAMPLING_RATE)

    
    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        num_workers=10
    ) 


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=10
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=10
    )


    if path_load_pretrained_imu_encoder:
        # Load the parameters
        print("Loaded Pretrained IMU Encoder")
        dic = torch.load(path_load_pretrained_imu_encoder)
        dic = {k[12:]: v for k, v in dic["state_dict"].items() if "imu_encoder" in k}
        encoder.load_state_dict(dic)

        clf = get_zeroshot_classifier(text_prompts=TEXT_PROMPTS, embed_dim=EMBED_DIM).cuda()
 
        
    name = args.wandb_run_name if args.wandb_run_name is not None else f"Num Shots: {args.num_shots}"

    wandb_logger = WandbLogger(project=args.wandb_project_name, entity=args.wandb_entity, name=name) if args.wandb else CSVLogger("downstream_clf_logs_realworld", name=name)
    fewshot_model = FewShotModel(imu_encoder=encoder, clf=clf, T_max=int(max(len(train_dataset) / BATCH_SIZE, 1) * NUM_EPOCHS), finetune=args.finetune, num_classes=NUM_CLASSES, emb_type=args.emb_type, imagebind=args.imagebind)     #fewshot_model = FewShotModel(imu_encoder=encoder, clf=clf, T_max=int(max(len(train_dataset) / BATCH_SIZE, 1) * NUM_EPOCHS), finetune=args.finetune, num_classes=NUM_CLASSES, emb_type=args.emb_type

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath="./downstream_clf_models_realworld_test2/",
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
    with open("./final_realworld_results.csv", "a") as f:
        if args.path_load_pretrained_imu_encoder is None:
            f.write(f"Finetuning,{args.seed},{args.num_shots},{acc}\n")
        else:
            f.write(f"{args.path_load_pretrained_imu_encoder.split('/')[1]},{args.seed},{args.num_shots},{acc}\n")