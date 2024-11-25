import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torchmetrics import Accuracy, Metric
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
# from imagebind.models.imagebind_model import ModalityType
# from imagebind import data

class MeanPerClassAccuracy(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("correct", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=1)
        for i in range(self.num_classes):
            mask = target == i
            self.correct[i] += torch.sum(preds[mask] == target[mask])
            self.total[i] += torch.sum(mask)

    def compute(self):
        # Avoid division by zero by replacing 0 totals with 1 (effectively ignoring them in the mean)
        total_nonzero = self.total.clone()
        total_nonzero[total_nonzero == 0] = 1
        mean_per_class_acc = self.correct.float() / total_nonzero.float()
        return torch.mean(mean_per_class_acc[self.total > 0])
    

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.T_max)) / 2
                for base_lr in self.base_lrs
            ]


def get_imagebind_input(imu, device):
    inputs = {
        ModalityType.IMU: imu.to(device).float()
    }
    return inputs

class FewShotModel(pl.LightningModule):
    def __init__(self, imu_encoder, clf, finetune=False, T_max=1000, num_classes=8, emb_type="mmcl", imagebind=False, metric="accuracy"):
        super(FewShotModel, self).__init__()
        self.imu_encoder = imu_encoder
        self.clf = clf
        self.criterion = nn.CrossEntropyLoss()
        if metric == "accuracy":
            self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
            self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        else:
            self.train_accuracy = MeanPerClassAccuracy(num_classes=num_classes)
            self.val_accuracy = MeanPerClassAccuracy(num_classes=num_classes)

        self.finetune = finetune
        self.emb_type = emb_type # For compatibility with SLIP multihead model
        self.imagebind = imagebind
        # self.mlp = mlp

        # freeze all parameters in imu_encoder
        if not self.finetune:
            for param in self.imu_encoder.parameters():
                param.requires_grad = False

        self.T_max = T_max

        self.test_step_yhat = []
        self.test_step_y = []


    def forward(self, x):
        if self.finetune:
            imu_embeddings = self.imu_encoder(x)
        else:
            with torch.no_grad():
                imu_embeddings = self.imu_encoder(x)

        if type(imu_embeddings) is dict:
            if self.imagebind:
                imu_embeddings = imu_embeddings[ModalityType.IMU]
            else:
                imu_embeddings = imu_embeddings[self.emb_type] # For compatibility with SLIP multihead model

        return self.clf(imu_embeddings)

    def training_step(self, batch, batch_idx):
        x,y = batch['imu'].float(),batch["labels"].long()

        if self.imagebind:
            x = get_imagebind_input(x, self.device)

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.train_accuracy(y_hat, y)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch['imu'].float(),batch["labels"].long()
        if self.imagebind:
            x = get_imagebind_input(x, self.device)
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        acc = self.val_accuracy(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x,y = batch['imu'].float(),batch["labels"].long()
        if self.imagebind:
            x = get_imagebind_input(x, self.device)
        y_hat = self(x)
        acc = self.val_accuracy(y_hat, y)
        self.test_step_yhat.append(y_hat.cpu())
        self.test_step_y.append(y.cpu())
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_test_epoch_end(self):
        y_hat = torch.cat(self.test_step_yhat)
        y = torch.cat(self.test_step_y)
        epoch_acc = self.val_accuracy(y_hat, y)
        self.test_step_yhat = []
        self.test_step_y = []
        return epoch_acc.item()

    def configure_optimizers(self):
        if self.finetune:
            optimizer = AdamW(list(self.imu_encoder.parameters()) + list(self.clf.parameters()), lr=1e-4, weight_decay=1e-4)
        else:
            optimizer = AdamW(self.clf.parameters(), lr=1e-4, weight_decay=1e-4)
        # optimizer = SGD(self.clf.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps=np.max([self.T_max//100, 10]), T_max=self.T_max)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }