import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model.mae import FactorizedAttentionViT
from utils.misc import EarlyStopping


class AuxiliaryMAE:
    def __init__(self, hps, train_hps, train_loader, val_loader, device):
        self._hps = hps
        self._train_hps = train_hps
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._init_hyperparameters()

        self.model = FactorizedAttentionViT(hps)

    def _init_hyperparameters(self):
        self.loss_hps = self._train_hps.loss
        self.optimizer_hps = self._train_hps.optimizer
        self.early_stopping_hps = getattr(self._train_hps, "early_stopping", None)
        self.scheduler_hps = getattr(self._train_hps, "scheduler", None)

        optimizer_params = self.optimizer_hps.params
        self.lr = float(optimizer_params.lr)
        self.betas = tuple(map(float, optimizer_params.betas))
        self.weight_decay = float(optimizer_params.weight_decay)

        if self.early_stopping_hps is not None:
            self.patience = int(self.early_stopping_hps.patience)
            self.tol = float(self.early_stopping_hps.tol)

        if self.scheduler_hps is not None:
            self.warmup_epochs = int(self.scheduler_hps.warmup_epochs)

        self.num_epochs = int(self._train_hps.num_epochs)
        self.checkpoints_dir = str(self._train_hps.checkpoints_dir)
        self.checkpoints_freq = int(self._train_hps.checkpoints_freq)

    def _init_training_scheme(self):
        optim = Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,  # type: ignore
            weight_decay=self.weight_decay,
        )

        scheduler = None
        if self.scheduler_hps is not None:
            num_training_steps = self.num_epochs * len(self._train_loader)
            num_warmup_steps = self.warmup_epochs * len(self._train_loader)
            scheduler = get_cosine_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        early_stopping = None
        if self.early_stopping_hps is not None:
            early_stopping = EarlyStopping(self.patience, self.tol)

        return optim, scheduler, early_stopping

    def _freeze_weights(self, m):
        for m in m.parameters():
            m.requires_grad = False

    def _init_weights(self):
        self.model.init_weights()

    def _loss_fn(self, x_pred_patches, x_patches, mask):
        B, C, N, D = x_pred_patches.shape
        loss = F.l1_loss(x_pred_patches, x_patches, reduction="none")
        loss = torch.mean(loss, dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss

    def train(self):
        optim, scheduler, early_stopping = self._init_training_scheme()

        self.model = self.model.to(self._device)
        self._init_weights()

        os.makedirs(self.checkpoints_dir, exist_ok=True)

        best_model = None
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            train_loss = 0.0
            for x, labels in tqdm(
                self._train_loader,
                desc=f"Epoch {epoch}/{self.num_epochs} - Training",
                leave=False,
            ):
                x, labels = x.to(self._device), labels.to(self._device)

                x_pred, x_pred_patches, x_patches, mask = self.model(x)
                loss = self._loss_fn(x_pred_patches, x_patches, mask)

                optim.zero_grad()
                loss.backward()
                optim.step()

                if scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()

            with torch.no_grad():
                self.model.eval()

                val_loss = 0.0
                for x, labels in tqdm(
                    self._val_loader,
                    desc=f"Epoch {epoch}/{self.num_epochs} - Validation",
                    leave=False,
                ):
                    x, labels = x.to(self._device), labels.to(self._device)

                    x_pred, x_pred_patches, x_patches, mask = self.model(x)
                    loss = self._loss_fn(x_pred_patches, x_patches, mask)

                    val_loss += loss.item()

            train_loss /= len(self._train_loader)
            val_loss /= len(self._val_loader)

            print(f"----Epcoh {epoch}----")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"LR: {optim.param_groups[0]['lr']:.6f}")

            if epoch % self.checkpoints_freq == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "encoder_state_dict": self.model.encoder.state_dict(),
                        "decoder_state_dict": self.model.decoder.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    f"{self.checkpoints_dir}/checkpoint_{epoch}.pth",
                )

            if early_stopping is not None:
                best_model = early_stopping.step(self.model, val_loss)
                if early_stopping.stop:
                    print("Early stopping triggered.")
                    break

        if best_model is None:
            best_model = self.model

        torch.save(
            {"model_state_dict": best_model.state_dict()},
            f"{self.checkpoints_dir}/best_model.pth",
        )

        return best_model
