import argparse
import os

import torch

from train import AuxiliaryMAE
from utils.dataset import create_dataset
from utils.misc import load_config, set_seeds

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Factorized Attention ViT with AuxiliaryMAE"
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/model_config.yml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--train-config-path",
        type=str,
        default="configs/train_config.yml",
        help="Path to the training config file.",
    )

    args = parser.parse_args()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)
    set_seeds(train_hps)

    train_loader, val_loader = create_dataset(train_hps.data)  # type: ignore
    model = AuxiliaryMAE(hps, train_hps, train_loader, val_loader, device)

    model.train()
