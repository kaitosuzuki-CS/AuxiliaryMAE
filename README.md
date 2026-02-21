# AuxiliaryMAE

## Introduction

AuxiliaryMAE is a PyTorch implementation of a Masked Autoencoder (MAE) utilizing a Factorized Attention Vision Transformer (ViT) architecture. This project is designed for self-supervised learning on multi-channel vision data, employing a high masking ratio strategy to learn robust feature representations.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Application Info](#application-info)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Files](#project-files)

## Project Overview

The core of this project is the `FactorizedAttentionViT`, which splits the image into patches, masks a significant portion of them (default 75%), and trains the model to reconstruct the missing pixels. This approach encourages the model to learn high-level semantic understanding of the visual data.

Key features:

- **Masked Autoencoder (MAE)** framework.
- **Factorized Attention** mechanism in the Vision Transformer.
- Configurable Encoder/Decoder architecture via YAML files.
- Training pipeline with early stopping, checkpointing, and learning rate scheduling.

## Project Structure

```
AuxiliaryMAE/
├── configs/                 # Configuration files
│   ├── model_config.yml     # Model hyperparameters (layers, dim, heads, etc.)
│   └── train_config.yml     # Training hyperparameters (epochs, lr, batch size)
├── model/                   # Model source code
│   ├── mae.py               # Main FactorizedAttentionViT model and masking logic
│   ├── blocks/              # Encoder/Decoder blocks
│   ├── components/          # Embeddings and helper layers
│   ├── layers/              # Layer definitions
│   └── models/              # Encoder and Decoder implementations
├── utils/                   # Utility scripts
│   ├── dataset.py           # Dataset loading and processing
│   └── misc.py              # Miscellaneous helpers (loading configs, seeding)
├── main.py                  # Entry point for the application
├── train.py                 # Training engine (AuxiliaryMAE class)
├── environment.yml          # Conda environment definition
└── requirements.txt         # Pip requirements
```

## Tech Stack

- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **Libraries:**
  - `transformers` (for schedulers)
  - `numpy`
  - `pyyaml` (configuration)

## Application Info

The model operates by:

1.  **Patch Embeddings:** Splitting input images into fixed-size patches.
2.  **Random Masking:** Randomly masking a large percentage of these patches (e.g., 75%).
3.  **Encoding:** Processing only the visible patches through a Factorized Attention Encoder.
4.  **Decoding:** Reconstructing the original image patches from the latent representation and mask tokens.
5.  **Loss Calculation:** Computing the L1 loss between the predicted and original pixel values for the masked patches.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kaitosuzuki-CS/AuxiliaryMAE.git
    cd AuxiliaryMAE
    ```

2.  **Set up the environment:**

    **Option A: Using Conda (Recommended)**

    ```bash
    conda env create -f environment.yml
    conda activate auxiliarymae
    ```

    **Option B: Using Pip**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

To start training the model, run the `main.py` script. You can specify custom configuration files if needed.

```bash
python main.py --model-config-path configs/model_config.yml --train-config-path configs/train_config.yml
```

**Note:** Ensure your `train_config.yml` or the loading logic in `utils/dataset.py` points to your actual dataset paths.

## Project Files

- **`main.py`**: The CLI entry point. It parses arguments, loads configurations, and initiates the training process.
- **`train.py`**: Contains the `AuxiliaryMAE` class, which manages the training loop, validation, loss calculation, optimization, and checkpointing.
- **`model/mae.py`**: Defines the `FactorizedAttentionViT` class. This file contains the logic for random masking (`random_masking`), the forward pass for the encoder (`forward_encoder`), and the decoder (`forward_decoder`).
- **`model/blocks/`**: Contains the building blocks for the encoder and decoder, such as `encoder_block.py` and `decoder_block.py`.
- **`model/components/`**: Houses shared components like embedding layers (`embeddings.py`) and other foundational layers (`layers.py`).
- **`model/layers/`**: Defines specific transformer layers, including `encoder_layer.py` and `decoder_layer.py`.
- **`model/models/`**: Contains the top-level `Encoder` and `Decoder` module implementations.
- **`utils/dataset.py`**: Handles data loading and dataset creation. You may need to modify this or your config to point to your specific data source.
- **`utils/misc.py`**: Provides miscellaneous utility functions, such as `load_config` for YAML files and `set_seeds` for reproducibility.
- **`configs/model_config.yml`**: Defines the structural parameters of the model (embedding dimensions, number of heads, depth, patch size).
- **`configs/train_config.yml`**: Controls training parameters like learning rate, batch size, epochs, and optimization settings.
