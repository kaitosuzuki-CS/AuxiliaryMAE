# AuxiliaryMAE

## Introduction

AuxiliaryMAE is a research project implementing a Masked Autoencoder (MAE) with a Factorized Attention Vision Transformer (ViT) architecture. This project aims to provide a flexible and configurable framework for self-supervised learning on high-dimensional data, utilizing an encoder-decoder structure to reconstruct masked input patches.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Application Info](#application-info)
- [Getting Started](#getting-started)
- [Project Files](#project-files)

## Project Overview

The core of this project is the `FactorizedAttentionViT`, which splits the processing into an encoder and a decoder. The model employs a random masking strategy to mask a high percentage of the input data (default 75%) and tasks the model with reconstructing the missing parts. This approach encourages the model to learn robust and generalized representations of the data.

Key features include:

- **Masked Autoencoder Architecture**: Efficient self-supervised learning by reconstructing masked patches.
- **Factorized Attention ViT**: Specialized Transformer backbone.
- **Configurable Design**: extensive use of YAML configuration files for model hyperparameters and training settings.
- **Robust Training Loop**: Includes features like early stopping, learning rate scheduling (Cosine with Warmup), and automatic checkpointing.

## Project Structure

```
AuxiliaryMAE/
├── .gitignore
├── main.py
├── requirements.txt
├── train.py
├── configs/
│   ├── model_config.yml
│   └── train_config.yml
├── model/
│   ├── mae.py
│   ├── blocks/
│   ├── components/
│   ├── layers/
│   └── models/
└── utils/
    ├── dataset.py
    └── misc.py
```

## Tech Stack

- **Language**: Python 3.x
- **Deep Learning Framework**: PyTorch
- **Libraries**:
  - `transformers`: For optimization schedules.
  - `numpy`: Numerical operations.
  - `pyyaml`: Configuration management.

## Application Info

The application is designed as a Command Line Interface (CLI) tool. It uses `argparse` to handle user inputs for configuration paths. The training process creates checkpoints and logs loss metrics to the console.

## Getting Started

### Prerequisites

- Anaconda or Miniconda installed on your system.
- CUDA-enabled GPU is recommended for training.

### Installation with Conda

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kaitosuzuki-CS/AuxiliaryMAE.git
    cd AuxiliaryMAE
    ```

2.  **Create and activate the environment:**
    ```bash
    # Create a new environment and install dependencies from requirements.txt
    conda create -n <env_name> python=3.10 --file requirements.txt
    conda activate <env_name>
    ```

### Usage

To start training the model, run the `main.py` script. You can specify custom configuration files if needed.

```bash
# Run with default configurations
python main.py

# Run with specific config files
python main.py --model-config-path configs/model_config.yml --train-config-path configs/train_config.yml
```

## Project Files

- **`main.py`**: The entry point of the application. It parses command-line arguments, loads configurations, initializes the dataset and model, and starts the training process.
- **`train.py`**: Contains the `AuxiliaryMAE` class, which manages the training lifecycle. It handles the optimizer, scheduler, early stopping, training loop, validation loop, and saving checkpoints.
- **`model/mae.py`**: Defines the `FactorizedAttentionViT` class. This is the core model that orchestrates the encoder and decoder, performs random masking of the input, and calculates the reconstruction.
- **`configs/*.yml`**: YAML files defining the hyperparameters for the model architecture (`model_config.yml`) and the training process (`train_config.yml`).
- **`utils/dataset.py`**: Contains the logic for loading and preprocessing the dataset.
- **`utils/misc.py`**: Helper functions for loading configurations, setting random seeds, and other miscellaneous tasks.
