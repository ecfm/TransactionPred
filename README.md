# Machine Learning Pipeline for Predicting Credit Card Transaction

This repository contains a comprehensive machine learning pipeline designed for processing and analyzing transaction data. Given the past spending history of millions of customers over 6 years on hundreds of thousands of brands. The goal is to predict the next time step's transactions (including spending amount and corresponding brands) of each customer given all the spending history. The main model is a customized [Transformer Encoder](src/models/models.py). During [training](src/training/train.py), each customer's spending history is [represented as a sequence](src/data/sequence_generators.py) and fed to the Transformer Encoder. The spending is predicted based on the last hidden-state of the encoder. The project is structured into several modules, each handling a specific stage of the machine learning workflow.

## Table of Contents

- [Machine Learning Pipeline for Predicting Credit Card Transaction](#machine-learning-pipeline-for-predicting-credit-card-transaction)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Key Components](#key-components)
    - [1. Config Module (`src/config/`)](#1-config-module-srcconfig)
    - [2. Data Module (`src/data/`)](#2-data-module-srcdata)
    - [3. Models Module (`src/models/`)](#3-models-module-srcmodels)
    - [4. Training Module (`src/training/`)](#4-training-module-srctraining)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Extending the Project](#extending-the-project)
    - [Adding New Feature Processors](#adding-new-feature-processors)
    - [Adding New Model Types](#adding-new-model-types)
    - [Adding New Loss Functions](#adding-new-loss-functions)
  - [Testing](#testing)

## Project Overview

This project implements a flexible and robust machine learning pipeline for transaction data analysis. It includes modules for data processing, model definition, training, and hyperparameter optimization using Optuna. The pipeline is designed to be easily configurable using Hydra and extensible through registry-based components.

## Directory Structure

```
├── README.md
├── conf/                  # Configuration files
│   ├── data/             # Data processing configs
│   ├── logging/          # Logging configs
│   ├── models/           # Model configs
│   └── storage/          # Storage configs
├── src/                  # Source code
│   ├── config/           # Configuration management
│   ├── data/             # Data processing
│   ├── models/           # Model definitions
│   ├── training/         # Training and evaluation
│   └── utils/            # Utility functions
├── tests/                # Unit tests
└── requirements.txt
```

## Key Components

### 1. Config Module (`src/config/`)
- Provides type-safe configuration management using Pydantic models
- Manages configurations for:
  - Data processing and loading (`data_config.py`)
  - Hyperparameter tuning (`grid_search_config.py`)
  - Model architecture (`model_config.py`)
  - Training parameters (`train_config.py`)
- Supports configuration validation and default values
- Integrates with Hydra for flexible configuration composition
- Includes logging setup and experiment tracking configurations

### 2. Data Module (`src/data/`)
- Implements sequence generators for different data types:
  - `ContinuousTimeSequenceGenerator`: Handles time-series data
  - `BrandOnlyMultiHotSequenceGenerator`: Processes brand-only sequences
  - `BrandAmountMultiHotSequenceGenerator`: Handles brand-amount pairs
  - `SeparateMultiHotSequenceGenerator`: Processes separate multi-hot sequences
- Feature processors for data transformation and caching
- Efficient data caching mechanisms

### 3. Models Module (`src/models/`)
- Implements neural network models with a focus on transformer architectures
- Provides modular encoder/decoder components:
  - `NoOpCoder`: Pass-through coder
  - `EmbeddingCoder`: Neural embeddings
  - `LinearCoder`: Linear transformations
  - `MLPCoder`: Multi-layer perceptron
  - `RTDLNumEmbeddingsProcessor`: Numerical embeddings
- Supports positional encoding:
  - Sinusoidal positional encoding
  - Learned positional encoding

### 4. Training Module (`src/training/`)
- Implements training and evaluation loops
- Grid search functionality for hyperparameter optimization
- Integration with MLflow for experiment tracking

## Configuration

The project uses Hydra for configuration management. Key configuration files include:

```yaml
defaults:
  - data: test              # Data processing settings
  - logging/console_only    # Logging configuration
  - models/transformer      # Model architecture settings
  - storage/local          # Storage settings
```

Example data configuration (`conf/data/default.yaml`):
```yaml
data:
  data_loader:
    batch_size: 32
    num_workers: 4
  splits:
    train: 0.7
    val: 0.15
    test: 0.15
    overlap: 0.2
  cutoffs:
    in_start: '2019-03-01'
    train:
      target_start: '2019-09-30'
    val:
      target_start: '2019-10-31'
    test:
      target_start: '2019-11-30'
```

Model configuration example (`conf/models/transformer/model.yaml`):
```yaml
model:
  type: transformer-encoder
  d_model: 256
  encoders:
    date:
      type: no_op
    brand:
      type: embedding
    amount:
      type: no_op
  transformer:
    nhead: 8
    num_encoder_layers: 6
    dropout: 0.1
  max_seq_length: 500
  positional_encoding: sinusoidal
```

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your experiment in the `conf/` directory

3. Run grid search for hyperparameter optimization:
   ```bash
   python src/training/grid_search.py
   ```

5. Monitor the training progress and results using MLflow.

## Extending the Project

### Adding New Feature Processors
1. Create a new class inheriting from `BaseFeatureProcessor` in `src/data/feature_processors.py`.
2. Register it with the `FeatureProcessorRegistry`.

### Adding New Model Types
1. Implement your model class in `src/models/models.py`, inheriting from `NeuralNetModelBase`.
2. Decorate it with `@ModelRegistry.register("your_model_name")`.
3. Update the `NeuralNetModelConfig` class in `src/config/model_config.py` if necessary.

### Adding New Loss Functions
1. Define your loss function in `src/training/losses.py`.
2. Register it using the custom loss registry.

## Testing

The project includes unit tests for each module. To run the tests:

```bash
python -m pytest tests/
```

For more detailed information about each module, please refer to the README files in their respective directories:
- [Config Module README](src/config/README.md)
- [Data Module README](src/data/README.md)
- [Models Module README](src/models/README.md)
- [Training Module README](src/training/README.md)
# TransactionPred
