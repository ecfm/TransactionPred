# Training Module

This module contains the core components for training and evaluating machine learning models, with a focus on hyperparameter optimization using Optuna and experiment tracking with MLflow.

## Contents

1. [Grid Search (`grid_search.py`)](#grid-search)
2. [Loss Functions (`losses.py`)](#loss-functions)
3. [Training and Evaluation (`train.py`)](#training-and-evaluation)
4. [Training Configuration (`train_config.py`)](#training-configuration)
5. [Main Configuration (`config.yaml`)](#main-configuration)

## Grid Search (`grid_search.py`)

This script implements hyperparameter optimization using Optuna and MLflow for experiment tracking.

### Key Features:
- Hyperparameter sampling based on configuration
- Integration with MLflow for logging parameters, metrics, and artifacts
- Support for both local and database storage for Optuna and MLflow
- Git information logging
- Automatic adjustment of model parameters (e.g., `d_model` and `nhead`)

### Usage:
Run the script using Hydra for configuration management:
```
python grid_search.py
```

## Loss Functions (`losses.py`)

This file defines custom loss functions using PyTorch's `nn.Module`.

### Available Loss Functions:
- `NoOpLoss`: A placeholder loss that always returns 0
- `L1Loss`: L1 (Mean Absolute Error) loss
- `L2Loss`: L2 (Mean Squared Error) loss

Loss functions are registered using a custom registry for easy selection in configuration files.

## Training and Evaluation (`train.py`)

This script contains the main training and evaluation loop for models.

### Key Features:
- Support for custom optimizers and learning rate schedulers
- Early stopping
- Evaluation on validation and test sets
- Integration with MLflow for logging metrics

### Main Functions:
- `train_and_evaluate`: Trains the model with early stopping and evaluates on the test set
- `evaluation_loop`: Performs a single evaluation pass over a dataset
- `evaluate_model`: Evaluates the model on a given dataset

## Training Configuration (`train_config.py`)

This file defines the structure for training configuration using Pydantic models.

### Configuration Classes:
- `OptimizerConfig`: Specifies the optimizer name and its parameters
- `SchedulerConfig`: Specifies the learning rate scheduler name and its parameters
- `TrainConfig`: Main configuration class for training, including:
  - Number of epochs
  - Early stopping patience
  - Optimizer configuration
  - Scheduler configuration
  - Loss function name

### Usage:
The `TrainConfig` class can be used to validate and structure the training configuration in your main scripts or configuration files.

## Main Configuration (`config.yaml`)

This YAML file serves as the main configuration file for the entire training module, utilizing Hydra's composition feature.

### Key Components:
- Default configurations for various aspects of the training process
- Model-specific configurations (e.g., transformer model and training settings)
- Storage configuration
- Experiment name and number of trials for hyperparameter optimization

### Structure:
```yaml
defaults:
  - _self_
  - data: test
  - logging/console_only@_global_
  - models/transformer/hyperparams_small@_global_
  - models/transformer/model
  - models/transformer/train
  - storage/local@_global_

trial:
  model: ${models.transformer.model}
  train: ${models.transformer.train}

experiment: "test_transformer_hyperparams_small"
n_trials: 2
```

This configuration file allows for easy customization of different aspects of the training process, including data, logging, model architecture, and hyperparameters.

## Configuration

The training module uses Hydra for configuration management. The main configuration file is `config.yaml`, which composes various sub-configurations. The `train_config.py` file provides the structure for the training-specific configuration.

## Dependencies

- PyTorch
- Optuna
- MLflow
- Hydra
- NumPy
- GitPython
- Pydantic

## Getting Started

1. Set up your main configuration file (`config.yaml`) in the `conf` directory, customizing it as needed.
2. Ensure that all referenced sub-configurations (e.g., data, logging, model) are properly set up.
3. Prepare your data and model.
4. Run the grid search script:
   ```
   python grid_search.py
   ```
5. Monitor the training progress and results using MLflow.
