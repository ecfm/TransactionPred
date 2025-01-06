# Config Module

This module provides a flexible and robust configuration management system for machine learning projects. It uses Pydantic models to define and validate configuration schemas, ensuring type safety and consistency across your project.

## Structure

The config module is structured as follows:

```
src/
└── config/
conf/
```

## Files

1. `data_config.py`: Defines configuration schemas for data processing and loading.
2. `grid_search_config.py`: Contains configuration schemas for hyperparameter tuning and experiment tracking.
3. `model_config.py`: Specifies configuration schemas for neural network models and their components.
4. `train_config.py`: Defines configuration schemas for training parameters.
5. `config_manager.py`: Provides utility functions for loading, validating, and managing configurations.

## Key Features

- **Type Safety**: All configuration schemas are defined using Pydantic models, ensuring type checking and validation.
- **Modular Design**: Separate configuration files for different aspects of the ML pipeline (data, model, training, etc.).
- **Hyperparameter Tuning**: Built-in support for defining hyperparameter search spaces.
- **Experiment Tracking**: Integration with MLflow for logging and tracking experiments.
- **Flexible Model Architecture**: Supports various encoder types and transformer configurations.
- **Logging**: Configurable logging setup.

## Configuration Files

The project uses YAML configuration files located in the `conf` directory. These files allow for easy modification of various aspects of the project without changing the code.

### Main Configuration File

The main configuration file is `conf/config.yaml`. It uses Hydra's composition feature to include other configuration files:

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

This structure allows for easy swapping of different configurations (e.g., different data settings, model architectures, or training parameters) by changing the included files or overriding specific values.

## Usage

To use the config module in your project:

1. Define your configuration in the appropriate YAML files in the `conf` directory.
2. In your Python code, use Hydra to load the configuration:

```python
import hydra
from omegaconf import DictConfig
from src.config.grid_search_config import GridSearchConfig
from src.config.config_manager import ConfigManager

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    config = ConfigManager.load_and_validate_config(cfg, GridSearchConfig)
    # Your code here, using the loaded and validated config
```

3. Run your script, and Hydra will automatically compose the configuration based on the YAML files.

## Configuration Classes

### DataContextConfig
Defines the data processing pipeline, including:
- Feature processors
- Input and output sequence configurations
- Data splitting and filtering
- Cutoff dates for train/val/test sets

### GridSearchConfig
Specifies the hyperparameter tuning process, including:
- Number of trials
- Hyperparameter search spaces
- MLflow and Optuna configurations

### NeuralNetModelConfig
Defines the neural network architecture, including:
- Encoder and decoder configurations
- Transformer parameters
- Positional encoding and dropout

### TrainConfig
Specifies training parameters such as:
- Number of epochs
- Optimizer and scheduler settings
- Loss function

## Extending the Config Module

To add new configuration options:

1. Create a new YAML file in the appropriate subdirectory of `conf/`.
2. Update the main `config.yaml` file to include your new configuration file if necessary.
3. Create a new Pydantic model in the appropriate file (e.g., `model_config.py` for new model architectures) to validate the new configuration options.
4. Update the relevant parent models to include your new configuration options.
5. If necessary, update the `ConfigManager` to handle any special loading or validation requirements for your new configuration.

## Best Practices

- Keep configuration files separate from implementation code.
- Use Hydra's composition feature to create modular and reusable configurations.
- Use type hints and Pydantic's validation features to catch configuration errors early.
- Leverage the hierarchical nature of the configuration system to create reusable sub-configurations.
- Use the `ConfigManager` to ensure all default values are properly applied and validated.
- When adding new features or modules, always update the corresponding configuration files and Pydantic models.