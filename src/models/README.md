# Models Module

This module contains the implementation of neural network models and their components for a machine learning project.

## Contents

- [Models Module](#models-module)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Files](#files)
  - [Key Components](#key-components)
    - [Encoders and Decoders](#encoders-and-decoders)
    - [Models](#models)
    - [Positional Encoding](#positional-encoding)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Customization](#customization)

## Overview

The `src.models` module provides a flexible framework for building and using neural network models, particularly focused on transformer-based architectures. It includes implementations for encoders, decoders, the main model structure, and configuration classes.

## Files

- `encoders_decoders.py`: Contains implementations of various encoder and decoder modules.
- `models.py`: Implements the main model classes and utility functions.
- `model_config.py`: Defines configuration classes for models and their components.

## Key Components

### Encoders and Decoders

The module supports various types of encoders and decoders:

- `NoOpCoder`: A pass-through coder that doesn't modify the input.
- `EmbeddingCoder`: Uses nn.Embedding for encoding.
- `LinearCoder`: Applies a linear transformation with optional activation.
- `MLPCoder`: Implements a Multi-Layer Perceptron.

These are registered in a `CoderRegistry` for easy access and extensibility.

### Models

The main model implementation is based on the `NeuralNetModelBase` class, which is extended by specific model types:

- `TransformerEncoderModel`: Implements a transformer encoder-based model.

Models are created using the `create_model` function, which uses a `ModelRegistry` to instantiate the appropriate model based on the configuration.

### Positional Encoding

The module supports different types of positional encoding:

- Sinusoidal positional encoding
- Learned positional encoding

## Configuration

The `model_config.py` file defines Pydantic models for configuring the neural network components:

- `BaseCoderConfig`: Base configuration for encoders and decoders.
- `LinearCoderConfig`: Configuration for linear coders.
- `MLPCoderConfig`: Configuration for MLP coders.
- `EmbeddingCoderConfig`: Configuration for embedding coders.
- `NoOpCoderConfig`: Configuration for no-op coders.
- `TransformerConfig`: Configuration for transformer models.
- `NeuralNetModelConfig`: Main configuration class for the neural network model.

These configuration classes allow for type-safe and validated configuration of the model components.

## Usage

To use a model from this module:

1. Create a configuration object (`NeuralNetModelConfig`) with the desired parameters:

```python
from src.config.model_config import NeuralNetModelConfig, LinearCoderConfig, TransformerConfig

config = NeuralNetModelConfig(
    type="transformer-encoder",
    d_model=512,
    encoders={
        "feature1": LinearCoderConfig(type="linear", input_dim=10, output_dim=64),
        # ... other encoder configs
    },
    decoder=LinearCoderConfig(type="linear", input_dim=512, output_dim=10),
    concat_layer=LinearCoderConfig(type="linear", input_dim=128, output_dim=512),
    max_seq_length=100,
    transformer=TransformerConfig(nhead=8, num_encoder_layers=6, dropout=0.1),
    positional_encoding="sinusoidal",
    dropout=0.1
)
```

2. Prepare your data context.
3. Use the `create_model` function to instantiate your model:

```python
from src.models.models import create_model

model = create_model(config, data_context)
```

4. Use the model for forward passes:

```python
output = model(inputs)
```

## Customization

To add new encoder/decoder types:

1. Implement your new coder class.
2. Decorate it with `@CoderRegistry.register("your_coder_name")`.
3. Create a corresponding configuration class in `model_config.py`.

To add new model types:

1. Implement your model class, inheriting from `NeuralNetModelBase`.
2. Decorate it with `@ModelRegistry.register("your_model_name")`.
3. Update the `NeuralNetModelConfig` class if necessary to include new parameters.

Remember to update the corresponding configuration classes in `src.config.model_config` when adding new components or parameters.