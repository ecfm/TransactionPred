# Data Module

This data module is designed to handle the preprocessing, caching, and generation of sequence data for machine learning tasks, particularly focused on transaction data.

## Table of Contents

- [Data Module](#data-module)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Components](#key-components)
  - [Main Features](#main-features)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Extending the Module](#extending-the-module)

## Overview

The data module provides a flexible and efficient way to process raw transaction data into sequence format suitable for machine learning models. It includes features for data filtering, feature processing, sequence generation, and caching of processed data for improved performance.

## Key Components

1. **DataContext**: The main class that orchestrates the data processing pipeline.
2. **FeatureProcessors**: Classes for processing individual features (e.g., MinMaxScaler, BrandToIdProcessor).
3. **Filters**: Classes for filtering raw data (e.g., TopBrandsFilter).
4. **SequenceGenerators**: Classes for generating different types of sequences from processed data.
5. **CachedDataHandler**: A class for handling the caching of processed data.

## Main Features

- **Flexible Data Processing**: Supports various feature processing techniques and data filters.
- **Sequence Generation**: Multiple sequence generation strategies for different model requirements.
- **Data Caching**: Efficient caching of processed data to speed up subsequent runs.
- **Configurable Pipeline**: Easily configurable data processing pipeline through configuration objects.
- **Extensible Architecture**: Utilizes a registry pattern for easy addition of new feature processors, filters, and sequence generators.

## Usage

To use this data module in your project:

1. Initialize a `DataContextConfig` object with your desired settings.
2. Create a `DataContext` instance with the config.
3. Call `prepare_data()` with the path to your raw data file.
4. Use the `get_dataloader()` method to get DataLoader objects for training, validation, and testing.

Example:

```python
from src.config.data_config import DataContextConfig
from src.data.data_context import DataContext

config = DataContextConfig(...)  # Initialize with your settings
data_context = DataContext(config)
data_context.prepare_data('path/to/your/data.csv')

train_loader = data_context.get_dataloader('train')
val_loader = data_context.get_dataloader('val')
test_loader = data_context.get_dataloader('test')
```

## Configuration

The data module is highly configurable. Key configuration options include:

- **Feature Processors**: Specify how each feature should be processed.
- **Filters**: Define filters to apply to the raw data.
- **Sequence Generation**: Choose the type of sequence generator and its parameters.
- **Data Splitting**: Configure how data is split into train, validation, and test sets.
- **Caching**: Enable or disable caching of processed data.

Refer to the `DataContextConfig` class in `src/config/data_config.py` for all available configuration options.

## Extending the Module

To extend the functionality of this module:

1. **Add New Feature Processors**: Create a new class inheriting from `BaseFeatureProcessor` and register it with the `FeatureProcessorRegistry`.
2. **Add New Filters**: Create a new class inheriting from `Filter` and register it with the `FilterRegistry`.
3. **Add New Sequence Generators**: Create a new class inheriting from `SequenceGenerator` and register it with the `SequenceGeneratorRegistry`.
4. **Update Configuration**: Modify the `src/config/data_config.py` file to include any new configuration options for your extensions.

Example of adding a new feature processor:

```python
@FeatureProcessorRegistry.register('my_new_processor')
class MyNewProcessor(BaseFeatureProcessor):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # Initialize your processor

    def fit(self, df: pd.DataFrame, feature: str):
        # Implement fitting logic

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        # Implement transformation logic

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        # Implement inverse transformation logic

    def get_eos_token(self):
        # Return end-of-sequence token
```

After adding new components:

1. Use them in your configuration by referencing their registered names in `src/config/data_config.py`.
2. Update the `FeatureProcessorConfig`, `FilterConfig`, or `SequenceConfig` classes in `src/config/data_config.py` if your new components require additional configuration options.

Remember that the data module files are located in `src/data/`, while the configuration file is in `src/config/`. When extending the module, you may need to modify files in both locations to fully integrate your new components.