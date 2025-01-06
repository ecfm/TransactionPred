#Write tests to ensure DataContext is properly initialized with various configurations.
#Check if all components (feature processors, filters, sequence generators) are correctly set up. 
# Use pytest mock objects to simulate different configurations.

import pytest
from src.data.data_context import DataContext
from src.config.data_config import DataContextConfig, SequenceConfig,SplitConfig, FeatureProcessorConfig, CutoffConfig, DataLoaderConfig
from src.data.sequence_generators import ContinuousTimeSequenceGenerator, SeparateMultiHotSequenceGenerator
from src.data.feature_processors import MinMaxFeatureProcessor, BrandToIdProcessor, TimeDeltaEncoder 

@pytest.fixture
def data_context_config():
    return DataContextConfig(
        time_interval="7D",
        file_path="",
        splits=SplitConfig(train=0.7, val=0.15, test=0.15, overlap=0.5),
        input=SequenceConfig(
            type="continuous_time",
            features=["date", "brand", "amount"]
        ),
        output=SequenceConfig(
            type="multi_hot_separate",
            features=["date", "brand", "amount"]
        ),
        feature_processors={
            "amount": FeatureProcessorConfig(
                type="no_op",
                params={"name": "amount"} 
            ),
            "amount": FeatureProcessorConfig(
                type="min_max_scaler"
            ),
            "brand": FeatureProcessorConfig(
                type="no_op",
                params={"name": "brand"} 
            ),
            "brand": FeatureProcessorConfig(
                type="brand_to_id"
            ),
            "date": FeatureProcessorConfig(
                type="time_delta",
            )
        },
        data_loader=DataLoaderConfig(),
        cutoffs=CutoffConfig(
            in_start="2021-01-01",
            train={"target_start": "2021-06-04"},
            val={"target_start": "2021-07-01"},
            test={"target_start": "2021-08-01"}
        )
    )


def test_data_context_initialization(data_context_config):
    # Initialize DataContext with the provided config
    data_context = DataContext(data_context_config)

    # Check if the config is correctly set
    assert data_context.config == data_context_config, "Config is not correctly set."

    # Check if feature_processors are correctly initialized
    assert "amount" in data_context.feature_processors
    assert isinstance(data_context.feature_processors["amount"], MinMaxFeatureProcessor), \
        f"Expected 'amount' processor to be an instance of MinMaxFeatureProcessor, got {type(data_context.feature_processors['amount'])} instead."

    assert "brand" in data_context.feature_processors
    assert isinstance(data_context.feature_processors["brand"], BrandToIdProcessor), \
        f"Expected 'brand' processor to be an instance of BrandToIdProcessor, got {type(data_context.feature_processors['brand'])} instead."

    assert "date" in data_context.feature_processors
    assert isinstance(data_context.feature_processors["date"], TimeDeltaEncoder), \
        f"Expected 'date' processor to be an instance of TimeDeltaProcessor, got {type(data_context.feature_processors['date'])} instead."

    # Check if input_generator is an instance of ContinuousTimeSequenceGenerator
    assert isinstance(data_context.input_generator, ContinuousTimeSequenceGenerator), \
        f"Expected input_generator to be an instance of ContinuousTimeSequenceGenerator, got {type(data_context.input_generator)} instead."

    # Check if target_generator is an instance of SeparateMultiHotSequenceGenerator
    assert isinstance(data_context.target_generator, SeparateMultiHotSequenceGenerator), \
        f"Expected target_generator to be an instance of SeparateMultiHotSequenceGenerator, got {type(data_context.target_generator)} instead."

    # Check if train_overlap_users and data_loaders are initialized to None
    assert data_context.train_overlap_users is None, "train_overlap_users should be initialized to None."
    assert data_context.data_loaders is None, "data_loaders should be initialized to None."

    # Check if cached_data_handler is correctly initialized
    assert data_context.cached_data_handler is not None, "cached_data_handler should be initialized."