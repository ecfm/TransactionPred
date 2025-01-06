import pytest
import pandas as pd
from src.config.data_config import (
    DataContextConfig, SplitConfig, SequenceConfig, 
    FeatureProcessorConfig, CutoffConfig, DataLoaderConfig
)
from src.data.data_context import DataContext
from src.data.feature_processors import BrandToIdProcessor

# Fixture for sample training data
@pytest.fixture
def sample_train_data():
    return pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 4],
        'amount': [100, 150, 200, 300, 350, 400, 450, 500],
        'brand': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'D'],
        'date': pd.to_datetime([
            '2021-06-03', '2021-06-03', '2021-06-03', '2021-06-01', 
            '2021-07-01', '2021-07-08', '2021-08-01', '2021-08-01'
        ])
    })

# Fixture for sample test data
@pytest.fixture
def sample_test_data():
    return pd.DataFrame({
        'user_id': [1, 1, 2],
        'amount': [100, 150, 200],
        'brand': ['A', 'F', 'E'],
        'date': pd.to_datetime(['2021-06-03', '2021-06-03', '2021-06-03'])
    })


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
            type='multi_hot_separate',
            features=["date", "brand", "amount"]
        ),
        feature_processors={
            "amount": FeatureProcessorConfig(
                type="min_max_scaler"
            ),
            "brand": FeatureProcessorConfig(
                type="brand_to_id",
                params={"top_n": 10} 
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

@pytest.fixture
def data_context_config_without_brand_filter():
    return DataContextConfig(
        time_interval="7D",
        file_path="",
        splits=SplitConfig(train=0.7, val=0.15, test=0.15, overlap=0.5),
        input=SequenceConfig(
            type="continuous_time",
            features=["date", "brand", "amount"]
        ),
        output=SequenceConfig(
            type='multi_hot_separate',
            features=["date", "brand", "amount"]
        ),
        feature_processors={
            "brand": FeatureProcessorConfig(
                type="brand_to_id"
                # No filters applied here
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

def test_wrong_brand_to_id_processor_config():
    with pytest.raises(ValueError, match="Only one of 'top_n' and 'freq_threshold' can be set for 'brand_to_id' processor"):
        _ = DataContextConfig(
            time_interval="7D",
            file_path="", 
            splits=SplitConfig(train=0.7, val=0.15, test=0.15, overlap=0.5),
            input=SequenceConfig(
                type="continuous_time",
                features=["date", "brand", "amount"]
            ),
            output=SequenceConfig(
                type='multi_hot_separate',
                features=["date", "brand", "amount"]
            ),
            feature_processors={
                "amount": FeatureProcessorConfig(
                    type="min_max_scaler"
                ),
                "brand": FeatureProcessorConfig(
                    type="brand_to_id",
                    params={"top_n": 10, "freq_threshold": 0.1 }  # Both set, should raise error
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
        


def test_BrandToId_initialization(data_context_config):
    data_context = DataContext(data_context_config)
    
    brand_to_id_processor = data_context.feature_processors["brand"]
    
    assert brand_to_id_processor.value_to_id == {'<SOS>': 0, '<EOS>': 1, '<UNK>': 2}, "Initial value_to_id mapping is incorrect"
    assert brand_to_id_processor.sos_token == 0, "SOS token should be 0"
    assert brand_to_id_processor.eos_token == 1, "EOS token should be 1"
    assert brand_to_id_processor.unk_token == 2, "UNK token should be 2"
    assert brand_to_id_processor.top_n == 10, "'top_brands' filter should be set with top_n=10"
    assert brand_to_id_processor.freq_threshold is None, "'freq_threshold' should be None when 'top_n' is set"

def test_brand_to_id_processor_freq_threshold():
    df = pd.DataFrame({
        'brand': ['A', 'A', 'A', 'B', 'B', 'C', 'D', 'E', 'E', 'E', 'E'],
        'value': [10, 15, 10, 20, 25, 30, 40, 50, 50, 60, 70]
    })

    params = {"freq_threshold": 0.3}  # 30% threshold
    processor = BrandToIdProcessor(params)
    processor.fit(df, 'brand')

    transformed = processor.transform(df, 'brand')

    expected_result = pd.Series([2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3], name='brand')
    pd.testing.assert_series_equal(transformed, expected_result, check_dtype=False)


def test_brand_to_id_processor_top_n_2(sample_train_data):
    expected_transform_result = pd.Series([3, 3, 4, 4, 2, 2, 3, 2], name='brand')
    expected_inverse_transform_result = pd.Series(['A', 'A', 'B', 'B', "<UNK>", "<UNK>", "A", "<UNK>"], name=None)

    params = {"top_n": 2}
    processor = BrandToIdProcessor(params)
    processor.fit(sample_train_data, 'brand')

    transformed = processor.transform(sample_train_data, 'brand')
    inverse_transformed = processor.inverse_transform(transformed)

    pd.testing.assert_series_equal(transformed, expected_transform_result, check_dtype=False)
    pd.testing.assert_series_equal(inverse_transformed, expected_inverse_transform_result, check_dtype=False)


def test_brand_to_id_processor_top_n_10(sample_train_data):
    expected_transform_result = pd.Series([3, 3, 4, 4, 5, 5, 3, 6], name='brand')
    expected_inverse_transform_result = pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'A', 'D'], name=None)

    params = {"top_n": 10}
    processor = BrandToIdProcessor(params)
    processor.fit(sample_train_data, 'brand')

    transformed = processor.transform(sample_train_data, 'brand')
    inverse_transformed = processor.inverse_transform(transformed)

    pd.testing.assert_series_equal(transformed, expected_transform_result, check_dtype=False)
    pd.testing.assert_series_equal(inverse_transformed, expected_inverse_transform_result, check_dtype=False)


def test_brand_to_id_processor_test_data(sample_train_data, sample_test_data):
    expected_transform_test_result = pd.Series([3, 2, 2], name='brand')
    expected_inverse_transform_test_result = pd.Series(['A', "<UNK>", "<UNK>"], name=None)

    params = {"top_n": 2}
    processor = BrandToIdProcessor(params)
    processor.fit(sample_train_data, 'brand')

    transform_test_data = processor.transform(sample_test_data, 'brand')
    inverse_transformed_test_data = processor.inverse_transform(transform_test_data)

    pd.testing.assert_series_equal(transform_test_data, expected_transform_test_result, check_dtype=False)
    pd.testing.assert_series_equal(inverse_transformed_test_data, expected_inverse_transform_test_result, check_dtype=False)


def test_BrandToIdProcessor_without_filter(data_context_config_without_brand_filter):
    data_context = DataContext(data_context_config_without_brand_filter)
    
    brand_to_id_processor = data_context.feature_processors["brand"]
    
    assert brand_to_id_processor.top_n is None, "top_n should be None when no filter is applied"
    assert brand_to_id_processor.freq_threshold is None, "freq_threshold should be None when no filter is applied"
    
    input_df = pd.DataFrame({
        'brand': ['A', 'B', 'C', 'A']
    })
    
    expected_output = pd.Series([3, 4, 5, 3], name='brand')
    
    brand_to_id_processor.fit(input_df, 'brand')
    transformed_output = brand_to_id_processor.transform(input_df, 'brand')
    
    pd.testing.assert_series_equal(transformed_output, expected_output, check_dtype=False)

