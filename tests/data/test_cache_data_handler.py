import pytest
import os
from src.data.data_context import DataContext
from src.config.data_config import (
    DataContextConfig,
    FeatureProcessorConfig,
    SequenceConfig,
    FilterConfig,
    SplitConfig,
    CutoffConfig,
    DataLoaderConfig
)
from src.data.cache_data_handler import CachedDataHandler
from unittest import mock
import hashlib
import json

@pytest.fixture
def sample_config():
    return DataContextConfig(
        time_interval="1H",
        file_path="/data/files/sample_data.csv",
        filters={
            "brand_filter": FilterConfig(top_n=5),
            "amount_filter": FilterConfig(freq_threshold=0.01)
        },
        splits=SplitConfig(
            train=0.7,
            val=0.2,
            test=0.1,
            overlap=0.05
        ),
        input=SequenceConfig(
            type="sequential",
            features=["user_id", "amount", "brand"],
            separate_brand_amount=True
        ),
        output=SequenceConfig(
            type="regression",
            features=["amount"]
        ),
        feature_processors={
            "brand_one_hot": FeatureProcessorConfig(
                type="one_hot",
                params={"columns": ["brand"]}
            ),
            "amount_scaler": FeatureProcessorConfig(
                type="minmax_scaler",
                params={"feature_range": (0, 1)}
            )
        },
        data_loader=DataLoaderConfig(
            batch_size=64,
            num_workers=4
        ),
        cutoffs=CutoffConfig(
            in_start="2022-01-01",
            train={"start": "2022-01-01", "end": "2022-06-30"},
            val={"start": "2022-07-01", "end": "2022-09-30"},
            test={"start": "2022-10-01", "end": "2022-12-31"}
        )
    )

@pytest.fixture
def data_context(sample_config):
    return DataContext(sample_config)

@pytest.fixture
def mock_cached_data_handler():
    handler = CachedDataHandler()
    # Mock the src_files_hash value
    handler.src_files_hash = "mocked_hash_value"
    return handler

@pytest.fixture
def mock_file_data():
    return b"This is a mock file content for testing."

def test_save_processed_data(mock_cached_data_handler, mocker):
    # Mock os.makedirs
    mock_makedirs = mocker.patch('os.makedirs')
    # Mock open to avoid actual file I/O
    mock_open = mocker.patch('builtins.open', mock.mock_open())
    # Mock pickle.dump to avoid actual serialization
    mock_pickle_dump = mocker.patch('pickle.dump')
    # Define a sample data dictionary
    data_to_save = {"key1": "value1"}
    # Call the method to be tested
    file_path = "/fake/path/to/save_data.pkl"
    mock_cached_data_handler.save_processed_data(file_path, data_to_save)
    # Check if os.makedirs was called with the correct parameters
    mock_makedirs.assert_called_once_with(os.path.dirname(file_path), exist_ok=True)
    # Check if the src_files_hash was added to the data
    assert data_to_save["src_files_hash"] == "mocked_hash_value"
    # Check if open was called with the correct parameters
    mock_open.assert_called_once_with(file_path, 'wb')
    # Check if pickle.dump was called with the correct parameters
    mock_pickle_dump.assert_called_once_with(data_to_save, mock_open())

def test_load_processed_data_cache_exists_but_modified(sample_config, mock_cached_data_handler, mocker):
    config = sample_config  
    # Mock the get_hashes method to return specific hashes
    mocker.patch.object(mock_cached_data_handler, 'get_hashes', return_value=("config_hash", "data_hash"))
    # Mock os.path.exists to simulate the existence of the cache file
    mocker.patch('os.path.exists', return_value=True)
    # Mock is_data_module_src_modified to simulate that the source code has been modified
    mocker.patch.object(mock_cached_data_handler, 'is_data_module_src_modified', return_value=True)
    # Call the method to be tested
    cached_data_path, loaded_data = mock_cached_data_handler.load_processed_data(config, "/fake/path/to/raw_data.csv")
    # Assertions
    mock_cached_data_handler.get_hashes.assert_called_once_with(config.dict(), "/fake/path/to/raw_data.csv")
    assert cached_data_path == os.path.join(mock_cached_data_handler.processed_data_path, "config_hash_data_hash.pkl")
    assert loaded_data is None

def test_get_hashes(sample_config, mock_cached_data_handler, mock_file_data, mocker):
    # Get hashes
    config = sample_config.dict()
    config_str = json.dumps(config, sort_keys=True)
    expected_config_hash = hashlib.md5(config_str.encode()).hexdigest()
    # mock open
    mocker.patch("builtins.open", mock.mock_open(read_data=mock_file_data))
    # calculate hashes
    hasher = hashlib.md5()
    hasher.update(mock_file_data)
    expected_data_hash = hasher.hexdigest()
    # test get_hashes
    actual_config_hash, actual_data_hash = mock_cached_data_handler.get_hashes(config, "fake_file_path")
    # assertions
    assert actual_config_hash == expected_config_hash, "Config hash does not match the expected value."
    assert actual_data_hash == expected_data_hash, "Data hash does not match the expected value."