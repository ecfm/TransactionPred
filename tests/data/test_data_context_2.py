# Test the prepare_data method of DataContext. 
# Verify that data is correctly loaded, processed, and split into train/val/test sets. 
# Check if the caching mechanism works correctly. 
# Use mock objects to simulate file I/O operations.

import pytest
import pandas as pd
from math import ceil, floor
import tempfile
import os
from src.data.data_context import DataContext
from src.config.data_config import DataContextConfig, SequenceConfig,SplitConfig, FeatureProcessorConfig, CutoffConfig, DataLoaderConfig
from src.data.cache_data_handler import CachedDataHandler



@pytest.fixture
def sample_data():
    data = {
        'user_id': [1, 1, 2, 2, 5, 6, 7, 8, 9, 10, 1],
        'date': [
            '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-03', 
            '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', 
            '2024-01-03', '2024-01-01', '2024-01-02'
        ],
        'brand': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A'],
        'amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 150]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df



@pytest.fixture
def sample_config():
    return DataContextConfig(
    time_interval="7D",
    file_path="",
    splits=SplitConfig(train=0.65, val=0.15, test=0.2, overlap=0.5),
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


def test_prepare_data_with_different_filepath_but_same_content(sample_config, mocker):
    """
    Test prepare_data with 2 CSV files that has different filepaths but the same content and config, 
    _process_and_save_data is called once
    """
    # Define the original CSV content as a string
    csv_content = """user_id,date,brand,amount
                     1,2024-09-01,B,100
                     2,2024-09-02,A,150
                     3,2024-09-03,C,200
                     4,2024-09-04,B,120
                     5,2024-09-05,A,130
                     6,2024-09-06,C,160
                     7,2024-09-07,B,140
                     8,2024-09-08,A,180
                     9,2024-09-09,C,110
                     10,2024-09-10,B,170"""
    
    # Create temporary CSV files
    tmp_file1 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tmp_file1.write(csv_content.encode('utf-8'))
        tmp_file_1_path = tmp_file1.name
        tmp_file1.close()

        tmp_file2.write(csv_content.encode('utf-8'))
        tmp_file_2_path = tmp_file2.name
        tmp_file2.close()

        # Calculate hashes and paths
        handler = CachedDataHandler()
        config_hash_1, data_hash_1 = handler.get_hashes(sample_config.dict(), tmp_file_1_path)
        cached_data_path_1 = os.path.join(handler.processed_data_path, f"{config_hash_1}_{data_hash_1}.pkl")

        config_hash_2, data_hash_2 = handler.get_hashes(sample_config.dict(), tmp_file_2_path)
        cached_data_path_2 = os.path.join(handler.processed_data_path, f"{config_hash_2}_{data_hash_2}.pkl")

        # Create DataContext instance
        dc = DataContext(sample_config)
    
        # Call prepare_data with the first file path
        dc.prepare_data(tmp_file1.name)

        # Patch the _process_and_save_data method
        mock_process = mocker.patch.object(dc, '_process_and_save_data')

        # Call prepare_data with the second file path
        dc.prepare_data(tmp_file2.name)

        # Assert that _process_and_save_data was not called with the second file
        mock_process.assert_not_called()

        # Assert that the cached data paths are the same
        assert cached_data_path_1 == cached_data_path_2

    finally:
        # Clean up: remove the temporary CSV files and cached data files
        if os.path.exists(tmp_file_1_path):
            os.remove(tmp_file_1_path)
        if os.path.exists(tmp_file_2_path):
            os.remove(tmp_file_2_path)
        if os.path.exists(cached_data_path_1):
            os.remove(cached_data_path_1)
        if os.path.exists(handler.processed_data_path) and not os.listdir(handler.processed_data_path):
            os.rmdir(handler.processed_data_path)
        data_path = os.path.dirname(handler.processed_data_path)  # Path to the 'data' folder
        if os.path.exists(data_path) and not os.listdir(data_path):
            os.rmdir(data_path)



def test_prepare_data_with_same_filepath_but_different_content(sample_config,mocker):
    """
    Test prepare_data with 2 CSV files that has the same filepath and config but different content, 
    check _process_and_save_data is called on both files.
    """
    # Define the original CSV content as a string
    original_csv_content = """user_id,date,brand,amount
                              1,2024-09-01,B,100
                              2,2024-09-02,A,150
                              3,2024-09-03,C,200
                              4,2024-09-04,B,120
                              5,2024-09-05,A,130
                              6,2024-09-06,C,160
                              7,2024-09-07,B,140
                              8,2024-09-08,A,180
                              9,2024-09-09,C,110
                              10,2024-09-10,B,170"""
    
    # Define different CSV content as a string
    different_csv_content =  """user_id,date,brand,amount
                                1,2024-09-02,B,100
                                2,2024-09-02,A,150
                                3,2024-09-03,C,200
                                4,2024-09-04,B,120
                                5,2024-09-05,A,130
                                6,2024-09-06,C,160
                                7,2024-09-07,B,140
                                8,2024-09-08,A,180
                                9,2024-09-09,C,110
                                10,2024-09-10,B,170"""
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

    try:
        tmp_file.write(original_csv_content.encode('utf-8'))
        tmp_file_path = tmp_file.name
        tmp_file.close()

        handler = CachedDataHandler()

        config_hash, data_hash = handler.get_hashes(sample_config.dict(), tmp_file_path)
        cached_data_path_1 = os.path.join(handler.processed_data_path, f"{config_hash}_{data_hash}.pkl")

        # Test prepare_data
        dc = DataContext(sample_config)
        dc.prepare_data(tmp_file_path)

        # Overwrite the content to the same file
        tmp_file = open(tmp_file_path, 'w', encoding='utf-8')
        tmp_file.write(different_csv_content)
        tmp_file.close()

        config_hash, data_hash = handler.get_hashes(sample_config.dict(), tmp_file_path)
        cached_data_path_2 = os.path.join(handler.processed_data_path, f"{config_hash}_{data_hash}.pkl")

        # Patch the _process_and_save_data method
        mock_process = mocker.patch.object(dc, '_process_and_save_data')

        # Call prepare_data with the second file path
        dc.prepare_data(tmp_file_path)

        # Assert that _process_and_save_data was not called with the second file
        mock_process.assert_called_once_with(tmp_file_path, cached_data_path_2)
        assert cached_data_path_1 != cached_data_path_2

    finally:
        # Clean up: remove the temporary CSV files, cached data files, and directories if they exist and is empty
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if os.path.exists(cached_data_path_1):
            os.remove(cached_data_path_1)
        if os.path.exists(cached_data_path_2):
            os.remove(cached_data_path_2)
        if os.path.exists(handler.processed_data_path) and not os.listdir(handler.processed_data_path):
            os.rmdir(handler.processed_data_path)
        data_path = os.path.dirname(handler.processed_data_path)  # Path to the 'data' folder
        if os.path.exists(data_path) and not os.listdir(data_path):
            os.rmdir(data_path)



def test_data_context_caching(sample_config, mocker):
    """
    test if the file is not cached before, 
    _process_and_save_data() method in the else clause is called.
    """
    # Use tempfile to create a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        pass

    try:
        #Mock the CachedDataHandler
        mock_cache_handler = mocker.patch('src.data.cache_data_handler.CachedDataHandler')
        mock_cache_handler.return_value.load_processed_data.return_value = (None, None)

        #Mock the _process_and_save_data method
        data_context=DataContext(sample_config)
        mocker.patch.object(data_context, '_process_and_save_data')

        #Call prepare_data using the path to the temporary CSV file
        data_context.prepare_data(tmp_file.name)

        #Assert that _process_and_save_data was called
        data_context._process_and_save_data.assert_called_once()

    finally:
        # Clean up the temporary file
        os.remove(tmp_file.name)



def test_split_data(sample_data, sample_config):

    data_context=DataContext(sample_config)

    train_df, valid_df, test_df, train_overlap_users =data_context._split_data(sample_data)

    # Calculate expected number of users
    num_users = len(sample_data['user_id'].unique())

     # Calculate expected number of users
    expected_train_users = floor(num_users * (1-0.15-0.2))
    overlap_users_count = ceil(expected_train_users * 0.5*0.65)
    expected_valid_users = floor((num_users-expected_train_users)* 0.15/(0.15+0.2)) + overlap_users_count
    expected_test_users = ceil((num_users-expected_train_users) * 0.2/(0.15+0.2)) + overlap_users_count

    # Assert the lengths
    assert len(train_df['user_id'].unique()) == expected_train_users
    assert len(valid_df['user_id'].unique()) == expected_valid_users
    assert len(test_df['user_id'].unique()) == expected_test_users

    # Check that the overlap users are in both the train and validation/test sets
    for user in train_overlap_users:
        assert user in train_df['user_id'].values
        assert user in valid_df['user_id'].values or user in test_df['user_id'].values

    # Ensure that there is no overlap between train, valid, and test users except the overlap users
    assert not set(train_df['user_id'].unique()).difference(set(train_overlap_users)).intersection(valid_df['user_id'].unique())
    assert not set(train_df['user_id'].unique()).difference(set(train_overlap_users)).intersection(test_df['user_id'].unique())
    assert not set(valid_df['user_id'].unique()).difference(set(train_overlap_users)).intersection(test_df['user_id'].unique())



@pytest.fixture
def sample_df2():
    # Sample data with 3 entries
    data = {
        'user_id': [1, 1, 2],
        'date': [
            '2024-01-01', '2024-01-07', '2024-01-08'
        ],
        'brand': ['A', 'A', 'B'],
        'amount': [100, 150, 200]
    }

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

def test_preprocess_data(sample_df2, sample_config):

    # Call the preprocessing function
    data_context=DataContext(sample_config)
    processed_df = data_context._preprocess_data(sample_df2)
    
    expected_df = pd.DataFrame({
        'user_id': [1, 1, 2, 2],
        'time_interval': [
            '(2023-12-31 23:59:59.999999999, 2024-01-08 00:00:00]',
            '(2023-12-31 23:59:59.999999999, 2024-01-08 00:00:00]',
            '(2023-12-31 23:59:59.999999999, 2024-01-08 00:00:00]',
            '(2023-12-31 23:59:59.999999999, 2024-01-08 00:00:00]',
        ],
        'brand': ['A', 'B', 'A', 'B'],
        'amount': [250, 0, 0, 200],
        'date': [pd.Timestamp('2024-01-01'), pd.NaT, pd.NaT, pd.Timestamp('2024-01-08')]
    })

    pd.testing.assert_frame_equal(processed_df, expected_df)


