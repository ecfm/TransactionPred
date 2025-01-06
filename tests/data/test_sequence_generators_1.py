import pytest
from src.config.data_config import SequenceConfig
from datetime import datetime
import pandas as pd
from src.data.sequence_generators import ContinuousTimeSequenceGenerator
from src.data.feature_processors import NoOpFeatureProcessor, BrandToIdProcessor, TimeDeltaEncoder
from torch import tensor
import torch
import numpy as np

@pytest.fixture
def generator():
    config = SequenceConfig(type="continuous_time", features=['brand', 'amount', 'date'])
    return ContinuousTimeSequenceGenerator(config)

def test_continuous_time_sequence_generator(generator):
    # Setup
    sample_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2],
        'brand': ['A', 'B', 'A', 'B'],
        'amount': [100, 200, 150, 250],
        'date': [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 1), datetime(2020, 1, 3)]
    })
    
    brand_processor = BrandToIdProcessor({'name': 'brand'})
    amount_processor = NoOpFeatureProcessor({'name': 'amount'})
    date_processor = TimeDeltaEncoder({'name': 'date'})
    date_processor.fit(sample_data, 'date')
    brand_processor.fit(sample_data, 'brand')

    feature_processors = {
        'brand': brand_processor,
        'amount': amount_processor,
        'date': date_processor
    }

    # Generate sequences
    user_ids = [1, 2]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 1, 4)
    sequences = generator.generate(user_ids, sample_data, start_date, end_date, feature_processors)
    #print(sequences)
    expected_structure = {
        'brand': [0, int, int, 1],
        'amount': [-1, int, int, 2],
        'date': [start_date, pd.Timestamp, pd.Timestamp, end_date]
    }
    for sequence in sequences:
        assert isinstance(sequence, dict), "Each sequence should be a dictionary"
        for key, expected_types in expected_structure.items():
            assert key in sequence, f"'{key}' key is missing in sequence"
            assert len(sequence[key]) == len(expected_types), \
                f"Length of '{key}' list is incorrect"
    
    # Collate sequences
    collated_sequences = generator.collate(sequences)
    assert 'sequences' in collated_sequences, "'sequences' key is missing in output"
    assert 'masks' in collated_sequences, "'masks' key is missing in output"
    expected_dtypes = {
        'brand': torch.int64,
        'amount': torch.int64,
        'date': torch.int64
    }
    for key, dtype in expected_dtypes.items():
        assert key in collated_sequences['sequences'], f"'{key}' key is missing in sequences"
        assert collated_sequences['sequences'][key].dtype == dtype, f"{key} tensor dtype is incorrect"
    assert collated_sequences['masks'].dtype == torch.bool, "masks tensor dtype is incorrect"
    
    # Recover original features
    recovered_sequences = generator.recover_original_features(collated_sequences, feature_processors)

    # Convert to dataframe
    df = generator.collated_sequences_to_df(user_ids, recovered_sequences)
    expected_columns = ['user_id', 'brand', 'amount', 'date']
    print(df)
    assert list(df.columns) == expected_columns, "Column names do not match expected values"
    assert pd.api.types.is_integer_dtype(df['user_id']), "user_id column dtype is incorrect"
    assert pd.api.types.is_string_dtype(df['brand']), "brand column dtype is incorrect"
    assert pd.api.types.is_integer_dtype(df['amount']), "amount column dtype is incorrect"
    assert pd.api.types.is_datetime64_any_dtype(df['date']), "date column dtype is incorrect"