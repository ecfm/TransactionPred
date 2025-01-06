import pandas as pd
import numpy as np
import pytest 

from src.data.feature_processors import NoOpFeatureProcessor, MinMaxFeatureProcessor, BrandToIdProcessor, \
    DateEncoder, TimeDeltaEncoder

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'brand': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'E', 'F', 'G'],
        'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05',
                                '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10'])
    })

@pytest.fixture
def sample_data_min_max():
    return pd.DataFrame({
        'amount': [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    })

@pytest.fixture
def new_sample_data_min_max():
    return pd.DataFrame({
        'amount': [-100, 0, 100, 200, 300, 400, 600]  # Includes negative values and values larger than original max
    })

@pytest.fixture
def sample_brand_data():
    return pd.DataFrame({
        'brand': ['A', 'B', 'C', 'A', 'B', 'D', 'E', 'F', 'G', 'H']
    })

@pytest.fixture
def new_sample_brand_data():
    return pd.DataFrame({
        'brand': ['I', 'J', 'K', 'L']  # New brands not seen during fitting
    })

@pytest.fixture
def sample_date_data():
    return pd.DataFrame({
        'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05',
                                '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10'])
    })

@pytest.fixture
def new_sample_date_data():
    return pd.DataFrame({
        'date': pd.to_datetime(['2021-01-11', '2021-02-01', '2021-03-01'])  # Dates beyond the original max date
    })

@pytest.fixture
def sample_min_max_brand_data():
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'brand': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
        'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05',
                                '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10'])
    })

def test_no_op_feature_processor(sample_data):
    params = {'name': 'amount'}
    processor = NoOpFeatureProcessor(params)

    processor.fit(sample_data, 'amount')
    transformed = processor.transform(sample_data, 'amount')
    inverse_transformed = processor.inverse_transform(transformed)

    pd.testing.assert_series_equal(transformed, sample_data['amount'])
    pd.testing.assert_series_equal(inverse_transformed, sample_data['amount'])
    assert processor.get_sos_token() == -1, "SOS token should be -1"
    assert processor.get_eos_token() == 2, "EOS token should be 2"

def test_min_max_feature_processor(sample_data_min_max, new_sample_data_min_max):
    params = {}
    processor = MinMaxFeatureProcessor(params)

    processor.fit(sample_data_min_max, 'amount')
    transformed = processor.transform(sample_data_min_max, 'amount')
    expected_transformed = (sample_data_min_max['amount'] - sample_data_min_max['amount'].min()) / (
            sample_data_min_max['amount'].max() - sample_data_min_max['amount'].min())
    expected_transformed.name = None  # Remove name attribute to avoid discrepancies

    pd.testing.assert_series_equal(transformed, expected_transformed, check_dtype=False)

    inverse_transformed = processor.inverse_transform(transformed)
    inverse_transformed = inverse_transformed.astype('int64')  # Ensure dtype consistency
    inverse_transformed.name = None
    expected_inverse_transformed = sample_data_min_max['amount']
    expected_inverse_transformed.name = None
    pd.testing.assert_series_equal(inverse_transformed, expected_inverse_transformed, check_dtype=False)
    assert processor.get_sos_token() == -1, "SOS token should be -1"
    assert processor.get_eos_token() == 2, "EOS token should be 2"

    # Test with new data
    transformed_new = processor.transform(new_sample_data_min_max, 'amount')
    expected_transformed_new = (new_sample_data_min_max['amount'] - sample_data_min_max['amount'].min()) / (
            sample_data_min_max['amount'].max() - sample_data_min_max['amount'].min())
    expected_transformed_new.name = None

    pd.testing.assert_series_equal(transformed_new, expected_transformed_new, check_dtype=False)

    inverse_transformed_new = processor.inverse_transform(transformed_new)
    inverse_transformed_new = inverse_transformed_new.astype('int64')
    inverse_transformed_new.name = None
    expected_inverse_transformed_new = new_sample_data_min_max['amount']
    expected_inverse_transformed_new.name = None

    pd.testing.assert_series_equal(inverse_transformed_new, expected_inverse_transformed_new, check_dtype=False)


def test_date_encoder(sample_date_data, new_sample_date_data):
    params = {}
    processor = TimeDeltaEncoder(params)

    processor.fit(sample_date_data, 'date')
    transformed = processor.transform(sample_date_data, 'date')

    # Compute expected transformed values
    expected_transformed = (sample_date_data['date'] - sample_date_data['date'].min()).dt.days

    pd.testing.assert_series_equal(transformed, expected_transformed)

    # Test inverse transformation
    inverse_transformed = processor.inverse_transform(transformed)
    pd.testing.assert_series_equal(inverse_transformed, sample_date_data['date'])

    # Test special token handling
    processor.set_eos_token('2021-01-11')  # Set EOS token to a date after the max date
    eos_token_transformed = processor.single_transform('2021-01-11')
    assert eos_token_transformed == (pd.to_datetime('2021-01-11') - sample_date_data['date'].min()).days
    assert processor.get_eos_token() == eos_token_transformed

    # Test with new data
    transformed_new = processor.transform(new_sample_date_data, 'date')
    expected_transformed_new = (new_sample_date_data['date'] - sample_date_data['date'].min()).dt.days

    pd.testing.assert_series_equal(transformed_new, expected_transformed_new)

    inverse_transformed_new = processor.inverse_transform(transformed_new)
    pd.testing.assert_series_equal(inverse_transformed_new, new_sample_date_data['date'])

def test_min_max_scaler_by_brand(sample_min_max_brand_data):
    params={}
    processor = MinMaxFeatureProcessor(params, True)
    processor.fit(sample_min_max_brand_data, feature='amount',  groupby_feature='brand')

    transformed = processor.transform(sample_min_max_brand_data, feature='amount',  groupby_feature='brand')
    inverse_transformed = processor.inverse_transform(transformed, sample_min_max_brand_data,  groupby_feature='brand')

    assert transformed[sample_min_max_brand_data['brand'] == 'A'].tolist() == [0.0, 0.5, 1.0]
    assert transformed[sample_min_max_brand_data['brand'] == 'B'].tolist() == [0.0, 1.0]
    assert transformed[sample_min_max_brand_data['brand'] == 'C'].tolist() == [0.0, 0.25, 0.5, 0.75, 1]
    assert inverse_transformed[sample_min_max_brand_data['brand'] == 'A'].tolist() == [100, 200, 300]
    assert inverse_transformed[sample_min_max_brand_data['brand'] == 'B'].tolist() == [400, 500]
    assert inverse_transformed[sample_min_max_brand_data['brand'] == 'C'].tolist() == [600, 700, 800, 900, 1000]
    assert not transformed.equals(sample_min_max_brand_data['amount'])
    
if __name__ == "__main__":
    pytest.main([__file__])
