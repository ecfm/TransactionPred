import pytest
import pandas as pd
from src.data.sequence_generators import BrandOnlyMultiHotSequenceGenerator, BrandAmountMultiHotSequenceGenerator, SeparateMultiHotSequenceGenerator
from src.config.data_config import SequenceConfig
from src.data.feature_processors import BrandToIdProcessor, NoOpFeatureProcessor
import pytest
import pandas as pd
import numpy as np

#set up
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2],
        'brand': ["A", "B", "A", "C"],
        'amount': [100, 200, 150, 300]
    })
@pytest.fixture
def brand_processor():
    processor = BrandToIdProcessor({})
    return processor

#test brand only multi hot sequence generator
@pytest.fixture
def brand_only_sequence_generator():
    config = SequenceConfig(type="multi_hot_brand_only", features=['brand'])
    return BrandOnlyMultiHotSequenceGenerator(config)

def test_brand_only_multi_hot_sequence(brand_only_sequence_generator, sample_data,brand_processor):
    
    # set the id_to_value mapping
    brand_processor.fit(sample_data, 'brand')
    processed_data = sample_data.copy()
    processed_data['brand'] = brand_processor.transform(processed_data, 'brand')
    brand_only_sequence_generator.id_to_value = brand_processor.id_to_value

    # Test for user_id 1
    group = processed_data[processed_data['user_id'] == 1]
    result = brand_only_sequence_generator._create_sequence(group)
    assert result == [[0,1,1,0]], f"Expected [[0,1,1,0]], but got {result}"
    
    # Test for user_id 2
    group = processed_data[processed_data['user_id'] == 2]
    result = brand_only_sequence_generator._create_sequence(group)
    assert result == [[0,1,0,1]], f"Expected [[0,1,0,1]], but got {result}"
    #The output will start with a 0 because of the special tokens, followed by whether the user has a transaction with the brand is A, B, or C
    #We can remove the zero by modifying the BrandToIDProcessor or the _create_sequence function


#test brand amount multi hot sequence generator
@pytest.fixture
def brand_amount_sequence_generator():
    config = SequenceConfig(type="multi_hot_brand_amount", features=['brand', 'amount'])
    return BrandAmountMultiHotSequenceGenerator(config)
    
@pytest.fixture(params=[
    NoOpFeatureProcessor({'name': 'amount', 'start': 0, 'end': 1000})
])
def amount_processor(request):
    if isinstance(request.param, str):
        return request.getfixturevalue(request.param)
    return request.param


def test_brand_amount_multi_hot_sequence(brand_amount_sequence_generator, sample_data, brand_processor, amount_processor):
    # Fit and transform the data
    brand_processor.fit(sample_data, 'brand')
    processed_data = sample_data.copy()
    processed_data['brand'] = brand_processor.transform(processed_data, 'brand')
    processed_data['amount'] = amount_processor.transform(processed_data, 'amount')
    brand_amount_sequence_generator.value_to_id = brand_processor.value_to_id

    # Set up the sequence generator
    brand_amount_sequence_generator.id_to_value = brand_processor.id_to_value
    brand_amount_sequence_generator.amount_processor = amount_processor

    #_create_sequence for each user using NoopFeatureProcessor or SimpleBinningProcessor
    sequences = [brand_amount_sequence_generator._create_sequence(processed_data[processed_data['user_id'] == user_id])[0] 
                 for user_id in sample_data['user_id'].unique()]
    expected_sequences = [
            [0, 100, 200, 0, 0, 0],  # User 1
            [0, 150, 0, 300, 0, 0]   # User 2
        ]
    #print('sequences:', sequences)
    np.testing.assert_array_equal(sequences, expected_sequences)
    #The result contains three tokens: sos,eso and unk, and bin/amount for each user;
    #We can remove them by modifying the BrandToIDProcessor or the _create_sequence function

    #recovery of original features
    processed_sequences = {'sequences': np.array(sequences)}
    print('processed_sequences:', processed_sequences)
    recovered = brand_amount_sequence_generator.recover_original_features(processed_sequences, 
                                                             {'brand': brand_processor, 'amount': amount_processor})
    #print(recovered)
    assert recovered['sequences'].shape == (2, 6), f"Expected recovered shape (2, 6), but got {recovered['sequences'].shape}"
    expected_sequences = [
            [0, 100, 200, 0, 0, 0],  # User 1
            [0, 150, 0, 300, 0, 0]   # User 2
        ]
    np.testing.assert_array_equal(recovered['sequences'], np.array(expected_sequences))

#test separate multi hot sequence generator
@pytest.fixture
def separate_multi_hot_sequence_generator():
    config = SequenceConfig(type="multi_hot_separate", features=['brand', 'amount'])
    return SeparateMultiHotSequenceGenerator(config)

def test_separate_multi_hot_sequence(separate_multi_hot_sequence_generator, sample_data, brand_processor, amount_processor):
    
    brand_processor.fit(sample_data, 'brand')
    processed_data = sample_data.copy()
    processed_data['brand'] = brand_processor.transform(processed_data, 'brand')
    processed_data['amount'] = amount_processor.transform(processed_data, 'amount')
    separate_multi_hot_sequence_generator.value_to_id = brand_processor.value_to_id
    separate_multi_hot_sequence_generator.id_to_value = brand_processor.id_to_value
    separate_multi_hot_sequence_generator.amount_processor = amount_processor

    # Create sequences for each user
    sequences = [separate_multi_hot_sequence_generator._create_sequence(processed_data[processed_data['user_id'] == user_id],
                                                                         prediction_interval=1, input_interval=1) 
                 for user_id in sample_data['user_id'].unique()]
    #print('sequences:', sequences)
    # Extract the brand and amount vectors
    brand_sequences = [seq[0] for seq in sequences]
    amount_sequences = [seq[1] for seq in sequences]
    #There is one 0 at the start of each sequence because of the special tokens, followed by the actual sequence.
    #We can remove the zero by modifying the BrandToIDProcessor or the _create_sequence function
    expected_brand_sequences = [
            [0, 1, 1, 0],  # User 1
            [0, 1, 0, 1]   # User 2
        ]
    expected_amount_sequences = [
            [0, 100, 200, 0],  # User 1
            [0, 150, 0, 300]   # User 2
        ]

    np.testing.assert_array_equal(brand_sequences, expected_brand_sequences)
    np.testing.assert_array_equal(amount_sequences, expected_amount_sequences)

    # Recover original features
    processed_sequences = separate_multi_hot_sequence_generator.collate(sequences)
    #print('processed_sequences:', processed_sequences)
    recovered = separate_multi_hot_sequence_generator.recover_original_features(processed_sequences, 
                                                                                {'brand': brand_processor, 'amount': amount_processor})
    # Validate the shape and correctness of the recovered sequences
    assert recovered['brand_sequences'].shape == (2, 4), f"Expected brand shape (2, 4), but got {recovered['brand_sequences'].shape}"
    assert recovered['amount_sequences'].shape == (2, 4), f"Expected amount shape (2, 4), but got {recovered['amount_sequences'].shape}"

    expected_recovered_amounts = [
            [0, 100, 200, 0],  # User 1
            [0, 150, 0, 300]   # User 2
        ]

    np.testing.assert_array_almost_equal(recovered['amount_sequences'], np.array(expected_recovered_amounts), decimal=5)
