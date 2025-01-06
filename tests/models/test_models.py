from unittest.mock import MagicMock

import pytest

import torch
import torch.nn as nn
from typing import Union, Dict, Any, Optional
from src.config.model_config import NeuralNetModelConfig, EmbeddingCoderConfig, RTDLNumEmbeddingsCoderConfig, \
    LinearCoderConfig, TransformerConfig, CoderConfig, NoOpCoderConfig
from src.models.encoders_decoders import PiecewiseLinearEmbeddings, RTDLNumEmbeddingsProcessor, NoOpCoder, LinearCoder
from src.models.models import SinusoidalPositionalEncoding, NeuralNetModelBase, TransformerEncoderModel

""" Sinusoidal Positional Encoding tests"""

@pytest.fixture
def mock_positional_encoding_params_zero_droupout():
    return {
        "d_model": 16,
        "dropout": 0.0,
        "max_len": 10
    }
def test_sinusoidal_positional_encoding(mock_positional_encoding_params_zero_droupout):
    positional_encoding = SinusoidalPositionalEncoding(**mock_positional_encoding_params_zero_droupout)
    input_tensor = torch.zeros(1, mock_positional_encoding_params_zero_droupout['max_len'], mock_positional_encoding_params_zero_droupout['d_model'])
    output_tensor = positional_encoding(input_tensor)
    
    assert torch.allclose(output_tensor[0, 0], input_tensor[0, 0] + positional_encoding.pe[:1].squeeze(0), atol=1e-6)

    assert hasattr(positional_encoding, 'pe')
    assert positional_encoding.dropout.p == mock_positional_encoding_params_zero_droupout['dropout']
    assert isinstance(positional_encoding.pe, torch.Tensor)
    assert positional_encoding.pe.size() == (mock_positional_encoding_params_zero_droupout['max_len'], 1, mock_positional_encoding_params_zero_droupout['d_model'])
    
    

@pytest.fixture(params=[
    {"d_model": 16, "dropout": 0.0, "max_len": 10},            
    {"d_model": 32, "dropout": 0.1, "max_len": 1000},          
    {"d_model": 64, "dropout": 1.0, "max_len": 500},           
    {"d_model": 10000, "dropout": 0.5, "max_len": 10}
])
def mock_positional_encoding_params(request):
    return request.param


# Input tensor with zero length
def test_sinusoidal_positional_encoding_input_tensor_with_zero_length(mock_positional_encoding_params):
    encoding_layer = SinusoidalPositionalEncoding(mock_positional_encoding_params['d_model'], 
                                                  mock_positional_encoding_params['dropout'], 
                                                  mock_positional_encoding_params['max_len'])

    input_tensor = torch.zeros(1, 0, mock_positional_encoding_params['d_model'])
    output_tensor = encoding_layer(input_tensor)

    assert output_tensor.size(1) == 0, "Output tensor should have zero length"


@pytest.fixture(params=[
    torch.randn(5, 10, 16),                                 
    torch.randn(1, 1000, 32),                               
    torch.randn(10, 500, 64),                               
    torch.randn(1, 10, 10000)                                
])
def mock_input_tensors(request):
    return request.param



# Forward method processes input tensor without errors
def test_sinusoidal_positional_encoding_forward_method_input_processing(mock_positional_encoding_params, mock_input_tensors):
    if (mock_positional_encoding_params['d_model'] == mock_input_tensors.shape[2]):
        encoding = SinusoidalPositionalEncoding(mock_positional_encoding_params['d_model'], 
                                            mock_positional_encoding_params['dropout'], 
                                            mock_positional_encoding_params['max_len'])
        
        output = encoding(mock_input_tensors)
        assert output.shape == mock_input_tensors.shape
        
        if mock_positional_encoding_params['dropout'] != 0.0:
            assert not torch.allclose(output[0, 0], mock_input_tensors[0, 0] + encoding.pe[:1].squeeze(0), atol=1e-6)



def test_sinusoidal_positional_encoding_buffer_registration_no_gradient():
   
    d_model = 512
    pos_enc = SinusoidalPositionalEncoding(d_model=d_model)
    x = torch.randn(1, 10, d_model, requires_grad=True)

    output = pos_enc(x)
    output.sum().backward()

    assert x.grad is not None, "Gradients were not computed for the input tensor."
    assert pos_enc.pe.grad is None, "Positional encoding buffer should not have gradients."



# Check for numerical stability with extreme values
def test_sinusoidal_positional_encoding_numerical_stability_with_extreme_values():
    d_model = 10000
    max_len = 10000
    encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

    # Assertions for numerical stability
    assert torch.all(torch.isfinite(encoding.pe)), "Positional encoding contains non-finite values"


# Validate performance with different device types (CPU/GPU)
def test_sinusoidal_positional_encoding_performance_with_cpu():
    # Check if CUDA (GPU) is available, otherwise default to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SinusoidalPositionalEncoding(d_model=512)
    model.to(device)
    
    input_tensor = torch.randn(3, 10, 512).to(device)
    output = model(input_tensor)
    
    assert output.device == device, f"Output should be on {device}, but found on {output.device}"


# Ensure positional encoding works with different batch sizes
def test_sinusoidal_positional_encoding_with_different_batch_sizes():
    pos_enc = SinusoidalPositionalEncoding(d_model=512, dropout=0.1, max_len=1000)

    input_tensor_batch_1 = torch.randn(5, 10, 512)
    input_tensor_batch_2 = torch.randn(10, 10, 512)

    output_batch_1 = pos_enc(input_tensor_batch_1)
    output_batch_2 = pos_enc(input_tensor_batch_2)

    assert output_batch_1.shape == input_tensor_batch_1.shape
    assert output_batch_2.shape == input_tensor_batch_2.shape



""" Learned Positional Embedding tests """

@pytest.fixture(params=[(50, 32), (100, 64), (200, 128)])
def embedding_layer(request):
    max_seq_length, d_model = request.param
    return nn.Embedding(max_seq_length, d_model), max_seq_length, d_model


@pytest.mark.parametrize("batch_size, seq_length", [(32, 50), (16, 100), (8, 200)])
def test_learned_positional_encoding_embedding_layer_shape(embedding_layer, batch_size, seq_length):
    embedding_layer, max_seq_length, d_model = embedding_layer
    positions = torch.arange(seq_length).clamp_max(max_seq_length-1).expand(batch_size, seq_length)

    position_embeddings = embedding_layer(positions)

    assert position_embeddings.shape == (batch_size, seq_length, d_model)

@pytest.mark.parametrize("batch_size, seq_length", [(32, 50), (16, 100), (8, 200)])
def test_learned_positional_encoding_embedding_layer_max_length(embedding_layer, batch_size, seq_length):
    embedding_layer, max_seq_length, d_model = embedding_layer

    seq_length = min(seq_length, max_seq_length)

    positions = torch.arange(seq_length).expand(batch_size, seq_length)

    position_embeddings = embedding_layer(positions)

    assert position_embeddings.shape == (batch_size, seq_length, d_model)


@pytest.mark.parametrize("batch_size, seq_length", [(32, 60), (16, 120), (8, 220)])
def test_learned_positional_encoding_embedding_layer_exceed_max_length(embedding_layer, batch_size, seq_length):
    embedding_layer, max_seq_length, _ = embedding_layer

    positions = torch.arange(seq_length).expand(batch_size, seq_length)

    if seq_length > max_seq_length:
        with pytest.raises(IndexError):
            embedding_layer(positions)
    else:
        embedding_layer(positions)


@pytest.mark.parametrize("batch_size, seq_length", [(32, 50), (16, 100), (8, 200)])
def test_learned_positional_encoding_embedding_layer_values(embedding_layer, batch_size, seq_length):
    embedding_layer, max_seq_length, _ = embedding_layer

    seq_length = min(seq_length, max_seq_length)
    positions = torch.arange(seq_length).expand(batch_size, seq_length)

    position_embeddings = embedding_layer(positions)

    assert torch.any(position_embeddings != 0), "Embeddings should not be all zeros."



@pytest.mark.parametrize("batch_size, seq_length", [(32, 50), (16, 100), (8, 200)])
def test_learned_positional_encoding_embedding_layer_device(embedding_layer, batch_size, seq_length):
    embedding_layer, max_seq_length, _ = embedding_layer

    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    positions = torch.arange(seq_length).clamp_max(max_seq_length-1).expand(batch_size, seq_length).to(device)

    embedding_layer = embedding_layer.to(device)
    position_embeddings = embedding_layer(positions)

    assert position_embeddings.device == device, f"Position embeddings should be on {device}, but found on {position_embeddings.device}"


@pytest.fixture
def mock_rtdl_num_embeddings_processor_params():
    # Convert Python lists to PyTorch tensors
    bins = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0])]
    d_embedding = 16
    activation = 'relu'
    return bins, d_embedding, activation


def test_rtdl_num_embeddings_processor_initialization(mock_rtdl_num_embeddings_processor_params):
    bins, d_embedding, activation = mock_rtdl_num_embeddings_processor_params


    adjusted_bins = [torch.tensor([0., 1.]), torch.tensor([1., 2.]), torch.tensor([2., 3.])]

    processor = RTDLNumEmbeddingsProcessor(adjusted_bins, d_embedding, activation)

    assert isinstance(processor, RTDLNumEmbeddingsProcessor)


    input_tensor = torch.tensor([[0., 1., 2.]])
    output = processor.embedding_layer(input_tensor)


    assert output.shape == (1, 3, d_embedding)
@pytest.fixture
def mock_data_context():
    class MockFeatureProcessor:
        def __init__(self, id_to_value, data):
            self.id_to_value = id_to_value
            self.data = data  # Store the data as a PyTorch tensor

        def get_feature_data(self):
            return self.data  # Return the data directly

    class MockDataContext:
        def __init__(self):
            self.config = type('Config', (object,), {
                'input': type('InputConfig', (object,), {'features': ['feature1', 'feature2']})
            })
            self.feature_processors = {
                'feature1': MockFeatureProcessor(id_to_value={0: 'a', 1: 'b'}, data=torch.tensor([[0.0], [1.0]])),
                'feature2': MockFeatureProcessor(id_to_value={0: 'c', 1: 'd'}, data=torch.tensor([[0.0], [1.0]]))
            }

    return MockDataContext()

@pytest.fixture
def mock_data_context():
    class MockFeatureProcessor:
        def __init__(self, id_to_value, data):
            self.id_to_value = id_to_value
            self.data = data  # Store the data as a PyTorch tensor

        def get_feature_data(self):
            return self.data  # Return the data directly

    class MockDataContext:
        def __init__(self):
            # Provide more samples for testing
            self.config = type('Config', (object,), {
                'input': type('InputConfig', (object,), {'features': ['feature1', 'feature2']})
            })
            self.feature_processors = {
                'feature1': MockFeatureProcessor(
                    id_to_value={0: 'a', 1: 'b'},
                    data=torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])  # Increased sample size
                ),
                'feature2': MockFeatureProcessor(
                    id_to_value={0: 'c', 1: 'd'},
                    data=torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])  # Increased sample size
                )
            }

    return MockDataContext()

def test_transformer_encoder_model_initialization(mock_data_context):
    config = NeuralNetModelConfig(
        type='transformer-encoder',
        d_model=16,
        encoders={
            'feature1': RTDLNumEmbeddingsCoderConfig(
                type='rtdl_num_embeddings',
                d_embedding=8,
                activation='relu'
            ),
            'feature2': RTDLNumEmbeddingsCoderConfig(
                type='rtdl_num_embeddings',
                d_embedding=8,
                activation='relu'
            )
        },
        decoder=LinearCoderConfig(input_dim=16, output_dim=16),
        concat_layer=LinearCoderConfig(input_dim=16, output_dim=16),
        max_seq_length=100,
        transformer=TransformerConfig(nhead=2, num_encoder_layers=2, dropout=0.1),
        positional_encoding='sinusoidal'
    )

    # Initialize the TransformerEncoderModel
    model = TransformerEncoderModel(config, mock_data_context)

    # Validate the initialization
    assert isinstance(model, TransformerEncoderModel)
    assert model.d_model == 16
    assert len(model.encoders) == 2  # Ensure encoders are initialized

    for feature, encoder in model.encoders.items():
        assert isinstance(encoder, RTDLNumEmbeddingsProcessor)
        assert hasattr(encoder, 'embedding_layer')  # Ensure embedding_layer is set
        assert hasattr(encoder, 'activation')  # Check for activation
        assert isinstance(encoder.embedding_layer,
                          PiecewiseLinearEmbeddings)  # Ensure it is an instance of PiecewiseLinearEmbeddings

    # Check concat_layer type and properties
    assert isinstance(model.concat_layer, nn.Linear)
    assert model.concat_layer.in_features == 16
    assert model.concat_layer.out_features == 16

    # Check positional encoder
    assert isinstance(model.pos_encoder, nn.Module)  # Verify that pos_encoder is a nn.Module
    if config.positional_encoding == 'sinusoidal':
        assert isinstance(model.pos_encoder, SinusoidalPositionalEncoding)
        assert model.pos_encoder.dropout.p == 0.1
    elif config.positional_encoding == 'learned':
        assert isinstance(model.pos_encoder, nn.Embedding)
        assert model.pos_encoder.num_embeddings == 100
        assert model.pos_encoder.embedding_dim == 16

    # Check transformer model
    assert isinstance(model.model, nn.TransformerEncoder)
    assert len(model.model.layers) == 2  # Check the number of encoder layers

    # check for specific layers or configurations within transformer
    for layer in model.model.layers:
        assert isinstance(layer, nn.TransformerEncoderLayer)
        assert layer.dropout.p == 0.1

    assert model


def test_transformer_encoder_model_forward(mock_data_context):
    config = NeuralNetModelConfig(
        type='transformer-encoder',
        d_model=16,
        encoders={
            'feature1': RTDLNumEmbeddingsCoderConfig(
                type='rtdl_num_embeddings',
                bins=[[-1, 0, 1], [-1, 0, 1]],
                d_embedding=8,
                activation='relu'
            ),
            'feature2': RTDLNumEmbeddingsCoderConfig(
                type='rtdl_num_embeddings',
                bins=[[-1, 0, 1], [-1, 0, 1]],
                d_embedding=8,
                activation='relu'
            )
        },
        decoder=LinearCoderConfig(input_dim=16, output_dim=1),  # Example linear decoder
        transformer=TransformerConfig(nhead=2, num_encoder_layers=2, dropout=0.1),
        concat_layer=LinearCoderConfig(input_dim=16, output_dim=16),
        positional_encoding='sinusoidal',
        max_seq_length=128  # Ensure this field is provided


    )




