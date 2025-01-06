from typing import List

import numpy as np
import pytest
from unittest import mock
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation


from src.models.encoders_decoders import CoderRegistry, NoOpCoder, EmbeddingCoder, LinearCoder, MLPCoder, \
    RTDLNumEmbeddingsProcessor
from src.config.model_config import EmbeddingCoderConfig, LinearCoderConfig, MLPCoderConfig, NoOpCoderConfig



""" No Op Coder Unit tests """

@pytest.fixture(params=[
    {},  
    {"input_dim": 10, 
     "output_dim": 5,
     "activation": "relu"}
])

# Mock configuration
def mock_config(request):
    return request.param


@pytest.fixture(params=[
    torch.randn(3, 5),  
    torch.tensor([[1, 2], [3, 4]]), 
    torch.tensor([[1, 2], [3.0, 4.0]]), 
    torch.tensor([[-1, 0], [2, -3]]), 
    torch.tensor([[1.0, 2.0], [3.0, 4.0]])
])

def mock_tensor_2d(request):
    return request.param


@pytest.fixture
def mock_tensor_3d():
    return torch.randn(3, 4, 5)


@pytest.fixture
def mock_tensor_empty():
    return torch.tensor([])


@pytest.fixture
def mock_tensor_high_dim():
    return torch.randn(3, 4, 5, 6)



# Registering NoOpCoder with CoderRegistry
def test_register_no_op_coder():
    assert "no_op" in CoderRegistry
    assert CoderRegistry.get("no_op") == NoOpCoder



# Initializing NoOpCoder with a valid config
def test_no_op_coder_initializing_with_valid_config(mock_config):
    no_op_coder = NoOpCoder(mock_config)
    assert isinstance(no_op_coder, NoOpCoder)


    
# Forward method processes 2D torch.Tensor correctly
def test_no_op_coder_forward_2d_tensor(mock_config, mock_tensor_2d):
    coder = NoOpCoder(mock_config)
    output = coder.forward(mock_tensor_2d)
    assert output.dim() == 3  
    assert output.size(-1) == 1
    assert isinstance(output, torch.Tensor)   



# Forward method receives tensor with dimensions other than 2
def test_no_op_coder_forward_method_tensor_dimensions_other_than_2(mock_config, mock_tensor_3d):
    coder = NoOpCoder(mock_config)
    result = coder.forward(mock_tensor_3d)
    assert torch.equal(result, mock_tensor_3d)
    assert isinstance(result, torch.Tensor)   



# Forward method receives empty tensor
def test_no_op_coder_forward_empty_tensor(mock_config, mock_tensor_empty):
    coder = NoOpCoder(mock_config)
    output = coder.forward(mock_tensor_empty)
    assert torch.equal(output, mock_tensor_empty)   
    assert isinstance(output, torch.Tensor)   



# Forward method receives tensor with large dimensions
def test_no_op_coder_forward_large_dimensions(mock_config, mock_tensor_high_dim):
    coder = NoOpCoder(mock_config)
    result = coder.forward(mock_tensor_high_dim)
    assert result.shape == mock_tensor_high_dim.shape
    assert isinstance(result, torch.Tensor)   





""" Embedding Coder Unit Tests """

@pytest.fixture(params=[
    EmbeddingCoderConfig(type='embedding', num_embeddings=1000, output_dim=64),
    EmbeddingCoderConfig(
        type='embedding',
        num_embeddings=100,
        output_dim=50,
        activation='relu'
    )
])

def mock_embedding_coder_config(request):
    return request.param


@pytest.fixture
def mock_embedding_coder_invalid_config_tensor_pairs():

    # Config with a valid range, but tensor with out-of-range indices
    config_1 = EmbeddingCoderConfig(num_embeddings=10, output_dim=5)
    tensor_1 = torch.tensor([11, 12, 13])  # Indices out of the range [0, 9]

    # Config with a very small number of embeddings and tensor with higher index
    config_2 = EmbeddingCoderConfig(num_embeddings=2, output_dim=4)
    tensor_2 = torch.tensor([0, 1, 2])  # Index 2 is out of the range [0, 1]

    # Config with zero embeddings (invalid), and tensor with any index
    config_3 = EmbeddingCoderConfig(num_embeddings=0, output_dim=3)
    tensor_3 = torch.tensor([0, 1, 2])  # Any index will be invalid

    return [
        (config_1, tensor_1),
        (config_2, tensor_2),
        (config_3, tensor_3)
    ]


@pytest.fixture
def mock_embedding_coder_empty_tensor():
    return torch.tensor([], dtype=torch.int64)


@pytest.fixture
def mock_tensor_2d():
    return torch.randint(0, 1000, (2, 3), dtype=torch.long)  # [batch_size, seq_len]


@pytest.fixture
def mock_tensor_3d():
    return torch.randint(0, 1000, (2, 3, 4), dtype=torch.long)  # [batch_size, seq_len, input_dim]


@pytest.fixture
def mock_tensor_edge_case():
    return torch.randint(0, 1000, (1, 1), dtype=torch.long)  # Minimal size [batch_size, seq_len]


@pytest.fixture(params=[
    EmbeddingCoderConfig(num_embeddings=10, output_dim=3.5),
    EmbeddingCoderConfig(num_embeddings=100.5, output_dim=5),
    EmbeddingCoderConfig(num_embeddings=50.5, output_dim=5.5)
])

def mock_embedding_coder_config_non_integer_attr(request):
    return request.param


@pytest.fixture
def mock_embedding_coder_large_dim_tensor_config_pair():
    return {'config' : EmbeddingCoderConfig(num_embeddings=1000, output_dim=100), 
            'input_tensor' : torch.randint(0, 1000, (1000, 1000))}


@pytest.fixture(params=[
    (1, 10), 
    (3, 5), 
    (2, 2),
    (2, 3, 4), 
    (2, 3, 4, 7)   
])

def mock_input_tensor(request, mock_embedding_coder_config):
    shape = request.param
    input_tensor = torch.randint(0, mock_embedding_coder_config.num_embeddings, shape)
    return input_tensor


@pytest.fixture(params=[
        [1, 2, 3],       # Invalid input: list of integers
        {"key": "value"},
         "string" # Invalid input: float tensor
])

def mock_invalid_input(request):
    return request.param


@pytest.fixture
def mock_float_tensor():
    return torch.randn(2, 3).float()



# EmbeddingCoder registers successfully in CoderRegistry
def test_embedding_coder_registration():
    assert "embedding" in CoderRegistry
    assert CoderRegistry.get("embedding") == EmbeddingCoder



# EmbeddingCoder initializes with valid EmbeddingCoderConfig
def test_embedding_coder_initialization_with_valid_config(mocker, mock_embedding_coder_config):
    coder = EmbeddingCoder(mock_embedding_coder_config)

    assert coder.embed.num_embeddings == mock_embedding_coder_config.num_embeddings
    assert coder.embed.embedding_dim == mock_embedding_coder_config.output_dim
    assert isinstance(coder, EmbeddingCoder)
    assert isinstance(coder.embed, nn.Embedding)

    mock_embedding = mocker.patch('torch.nn.Embedding', autospec=True)
    
    encoder = EmbeddingCoder(mock_embedding_coder_config)
    
    mock_embedding.assert_called_once_with(mock_embedding_coder_config.num_embeddings, mock_embedding_coder_config.output_dim)
    
    assert isinstance(encoder.embed, MagicMock)
    


# EmbeddingCoderConfig set to None/ attributes set to None/ non-integer attributes
def test_embedding_coder_initialization_with_none(mock_embedding_coder_config_non_integer_attr, mock_tensor_2d):
    config = EmbeddingCoderConfig(num_embeddings=None, output_dim=5)
    with pytest.raises(TypeError):
        coder = EmbeddingCoder(config)

    config = EmbeddingCoderConfig(num_embeddings=10, output_dim=None)
    with pytest.raises(TypeError):
        coder = EmbeddingCoder(config)

    config = EmbeddingCoderConfig(num_embeddings=None, output_dim=None)
    with pytest.raises(TypeError):
        coder = EmbeddingCoder(config)

    with pytest.raises(AttributeError):
        coder = EmbeddingCoder(None)

    config = mock_embedding_coder_config_non_integer_attr
   
    encoder = EmbeddingCoder(config)
    with pytest.raises(IndexError):
        output = encoder(mock_tensor_2d)

    

   

# Forward method receives tensors with varying shapes
def test_embedding_coder_forward_with_different_shapes(mocker, mock_embedding_coder_config, mock_input_tensor):
    
    encoder = EmbeddingCoder(mock_embedding_coder_config)
    mock_embedding = mocker.patch.object(encoder.embed, 'forward', autospec=True)

    input_tensor = mock_input_tensor
    shape = input_tensor.shape
    
    mock_embedding.return_value = torch.randn(*shape, mock_embedding_coder_config.output_dim)
    output_tensor = encoder(input_tensor)
    mock_embedding.assert_called_with(input_tensor)
    
    expected_shape = (*shape, mock_embedding_coder_config.output_dim)
    assert output_tensor.shape == expected_shape
    assert output_tensor.dtype == torch.float32



def test_embedding_coder_with_extreme_values(mock_embedding_coder_config):
   
    mock_embedding_coder_config.num_embeddings = 10**4  
    mock_embedding_coder_config.output_dim = 10**4 
    
    encoder = EmbeddingCoder(mock_embedding_coder_config)
    
    input_tensor = torch.randint(0, mock_embedding_coder_config.num_embeddings, (1, 100))
    output_tensor = encoder(input_tensor)
    
    expected_shape = (*input_tensor.shape, mock_embedding_coder_config.output_dim)
    assert output_tensor.shape == expected_shape
    assert output_tensor.dtype == torch.float32




# Forward method receives edge case tensor - torch.randint(0, 1000, (1, 1), dtype=torch.long) 
def test_embedding_coder_forward_edge_case_tensor(mocker, mock_embedding_coder_config, mock_tensor_edge_case):
    encoder = EmbeddingCoder(mock_embedding_coder_config)
    
    return_shape = (*mock_tensor_edge_case.shape, mock_embedding_coder_config.output_dim)
    mock_embedding = mocker.patch.object(encoder.embed, 'forward', return_value=torch.randn(*return_shape))
    
    input_tensor = mock_tensor_edge_case
    output_tensor = encoder(input_tensor)
    
    mock_embedding.assert_called_once_with(input_tensor)
    assert output_tensor.shape == return_shape  
    assert output_tensor.dtype == torch.float32



# Forward method receives input tensor with invalid values
def test_embedding_coder_with_invalid_config_tensor_pairs(mock_embedding_coder_invalid_config_tensor_pairs):
    for config, tensor in mock_embedding_coder_invalid_config_tensor_pairs:
        encoder = EmbeddingCoder(config)
        
        with pytest.raises((IndexError, RuntimeError)):
            encoder(tensor)



# Forward method receives empty input tensor
def test_forward_empty_input_tensor(mock_embedding_coder_config, mock_embedding_coder_empty_tensor):
   
    coder = EmbeddingCoder(mock_embedding_coder_config)
    output = coder.forward(mock_embedding_coder_empty_tensor)

    expected_tensor = torch.tensor([], dtype=torch.float32).reshape(0, mock_embedding_coder_config.output_dim)
    
    assert output.numel() == expected_tensor.numel()
    assert output.shape == expected_tensor.shape
    assert output.dtype == expected_tensor.dtype
    assert torch.equal(output, expected_tensor)



# EmbeddingCoder handles large input tensors efficiently
def test_handles_large_input_tensors_efficiently(mock_embedding_coder_large_dim_tensor_config_pair):
    config = mock_embedding_coder_large_dim_tensor_config_pair['config']
    coder = EmbeddingCoder(config)

    input_tensor = mock_embedding_coder_large_dim_tensor_config_pair['input_tensor']
    output = coder(input_tensor)

    expected_shape = (input_tensor.shape[0], input_tensor.shape[1], config.output_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    assert output.dtype == torch.float32, f"Expected output dtype torch.float32, but got {output.dtype}"

    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains infinities"



# Invalid input types
def test_embedding_coder_with_invalid_input_type(mocker, mock_embedding_coder_config, mock_invalid_input, mock_float_tensor):

    encoder = EmbeddingCoder(mock_embedding_coder_config)
    with pytest.raises(TypeError):
        encoder(mock_invalid_input)

    with pytest.raises(RuntimeError):
        encoder(mock_float_tensor)
        

       
""" Linear Coder Unit Tests """


@pytest.fixture(params=[
    LinearCoderConfig(input_dim=10, output_dim=5),
    LinearCoderConfig(input_dim=20, output_dim=15, activation='relu')
])

def mock_linear_coder_config(request):
    return request.param


@pytest.fixture
def mock_linear_coder_invalid_config():
    return [
        LinearCoderConfig(input_dim=10, output_dim=None),
        LinearCoderConfig(input_dim=None, output_dim=5),
        LinearCoderConfig(input_dim=None, output_dim=None)
    ]


@pytest.fixture
def mock_linear_coder_large_dim_tensor_config_pair():
    return {'config': LinearCoderConfig(input_dim=1000, output_dim=1000),
            'input_tensor': torch.randn(100, 1000)}


@pytest.fixture(params=[
    (1, 10),
    (3, 5),
    (2, 2),
    (1, 1, 10), 
    (5, 10, 10),  
    (2, 3, 10), 
    (3, 5, 1, 4, 2)
])

def mock_input_tensor(request, mock_linear_coder_config):
    shape = request.param
    return torch.randn(*shape, mock_linear_coder_config.input_dim)


@pytest.fixture(params=[
        [1, 2, 3],       
        {"key": "value"},
         "string" 
])

def mock_invalid_input(request):
    return request.param


@pytest.fixture
def mock_float_tensor():
    return torch.randn(2, 3).float()


@pytest.fixture
def mock_linear_coder_invalid_config_tensor_pairs():
    invalid_pairs = [
        # Tensor has more dimensions than expected
        (LinearCoderConfig(input_dim=10, output_dim=5, activation='relu'), torch.randn(2, 3, 4, 10)),
    ]
    
    return invalid_pairs


# LinearCoder registers successfully in CoderRegistry
def test_linear_coder_registration():
    assert "linear" in CoderRegistry
    assert CoderRegistry.get("linear") == LinearCoder



# LinearCoder initializes with valid LinearCoderConfig
def test_linear_coder_initialization_with_valid_config(mocker, mock_linear_coder_config):
    coder = LinearCoder(mock_linear_coder_config)
    
    assert isinstance(coder, LinearCoder)
    assert isinstance(coder.linear, nn.Linear)
    assert coder.linear.in_features == mock_linear_coder_config.input_dim
    assert coder.linear.out_features == mock_linear_coder_config.output_dim
    if mock_linear_coder_config.activation:
        assert coder.activation is not None
    else:
        assert coder.activation is None



# LinearCoderConfig set to None/ attributes set to None
def test_linear_coder_initialization_with_none():
    config = LinearCoderConfig(input_dim=None, output_dim=5)
    with pytest.raises(TypeError):
        LinearCoder(config)

    config = LinearCoderConfig(input_dim=10, output_dim=None)
    with pytest.raises(TypeError):
        LinearCoder(config)

    config = LinearCoderConfig(input_dim=None, output_dim=None)
    with pytest.raises(TypeError):
        LinearCoder(config)

    with pytest.raises(AttributeError):
        LinearCoder(None)



# Test with various activation functions
@pytest.mark.parametrize("activation", [None, "relu", "sigmoid"])
def test_linear_coder_with_various_activations(activation, mock_input_tensor):
    config = LinearCoderConfig(input_dim=mock_input_tensor.size(-1), output_dim=5, activation=activation)
    encoder = LinearCoder(config)
    output_tensor = encoder(mock_input_tensor)
    
    expected_shape = mock_input_tensor.shape[:-1] + (5,)
    assert output_tensor.shape == expected_shape
    assert output_tensor.dtype == torch.float32

    linear_output = encoder.linear(mock_input_tensor.reshape(-1, mock_input_tensor.size(-1))).reshape(*expected_shape)
    
    if activation:
        expected_activation_fn = getattr(F, activation)
        expected_output = expected_activation_fn(linear_output)
        
        assert not torch.allclose(output_tensor, linear_output)
        
        assert torch.allclose(output_tensor, expected_output)
    else:
        assert torch.allclose(output_tensor, linear_output)



# LinearCoder handles large input tensors efficiently
def test_linear_coder_handles_large_input_tensors_efficiently(mock_linear_coder_large_dim_tensor_config_pair):
    config = mock_linear_coder_large_dim_tensor_config_pair['config']
    coder = LinearCoder(config)
    input_tensor = mock_linear_coder_large_dim_tensor_config_pair['input_tensor']
    
    output_tensor = coder(input_tensor)

    expected_shape = (input_tensor.shape[0], config.output_dim)
    assert output_tensor.shape == expected_shape
    assert output_tensor.dtype == torch.float32
    assert not torch.isnan(output_tensor).any(), "Output contains NaNs"
    assert not torch.isinf(output_tensor).any(), "Output contains infinities"



# Forward method receives varying input shapes
def test_linear_coder_forward_with_different_shapes(mocker, mock_linear_coder_config, mock_input_tensor):
    encoder = LinearCoder(mock_linear_coder_config)
    mock_linear = mocker.patch.object(encoder.linear, 'forward', autospec=True)
    
    input_tensor = mock_input_tensor
    shape = input_tensor.shape
    mock_linear.return_value = torch.randn(*shape[:-1], mock_linear_coder_config.output_dim)
    
    output_tensor = encoder(input_tensor)
    # mock_linear.assert_called_with(input_tensor)
    
    expected_shape = (*shape[:-1], mock_linear_coder_config.output_dim)
    assert output_tensor.shape == expected_shape
    assert output_tensor.dtype == torch.float32



# Forward method receives input tensor with invalid values
def test_linear_coder_with_invalid_input_type(mock_linear_coder_config, mock_invalid_input, mock_float_tensor):
    encoder = LinearCoder(mock_linear_coder_config)
    
    with pytest.raises(AttributeError):
        output = encoder(mock_invalid_input)
    
    with pytest.raises(RuntimeError):
        output = encoder(mock_float_tensor)



# Forward method receives empty input tensor
def test_linear_coder_forward_empty_input_tensor(mock_linear_coder_config):
    coder = LinearCoder(mock_linear_coder_config)
    empty_tensor = torch.empty(0, mock_linear_coder_config.input_dim)
    output = coder(empty_tensor)

    expected_tensor = torch.empty(0, mock_linear_coder_config.output_dim)
    assert output.shape == expected_tensor.shape
    assert output.dtype == expected_tensor.dtype



# LinearCoder with large values in config
def test_linear_coder_with_extreme_values(mocker, mock_linear_coder_config):
    mock_linear_coder_config.input_dim = 10**4
    mock_linear_coder_config.output_dim = 10**4
    
    encoder = LinearCoder(mock_linear_coder_config)
    input_tensor = torch.randn(1, mock_linear_coder_config.input_dim)
    output_tensor = encoder(input_tensor)
    
    expected_shape = (1, mock_linear_coder_config.output_dim)
    assert output_tensor.shape == expected_shape
    assert output_tensor.dtype == torch.float32



""" MLP Coder Unit Tests """

@pytest.fixture(params=[
    MLPCoderConfig(
        input_dim=64,
        output_dim=10,
        hidden_dims=[128, 256],
        activation='ReLU',
        final_activation='Sigmoid'
    ),
    MLPCoderConfig(
        input_dim=64,
        output_dim=20,
        hidden_dims=[64, 32],
        activation='LeakyReLU',
        final_activation='Tanh'
    )
])
def mock_mlp_coder_config(request):
    return request.param


@pytest.fixture(params=[
    (torch.randn(2, 1, 8, 8), MLPCoderConfig(input_dim=64, output_dim=10, hidden_dims=[128, 256], activation='ReLU', final_activation='Sigmoid')),
    (torch.randn(4, 3, 32, 32), MLPCoderConfig(input_dim=3072, output_dim=20, hidden_dims=[64, 32], activation='LeakyReLU', final_activation='Tanh')),
])
def mock_mlp_coder_input_tensor_and_config(request):
    tensor, config = request.param
    if tensor.size(1) * tensor.size(2) * tensor.size(3) == config.input_dim:
        tensor = tensor.view(tensor.size(0), -1)  # Flattened to match input_dim
    else:
        tensor = tensor.view(tensor.size(0), tensor.size(2) * tensor.size(3))  # Adjusted based on input_dim
    tensor = tensor.unsqueeze(1)  
    return tensor, config


@pytest.fixture
def mock_input_tensor():
    return torch.randn(2, 3, 64)  

@pytest.fixture
def mock_input_tensor_3d():
    return torch.randn(10, 5, 64)  


@pytest.fixture
def mock_invalid_input():
    return [1, 2, 3]  


@pytest.fixture
def mock_mlp_coder_edge_case_config_tensor_pair():
    return {'config' : MLPCoderConfig(
        input_dim=10,
        output_dim=5,
        hidden_dims=[20, 15],
        activation='ReLU',
        final_activation='Sigmoid'
    ),
    'input_tensor' : torch.tensor(
        [[
            [1e10, -1e10, 1e5, -1e5, 1e2, 1e10, -1e10, 1e5, -1e5, 1e2],  
            [1e10, -1e10, 1e5, -1e5, 1e2, 1e10, -1e10, 1e5, -1e5, 1e2],  
            [1e10, -1e10, 1e5, -1e5, 1e2, 1e10, -1e10, 1e5, -1e5, 1e2]   
        ]])}



def test_mlp_coder_registration():
    assert "mlp" in CoderRegistry
    assert CoderRegistry.get("mlp") == MLPCoder



def test_mlp_coder_initialization_with_valid_config(mocker, mock_mlp_coder_config):
    coder = MLPCoder(mock_mlp_coder_config)
    assert isinstance(coder, MLPCoder)
    assert isinstance(coder.mlp, nn.Sequential)



@pytest.mark.parametrize("config", [
    MLPCoderConfig(
        input_dim=None,
        output_dim=None,
        hidden_dims=[128, 256],
        activation=None,
        final_activation=None
    ),
    MLPCoderConfig(
        input_dim=64,
        output_dim=None,
        hidden_dims=[128, 256],
        activation=None,
        final_activation=None
    ),
    MLPCoderConfig(
        input_dim=64,
        output_dim=10,
        hidden_dims=[128, 256],  
        activation=None,
        final_activation=None
    ),
    MLPCoderConfig(
        input_dim=64,
        output_dim=10,
        hidden_dims=[128, 256],
        activation=None,
        final_activation=None
    )
])
def test_mlp_coder_initialization_with_none(config):
    
    with pytest.raises(TypeError):
        coder = MLPCoder(config)



# Handling of empty hidden_dims list in MLPCoderConfig
def test_mlp_coder_empty_hidden_dims(mock_mlp_coder_config):
    coder = MLPCoder(mock_mlp_coder_config)

    
    expected_layer_count = len(mock_mlp_coder_config.hidden_dims) * 2  # Each hidden dim adds a Linear layer and an activation
    expected_layer_count += 1  # Final Linear layer
    
    if mock_mlp_coder_config.final_activation:
        expected_layer_count += 1  # One more layer if final activation is present

    
    assert len(coder.mlp) == expected_layer_count
    assert isinstance(coder.mlp[0], nn.Linear)
    
    assert isinstance(coder.mlp[0], nn.Linear)
    assert coder.mlp[0].in_features == mock_mlp_coder_config.input_dim
    assert coder.mlp[0].out_features == mock_mlp_coder_config.hidden_dims[0] if mock_mlp_coder_config.hidden_dims else mock_mlp_coder_config.output_dim

    for i, hidden_dim in enumerate(mock_mlp_coder_config.hidden_dims):
        linear_layer_idx = i * 2
        activation_layer_idx = linear_layer_idx + 1

        assert isinstance(coder.mlp[linear_layer_idx], nn.Linear)
        if i == 0:
            assert coder.mlp[linear_layer_idx].in_features == mock_mlp_coder_config.input_dim
        else:
            assert coder.mlp[linear_layer_idx].in_features == mock_mlp_coder_config.hidden_dims[i - 1]
        assert coder.mlp[linear_layer_idx].out_features == hidden_dim

        assert isinstance(coder.mlp[activation_layer_idx], getattr(nn, mock_mlp_coder_config.activation))

    # Asserting the final Linear layer has correct output features
    final_linear_layer_idx = len(mock_mlp_coder_config.hidden_dims) * 2
    assert isinstance(coder.mlp[final_linear_layer_idx], nn.Linear)
    assert coder.mlp[final_linear_layer_idx].in_features == (mock_mlp_coder_config.hidden_dims[-1] if mock_mlp_coder_config.hidden_dims else mock_mlp_coder_config.input_dim)
    assert coder.mlp[final_linear_layer_idx].out_features == mock_mlp_coder_config.output_dim

    # If final activation is specified, check the last layer is the activation function
    if mock_mlp_coder_config.final_activation:
        assert isinstance(coder.mlp[final_linear_layer_idx + 1], getattr(nn, mock_mlp_coder_config.final_activation))



# Forward pass with valid input tensor producing expected output shape
def test_mlp_coder_forward_pass_with_valid_input(mock_mlp_coder_config, mock_input_tensor):
    mlp_coder = MLPCoder(mock_mlp_coder_config)
    output = mlp_coder(mock_input_tensor)

    expected_shape = (
        mock_input_tensor.shape[0],  # batch_size
        mock_input_tensor.shape[1],  # seq_len
        mock_mlp_coder_config.output_dim  # output_dim
    )
    assert output.shape == expected_shape



def test_mlp_coder_activation_functions_applied_correctly(mock_mlp_coder_config, mock_input_tensor):
    mlp_coder = MLPCoder(mock_mlp_coder_config)

    output = mlp_coder(mock_input_tensor)

    def get_expected_layer_types(config):
        expected_layer_types = []
        
        for hidden_dim in config.hidden_dims:
            expected_layer_types.append(nn.Linear)
            expected_layer_types.append(getattr(nn, config.activation))
        
        # output layer
        expected_layer_types.append(nn.Linear)
        
        # Final activation if it exists
        if config.final_activation:
            expected_layer_types.append(getattr(nn, config.final_activation))
        
        return expected_layer_types
    expected_layer_types = get_expected_layer_types(mock_mlp_coder_config)

    for i, expected_layer_type in enumerate(expected_layer_types):
        assert isinstance(mlp_coder.mlp[i], expected_layer_type)

    expected_output_shape = (mock_input_tensor.shape[0], mock_input_tensor.shape[1], mock_mlp_coder_config.output_dim)
    assert output.shape == expected_output_shape



# Handling of invalid activation function names in config
def test_mlp_coder_invalid_activation_function_name():
    with pytest.raises(AttributeError):
        config = MLPCoderConfig(input_dim=10, hidden_dims=[20, 30], activation='invalid_activation', output_dim=5)
        coder = MLPCoder(config)



@pytest.mark.parametrize(
    "input_tensor_shape",
    [
        (32, 10),  # [batch_size, input_dim]
        (2, 5, 10),  # [batch_size, seq_len, input_dim]
        (2, 3, 10, 10),  # [batch_size, seq_len, extra_dim, input_dim]
        (2, 3, 5)  # [batch_size, seq_len, mismatched_dim]
    ]
)
def test_forward_pass_with_incorrect_shape(mock_mlp_coder_config, input_tensor_shape):
   
    mlp_coder = MLPCoder(mock_mlp_coder_config)
    incorrect_shape_tensor = torch.randn(*input_tensor_shape)

    with pytest.raises(RuntimeError):
        output = mlp_coder(incorrect_shape_tensor)



# Handling of input tensor with extreme values
def test_mlp_coder_large_values(mock_mlp_coder_edge_case_config_tensor_pair):
    
    config = mock_mlp_coder_edge_case_config_tensor_pair['config']
    mlp_coder = MLPCoder(config)

    large_values_tensor = mock_mlp_coder_edge_case_config_tensor_pair['input_tensor']
    output = mlp_coder(large_values_tensor)

    assert output.dtype == torch.float32
    assert torch.all(torch.isfinite(output))  




""" Combined Tests"""

@pytest.fixture
def mock_no_op_config():
    return NoOpCoderConfig(input_dim=1, output_dim=1)

@pytest.fixture
def mock_embedding_config():
    return EmbeddingCoderConfig(num_embeddings=100,output_dim=64)

@pytest.fixture
def mock_mlp_config():
    return MLPCoderConfig(input_dim=66, hidden_dims=[128, 128], output_dim=256, activation="ReLU", final_activation="ReLU")

@pytest.fixture
def mock_linear_config():
    return LinearCoderConfig(input_dim=256, output_dim=256)

def test_full_model_behavior(mocker, mock_no_op_config, mock_embedding_config, mock_mlp_config, mock_linear_config):
   
    date_encoder = NoOpCoder(mock_no_op_config)
    brand_encoder = EmbeddingCoder(mock_embedding_config)
    amount_encoder = NoOpCoder(mock_no_op_config)
    concat_layer = MLPCoder(mock_mlp_config)
    decoder = LinearCoder(mock_linear_config)

    # mock input tensors
    date_input = torch.randn(10, 1)  
    brand_input = torch.randint(0, 100, (10,))  
    amount_input = torch.randn(10, 1)  

    # Forward pass through each encoder
    encoded_date = date_encoder(date_input)
    encoded_brand = brand_encoder(brand_input)
    encoded_amount = amount_encoder(amount_input)

    # Concatenating the encoded outputs
    combined_input = torch.cat([encoded_date.squeeze(-1), encoded_brand, encoded_amount.squeeze(-1)], dim=1)

    mlp_output = concat_layer(combined_input)

    # Final output from the decoder
    final_output = decoder(mlp_output)

    assert encoded_date.shape == (10, 1, 1)
    assert encoded_brand.shape == (10, 64)
    assert encoded_amount.shape == (10, 1, 1)
    assert combined_input.shape == (10, 66)
    assert mlp_output.shape == (10, 256)
    assert final_output.shape == (10, 256)

    mock_linear = mocker.patch.object(concat_layer.mlp[-1], 'forward', wraps=concat_layer.mlp[-1].forward)
    _ = concat_layer(combined_input)
    mock_linear.assert_called_once()

    mock_activation = mocker.patch.object(F, 'relu', wraps=F.relu)
    mlp_output = concat_layer(combined_input)
    mock_activation.assert_called()

def test_rtdl_num_embeddings_processor():
    # Provide bins as a list of tensors
    bins = [
        torch.tensor([0.0, 0.2]),  # For feature 1
        torch.tensor([0.2, 0.4]),  # For feature 2
        torch.tensor([0.4, 0.6])  # For feature 3
    ]
    d_embedding = 16
    activation = True

    processor = RTDLNumEmbeddingsProcessor(bins=bins, d_embedding=d_embedding, activation=activation)

    # Input tensor with the correct number of features
    x = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, 0.6]])


    output = processor(x)

    assert output.shape == (x.shape[0], x.shape[1], d_embedding)
