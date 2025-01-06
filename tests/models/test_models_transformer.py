import pytest
import torch
import torch.nn as nn
from src.models import TransformerEncoderModel, SinusoidalPositionalEncoding
from src.config.model_config import NeuralNetModelConfig, TransformerConfig, LinearCoderConfig, MLPCoderConfig

@pytest.fixture
def mock_data_context():
    return MockDataContext()

@pytest.fixture
def transformer_config():
    return TransformerConfig(
        nhead=8,
        num_encoder_layers=2,
        dropout=0.1
    )

@pytest.fixture
def model_config(transformer_config):
    return NeuralNetModelConfig(
        type='transformer-encoder',
        d_model=32,
        encoders={
            'linear': LinearCoderConfig(input_dim=64, output_dim=32),
            'mlp': MLPCoderConfig(input_dim=64, output_dim=32, hidden_dims=[128, 64], final_activation='relu')
        },
        decoder=LinearCoderConfig(input_dim=32, output_dim=10),
        concat_layer=LinearCoderConfig(input_dim=64, output_dim=32),
        max_seq_length=100,
        transformer=transformer_config,
        positional_encoding='sinusoidal'
    )

def test_linear_coder_initialization(mocker, model_config, mock_data_context):
    # Test LinearCoder initialization
    mock_linear = mocker.patch('torch.nn.Linear')
    model = TransformerEncoderModel(model_config, mock_data_context)
    mock_linear.assert_called_with(64, 32)

def test_mlp_coder_initialization(mocker, model_config, mock_data_context):
    # Test MLPCoder initialization
    mock_mlp = mocker.patch('torch.nn.Sequential')
    model = TransformerEncoderModel(model_config, mock_data_context)
    mock_mlp.assert_called()

def test_transformer_encoder_layer(mocker, model_config, mock_data_context):
    # Test the TransformerEncoderModel to ensure correct output shape
    model = TransformerEncoderModel(model_config, mock_data_context)
    x = torch.rand(2, 10, 64)  # batch_size=2, seq_len=10, d_model=64
    mask = torch.zeros(2, 10).bool()

    # Mock the concatenation layer to ensure correct concatenation
    mock_concat_layer = mocker.patch.object(model, 'concat_layer', return_value=torch.rand(2, 10, 32))  # Ensuring d_model=32 after concat
    output = model({'sequences': {'linear': x, 'mlp': x}, 'masks': mask})

    assert output.shape == torch.Size([2, 10])  # Ensure the decoder output shape matches the expected output_dim
    mock_concat_layer.assert_called_once()

def test_full_encoder_stack(mocker, model_config, mock_data_context):
    # Test the full encoder stack to ensure correct output shape after all layers
    model = TransformerEncoderModel(model_config, mock_data_context)
    x = torch.rand(2, 10, 64)
    mask = torch.zeros(2, 10).bool()

    mock_concat_layer = mocker.patch.object(model, 'concat_layer', return_value=torch.rand(2, 10, 32))  # Ensuring d_model=32 after concat
    output = model({'sequences': {'linear': x, 'mlp': x}, 'masks': mask})

    assert output.shape == torch.Size([2, 10])  # Final output shape should match decoder output
    mock_concat_layer.assert_called_once()

def test_sinusoidal_positional_encoding():
    # Test SinusoidalPositionalEncoding to ensure it produces correct output shape
    pos_encoder = SinusoidalPositionalEncoding(d_model=32)
    x = torch.zeros(10, 32)  # 10 time steps, 32-dimensional input
    output = pos_encoder(x)
    
    assert output.shape == torch.Size([10, 32])

# Mock classes to simulate data context and feature processors
class MockDataContext:
    def __init__(self):
        self.config = MockConfig()
        self.feature_processors = {
            'linear': MockFeatureProcessor(),
            'mlp': MockFeatureProcessor(),
            'brand': MockFeatureProcessor()  # To handle the decoder config
        }

class MockConfig:
    def __init__(self):
        self.input = MockInputConfig()
        self.output = MockOutputConfig()

class MockInputConfig:
    def __init__(self):
        self.type = "continuous_time"
        self.features = ['linear', 'mlp']

class MockOutputConfig:
    def __init__(self):
        self.type = "multi_hot_brand_amount"

class MockFeatureProcessor:
    def __init__(self):
        self.id_to_value = {0: 'feature_a', 1: 'feature_b'}