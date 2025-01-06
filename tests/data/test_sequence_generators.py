import torch
import pytest
import pandas as pd
from src.data.feature_processors import Time2VecEncoder


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'time_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
    })

@pytest.fixture
def time2vec_encoder():
    return Time2VecEncoder(in_features=1, out_features=3, means="sine")

def test_t2v_function(time2vec_encoder, sample_data):
    # Convert sample data to tensor
    tau = torch.tensor(sample_data['time_feature'].values, dtype=torch.float32).unsqueeze(1)
    
    # Manually set known parameters for reproducibility
    time2vec_encoder.w0.data.fill_(0.5)
    time2vec_encoder.b0.data.fill_(0.1)
    time2vec_encoder.w.data.fill_(0.5)
    time2vec_encoder.b.data.fill_(0.2)

    # Call the t2v function directly
    output = Time2VecEncoder.t2v(tau, time2vec_encoder.f, time2vec_encoder.w, time2vec_encoder.b, time2vec_encoder.w0, time2vec_encoder.b0)

    # Print shapes for debugging
    print(f"tau shape: {tau.shape}")
    print(f"w shape: {time2vec_encoder.w.shape}")
    print(f"b shape: {time2vec_encoder.b.shape}")
    print(f"w0 shape: {time2vec_encoder.w0.shape}")
    print(f"b0 shape: {time2vec_encoder.b0.shape}")
    print(f"output shape: {output.shape}")

    # Manually compute expected output based on the fixed parameters
    expected_v1 = torch.sin(torch.matmul(tau, time2vec_encoder.w) + time2vec_encoder.b)
    expected_v2 = torch.matmul(tau, time2vec_encoder.w0) + time2vec_encoder.b0
    expected_output = torch.cat([expected_v1, expected_v2], dim=-1)

    # Assert that the output values are close to the expected ones
    assert torch.allclose(output, expected_output, atol=1e-6), "Output values from t2v do not match the expected values"

    # Assert the output shape
    assert output.shape == (5, 3), f"Expected output shape (5, 3), but got {output.shape}"


def test_forward_function(time2vec_encoder, sample_data):
    # Convert sample data to tensor
    tau = torch.tensor(sample_data['time_feature'].values, dtype=torch.float32).unsqueeze(1)
    
    # Manually set known parameters for reproducibility
    time2vec_encoder.w0.data.fill_(0.5)
    time2vec_encoder.b0.data.fill_(0.1)
    time2vec_encoder.w.data.fill_(0.5)
    time2vec_encoder.b.data.fill_(0.2)

    # Call the forward function
    output = time2vec_encoder.forward(tau)
    
    # Print shapes for debugging
    print(f"tau shape: {tau.shape}")
    print(f"w shape: {time2vec_encoder.w.shape}")
    print(f"b shape: {time2vec_encoder.b.shape}")
    print(f"w0 shape: {time2vec_encoder.w0.shape}")
    print(f"b0 shape: {time2vec_encoder.b0.shape}")
    print(f"output shape: {output.shape}")

    # Manually compute expected output based on the fixed parameters
    expected_v1 = torch.sin(torch.matmul(tau, time2vec_encoder.w) + time2vec_encoder.b)
    expected_v2 = torch.matmul(tau, time2vec_encoder.w0) + time2vec_encoder.b0
    expected_output = torch.cat([expected_v1, expected_v2], dim=-1)

    # Assert that the output values are close to the expected ones
    assert torch.allclose(output, expected_output, atol=1e-6), "Output values from forward do not match the expected values"

    # Assert the output shape
    assert output.shape == (5, 3), f"Expected output shape (5, 3), but got {output.shape}"