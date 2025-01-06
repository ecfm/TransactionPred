import pytest
from unittest import mock


import torch
import torch.nn as nn

from src.training.losses import LossRegistry, NoOpLoss, L1Loss, L2Loss


# Mock configuration
@pytest.fixture
def mock_config():
    return {}


@pytest.fixture(params=[
    {'invalid_param': 123},  
    {'learning_rate': -0.1},
    None
])

def mock_incorrect_config(request):
    return request.param

# Mock predictions and targets
@pytest.fixture
def mock_predictions_targets():
    predictions = torch.tensor([0.5, 0.7, 0.2, -1.4])
    targets = torch.tensor([0.5, 0.6, 0.1, -12.0])
    return predictions, targets


# Mock predictions and targets w/ different shapes
@pytest.fixture
def mock_predictions_targets_different_shape():
    predictions = torch.tensor([0.5, 0.7, 0.2, -1.4])
    targets = torch.tensor([[0.5], [0.6], [0.1], [-12.0]])
    return predictions, targets

@pytest.fixture
def mock_predictions_targets_large():
    predictions = torch.ones((1000, 1000))
    targets = torch.zeros((1000, 1000))
    return predictions, targets



@pytest.fixture
def mock_predictions_targets_with_nan_infinite():
    predictions = torch.tensor([1.0, float('nan'), float('inf')])
    targets = torch.tensor([0.0, float('nan'), float('-inf')])
    return predictions, targets

@pytest.fixture
def mock_predictions_targets_large_batch():
    predictions = torch.randn(1000, 10)
    targets = torch.randn(1000, 10)
    return predictions, targets

@pytest.fixture(params=[
        "string",
        123,
        12.34,
        None,
        ["list", "of", "strings"],
        {"key": "value"}
    ])

def mock_predictions_targets_non_tensor(request):
    return request.param


@pytest.fixture
def mock_predictions_targets_requires_grad():
    predictions = torch.tensor([0.5, 0.77, 0.2], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0.5, 0.6, 0.1], dtype=torch.float32)
    return predictions, targets



# NoOpLoss can be successfully registered in LossRegistry
def test_no_op_loss_registration():
    assert "no_op" in LossRegistry._registry, "NoOpLoss not found in LossRegistry"
    assert LossRegistry._registry["no_op"] == NoOpLoss


# NoOpLoss can be instantiated with a configuration object
def test_no_op_loss_instantiation_with_config():
    config = {"param": "value"}
    loss_fn = NoOpLoss(config)
    assert isinstance(loss_fn, NoOpLoss)



# NoOpLoss forward method accepts predictions and targets as inputs, returns a tensor with value 0.0 and maintains tensor type consistency
def test_no_op_loss_forward_returns_zero_tensor(mock_config, mock_predictions_targets):
    loss_fn = NoOpLoss(mock_config)
    
    output = loss_fn.forward(None, None)
    assert torch.equal(output, torch.tensor(0.0))

    predictions, targets = mock_predictions_targets
    output = loss_fn(predictions, targets)
    assert output.item() == 0.0, f"Expected loss to be 0.0, but got {output.item()}"

    assert isinstance(output, torch.Tensor)




# NoOpLoss forward method handles empty tensors for predictions and targets
def test_no_op_loss_forward_empty_tensors(mock_config):
    loss_fn = NoOpLoss(mock_config)
    predictions = torch.tensor([])
    targets = torch.tensor([])

    output = loss_fn(predictions, targets)
    assert output.item() == 0.0



# NoOpLoss forward method handles tensors with different shapes
def test_no_op_loss_forward_handles_different_shapes(mock_config, mock_predictions_targets_different_shape):
    loss_fn = NoOpLoss(mock_config)
    predictions, targets = mock_predictions_targets_different_shape
    output = loss_fn.forward(predictions, targets)
    assert torch.equal(output, torch.tensor(0.0))



# NoOpLoss forward method handles non-tensor inputs 
def test_no_op_loss_handles_non_tensor_inputs(mock_config):
    loss_fn = NoOpLoss(mock_config)
    output = loss_fn.forward("input", "target")
    assert torch.is_tensor(output) is True # forward() returns torch.tensor(0.0) no matter what the input is
    assert output.item() == 0.0



# NoOpLoss works with different device types (CPU/GPU)
def test_no_op_loss_device_types(mock_config, mock_predictions_targets):
    device = torch.device('cpu')
    loss_fn = NoOpLoss(mock_config)
    loss_fn.to(device)
    predictions, targets = mock_predictions_targets
    assert loss_fn(predictions, targets).device == device



# NoOpLoss forward method does not modify input tensors
def test_no_op_loss_forward_no_modification(mock_config, mock_predictions_targets):
    loss_fn = NoOpLoss(mock_config)
    predictions, targets = mock_predictions_targets

    predictions_copy = predictions.clone()
    targets_copy = targets.clone()

    output = loss_fn.forward(predictions, targets)

    assert torch.all(torch.eq(predictions, predictions_copy))
    assert torch.all(torch.eq(targets, targets_copy))




# L1Loss class is correctly registered in LossRegistry
def test_l1loss_registration():
    assert "l1" in LossRegistry, "L1Loss is not registered in LossRegistry"
    assert LossRegistry.get("l1") == L1Loss
    


# L1Loss instance can be created with a valid config + cannot be created without a config
def test_l1loss_instance_creation(mock_config):
    l1_loss_instance = LossRegistry.get("l1")(mock_config)
    assert isinstance(l1_loss_instance, L1Loss), "L1Loss instance was not created"

    assert isinstance(l1_loss_instance.l1_loss, torch.nn.modules.loss.L1Loss)

    with pytest.raises(TypeError):
        l1_loss_instance = LossRegistry.get("l1")()


        

# forward method handles empty tensor inputs 
def test_l1loss_forward_empty_tensors(mock_config):
    l1_loss = L1Loss(mock_config)
    predictions = torch.tensor([])
    targets = torch.tensor([])

    result = l1_loss(predictions, targets)
    assert torch.isnan(result)



# forward method computes L1 loss correctly for given predictions and targets
def test_l1loss_forward_computes_correctly(mock_config, mock_predictions_targets):
   
    l1_loss = L1Loss(mock_config)
    predictions, targets = mock_predictions_targets

    output = l1_loss.forward(predictions, targets)
    expected_loss_1 = torch.nn.functional.l1_loss(predictions, targets)
    expected_loss_2 = torch.abs(predictions - targets).mean()

    # Assert the output matches the expected loss
    assert torch.allclose(output, expected_loss_1)
    assert torch.allclose(output, expected_loss_2)


def test_l1loss_forward_mock_internal_method(mocker, mock_config):
    
    nn_l1_loss_mock = mocker.patch('torch.nn.L1Loss', autospec=True)
    l1_loss_instance_mock = nn_l1_loss_mock.return_value

    l1_loss = L1Loss(mock_config)

    predictions = torch.tensor([0.5, 0.7])
    targets = torch.tensor([0.3, 0.9])

    l1_loss_instance_mock.return_value = torch.tensor([0.2, 0.2]) # example return value

    output = l1_loss.forward(predictions, targets)

    nn_l1_loss_mock.assert_called_once()

    l1_loss_instance_mock.assert_called_once_with(predictions, targets)

    assert torch.allclose(output, torch.tensor([0.2, 0.2])) # match the mocked return value



# forward method handles mismatched tensor shapes for predictions and targets
def test_l1loss_forward_handles_mismatched_shapes(mock_config, mock_predictions_targets_different_shape):
    
    l1_loss = L1Loss(mock_config)
    predictions, targets = mock_predictions_targets_different_shape

    with pytest.warns(UserWarning, match="ensure they have the same size."):
        loss = l1_loss(predictions, targets)   


# L1Loss forward method with non-tensor inputs
def test_l1loss_forward_non_tensor_inputs(mock_config, mock_predictions_targets_non_tensor):
    loss_fn = L1Loss(mock_config)
    predictions = mock_predictions_targets_non_tensor
    targets = mock_predictions_targets_non_tensor

    
    with pytest.raises(Exception):
        loss_fn.forward(predictions, targets)



# L1Loss instance creation with missing or invalid config parameters
def test_l1loss_instance_creation_missing_config(mock_incorrect_config):
    with pytest.raises(KeyError):
        LossRegistry.get("invalid_config")
    
    # with pytest.raises(ValueError):
    #     L1Loss(mock_incorrect_config)



# L1Loss works with different device types (CPU/GPU)
def test_l1loss_forward_different_devices(mock_config):

    predictions_cpu = torch.randn(3, 5)
    targets_cpu = torch.randn(3, 5)
    
    if torch.cuda.is_available():
        predictions_gpu = torch.randn(3, 5).cuda()
        targets_gpu = torch.randn(3, 5).cuda()
    else:
        pytest.skip("CUDA is not available")

    l1_loss_instance = L1Loss(mock_config)

    
    loss_cpu = l1_loss_instance(predictions_cpu, targets_cpu)
    loss_gpu = l1_loss_instance(predictions_gpu, targets_gpu)

    
    assert isinstance(loss_cpu, torch.Tensor), "CPU loss is not a tensor"
    assert isinstance(loss_gpu, torch.Tensor), "GPU loss is not a tensor"

   
    assert loss_cpu.shape == (), "CPU loss has incorrect shape"
    assert loss_gpu.shape == (), "GPU loss has incorrect shape"

    
    assert loss_cpu.device == predictions_cpu.device, "CPU loss device mismatch"
    assert loss_gpu.device == predictions_gpu.device, "GPU loss device mismatch"

    
    if torch.cuda.is_available():
        # Check if the losses are the same on CPU and GPU (after moving GPU loss to CPU)
        loss_gpu_cpu = loss_gpu.cpu()
        assert torch.allclose(loss_cpu, loss_gpu_cpu), "Losses on CPU and GPU are not the same"


# forward method handles extremely large tensor inputs
def test_l1loss_forward_handles_extremely_large_inputs(mock_config, mock_predictions_targets_large):
    
    l1_loss = L1Loss(mock_config)
    predictions, targets = mock_predictions_targets_large
    output = l1_loss.forward(predictions, targets)
    assert output is not None
    assert output.shape == torch.Size([])



# forward method handles NaN or infinite values in predictions or targets
def test_l1loss_forward_handles_nan_or_infinite_values(mock_config, mock_predictions_targets_with_nan_infinite):
   
    l1_loss = L1Loss(mock_config)
    predictions, targets = mock_predictions_targets_with_nan_infinite
    result = l1_loss(predictions, targets)
    assert torch.isnan(result)



# forward method performance with large batch sizes
def test_l1loss_forward_large_batch(mocker, mock_config, mock_predictions_targets_large_batch):
    l1_loss = L1Loss(mock_config)
    predictions, targets = mock_predictions_targets_large_batch

    # mocker.patch('src.training.losses.L1Loss', return_value=nn.L1Loss())
    output = l1_loss.forward(predictions, targets)

    assert output.dim() == 0, f"Expected output to be a scalar, but got {output.dim()}-dimensional tensor"
    assert output.item() >= 0, f"Expected non-negative loss value, but got {output.item()}"




# forward method's gradient computation correctness
def test_l1loss_forward_gradient_computation(mocker, mock_config, mock_predictions_targets_requires_grad):
    l1loss = L1Loss(mock_config)
    
    predictions, targets = mock_predictions_targets_requires_grad
    output = l1loss.forward(predictions, targets)

    output.backward() 
    expected_gradients = torch.sign(predictions - targets) / predictions.numel()

    assert predictions.grad is not None
    
    assert torch.allclose(predictions.grad, expected_gradients), \
        f"Expected gradients: {expected_gradients}, but got: {predictions.grad}"


# forward method's behavior with mixed precision inputs
def test_l1loss_forward_mixed_precision_inputs(mock_config):
    l1_loss = L1Loss(mock_config)

    predictions = torch.tensor([0.5, 0.7], dtype=torch.float16)
    targets = torch.tensor([0.3, 0.9], dtype=torch.float32)

    output = l1_loss.forward(predictions, targets)

    assert output.dtype == torch.float32





# L2Loss class is registered in LossRegistry
def test_l2loss_registration():
    assert "l2" in LossRegistry, "L2Loss is not registered in LossRegistry"
    assert LossRegistry.get("l2") == L2Loss



# L2Loss instance can be created with a valid config + cannot be created without a config
def test_l2loss_instance_creation(mock_config):

    l2_loss_instance = LossRegistry.get("l2")(mock_config)
    assert isinstance(l2_loss_instance, L2Loss), "L2Loss instance was not created"

    with pytest.raises(TypeError):
        l2_loss_instance = LossRegistry.get("l1")()



# L2Loss forward method with empty tensor inputs
def test_l2loss_forward_empty_inputs(mock_config):
    loss_fn = L2Loss(mock_config)
    predictions = torch.tensor([])
    targets = torch.tensor([])
    result = loss_fn.forward(predictions, targets)
    assert torch.isnan(result)



# L2Loss forward method computes the mean squared error correctly
def test_l2loss_forward_computation(mock_config, mock_predictions_targets):
    loss_fn = LossRegistry.get("l2")(mock_config)
    predictions, targets = mock_predictions_targets
    result = loss_fn.forward(predictions, targets)
    expected_loss = torch.nn.functional.mse_loss(predictions, targets)
    assert torch.allclose(result, expected_loss)


def test_l2loss_forward_mock_internal_method(mocker, mock_config):
    
    nn_mse_loss_mock = mocker.patch('torch.nn.MSELoss', autospec=True)
    l2_loss_instance_mock = nn_mse_loss_mock.return_value

    l2_loss = L2Loss(mock_config)

    predictions = torch.tensor([0.5, 0.7])
    targets = torch.tensor([0.3, 0.9])

    l2_loss_instance_mock.return_value = torch.tensor([0.02, 0.02])  # example return value

    output = l2_loss.forward(predictions, targets)

    nn_mse_loss_mock.assert_called_once()

    l2_loss_instance_mock.assert_called_once_with(predictions, targets)

    assert torch.allclose(output, torch.tensor([0.02, 0.02]))  # match the mocked return value






# L2Loss forward method works with valid tensor inputs and returns a scalar loss value
def test_l2loss_forward_returns_scalar_loss(mock_config, mock_predictions_targets):
    loss_fn = L2Loss(mock_config)
    predictions, targets = mock_predictions_targets
    loss = loss_fn(predictions, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0



# L2Loss forward method with tensors of different shapes
def test_l2loss_forward_different_shapes(mock_config, mock_predictions_targets_different_shape):
    loss_fn = L2Loss(mock_config)
    predictions, targets = mock_predictions_targets_different_shape

    with pytest.warns(UserWarning, match="ensure they have the same size."):
        loss = loss_fn(predictions, targets)   
    


# L2Loss forward method with non-tensor inputs
def test_l2loss_forward_non_tensor_inputs(mock_config, mock_predictions_targets_non_tensor):
    loss_fn = L2Loss(mock_config)
    predictions = mock_predictions_targets_non_tensor
    targets = mock_predictions_targets_non_tensor

    
    with pytest.raises(Exception):
        loss_fn.forward(predictions, targets)



# L2Loss instance creation with missing or invalid config parameters
def test_l2loss_instance_creation_missing_config():
    with pytest.raises(KeyError):
        LossRegistry.get("invalid_config")

    # with pytest.raises(ValueError):
    #     L1Loss(mock_incorrect_config)



def test_l2loss_forward_different_devices(mock_config):

    predictions_cpu = torch.randn(3, 5)
    targets_cpu = torch.randn(3, 5)
    
    if torch.cuda.is_available():
        predictions_gpu = torch.randn(3, 5).cuda()
        targets_gpu = torch.randn(3, 5).cuda()
    else:
        pytest.skip("CUDA is not available")

    l2_loss_instance = L2Loss(mock_config)

    
    loss_cpu = l2_loss_instance(predictions_cpu, targets_cpu)
    loss_gpu = l2_loss_instance(predictions_gpu, targets_gpu)

    
    assert isinstance(loss_cpu, torch.Tensor), "CPU loss is not a tensor"
    assert isinstance(loss_gpu, torch.Tensor), "GPU loss is not a tensor"

   
    assert loss_cpu.shape == (), "CPU loss has incorrect shape"
    assert loss_gpu.shape == (), "GPU loss has incorrect shape"

    
    assert loss_cpu.device == predictions_cpu.device, "CPU loss device mismatch"
    assert loss_gpu.device == predictions_gpu.device, "GPU loss device mismatch"

    
    if torch.cuda.is_available():
        # Check if the losses are the same on CPU and GPU (after moving GPU loss to CPU)
        loss_gpu_cpu = loss_gpu.cpu()
        assert torch.allclose(loss_cpu, loss_gpu_cpu), "Losses on CPU and GPU are not the same"



# L2Loss forward method with NaN or infinite values in tensors
def test_l2loss_forward_nan_or_infinite(mock_config, mock_predictions_targets_with_nan_infinite):
    l2_loss = L2Loss(mock_config)
    predictions, targets = mock_predictions_targets_with_nan_infinite
    result = l2_loss(predictions, targets)
    assert torch.isnan(result)



# L2Loss forward method with very large tensor inputs
def test_l2loss_forward_large_inputs(mock_config, mock_predictions_targets_large):
    l2_loss = L2Loss(mock_config)
    predictions, targets = mock_predictions_targets_large

    output = l2_loss.forward(predictions, targets)
    assert output is not None
    assert output.shape == torch.Size([])


# forward method performance with large batch sizes
def test_l2loss_forward_large_batch(mocker, mock_config, mock_predictions_targets_large_batch):
    l2_loss = L2Loss(mock_config)
    predictions, targets = mock_predictions_targets_large_batch

    output = l2_loss.forward(predictions, targets)

    assert output.dim() == 0, f"Expected output to be a scalar, but got {output.dim()}-dimensional tensor"
    assert output.item() >= 0, f"Expected non-negative loss value, but got {output.item()}"




# L2Loss forward method with mixed data types in tensors
def test_l2loss_forward_mixed_data_types(mock_config):

    l2_loss = L2Loss(mock_config)
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1, 2, 3])
    result = l2_loss.forward(predictions, targets)
    assert result.item() == 0.0
    assert result.dtype == torch.float



# L2Loss forward method with tensors requiring gradients
def test_l2loss_forward_gradients(mock_config, mock_predictions_targets_requires_grad):
    l2_loss = L2Loss(mock_config)
    predictions, targets = mock_predictions_targets_requires_grad


    output = l2_loss(predictions, targets)
    output.backward()

    assert predictions.grad is not None
    # assert targets.grad is not None
    expected_gradients = 2 * (predictions - targets) / predictions.numel()
    assert torch.allclose(predictions.grad, expected_gradients), \
        f"Expected gradients: {expected_gradients}, but got: {predictions.grad}"





