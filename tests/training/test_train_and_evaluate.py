import pytest
from unittest.mock import MagicMock
import torch
from src.training.train import train_and_evaluate

@pytest.fixture
def mock_model(mocker):
    model = MagicMock()
    model.parameters.return_value = []
    model.eval.return_value = None
    model.train.return_value = None
    model.copy.return_value = model
    return model

@pytest.fixture
def mock_data_loader():
    return [create_mock_batch()] * 10

def create_mock_batch():
    # Create a mock batch with the structure expected by train_and_evaluate function
    batch = {
        'inputs': {'sequences': torch.rand(10, 32)},  # Example input data structure
        'targets': {'sequences': torch.rand(10, 32)},  # Example target data structure
        'unprocessed_targets': {'sequences': torch.rand(10, 32)},  # Example unprocessed target data
        'masks': torch.ones(10, 32).bool()  # Example mask
    }
    return batch

@pytest.fixture
def mock_config(mocker):
    config = MagicMock()
    config.train.num_epochs = 5
    config.train.patience = 2
    config.train.loss = "mse"
    config.train.optimizer.name = "torch.optim.SGD"
    config.train.optimizer.params = {"lr": 0.01}
    config.train.scheduler.name = "torch.optim.lr_scheduler.StepLR"
    config.train.scheduler.params = {"step_size": 1, "gamma": 0.1}
    return config

@pytest.fixture
def mock_device():
    return torch.device('cpu')

def test_training_loop(mock_model, mock_data_loader, mock_config, mock_device, mocker):
    # Patch optimizer and scheduler retrieval
    mocker.patch('src.training.train.get_object', side_effect=[torch.optim.SGD, torch.optim.lr_scheduler.StepLR])

    # Run the training loop and validate the outputs
    train_losses, val_losses, test_avg_loss, test_outputs, best_model = train_and_evaluate(
        mock_model, mock_data_loader, mock_data_loader, mock_data_loader, mock_config, mock_device
    )

    # Check the number of train and validation losses matches the number of epochs
    assert len(train_losses) == mock_config.train.num_epochs
    assert len(val_losses) == mock_config.train.num_epochs
    
    # Ensure early stopping is respected
    assert len(train_losses) <= mock_config.train.num_epochs
    
    # Check that the best model is returned
    assert best_model is not None

def test_early_stopping(mock_model, mock_data_loader, mock_config, mock_device, mocker):
    # Patch optimizer and scheduler retrieval
    mocker.patch('src.training.train.get_object', side_effect=[torch.optim.SGD, torch.optim.lr_scheduler.StepLR])

    # Simulate the validation loss to trigger early stopping
    def mock_eval_loop(*args, **kwargs):
        if kwargs['split'] == 'val':
            if kwargs['epoch'] < 2:
                return 1.0  # Simulated improvement in loss
            else:
                return 2.0  # Simulated stagnation in loss
        return 0.5  # Test loss

    mocker.patch('src.training.train.evaluate_model', side_effect=mock_eval_loop)

    # Run the training loop and validate early stopping
    train_losses, val_losses, test_avg_loss, test_outputs, best_model = train_and_evaluate(
        mock_model, mock_data_loader, mock_data_loader, mock_data_loader, mock_config, mock_device
    )

    # Check if early stopping was triggered after 3 epochs (2 epochs with no improvement)
    assert len(train_losses) == 3
    assert len(val_losses) == 3

def test_model_evaluation(mock_model, mock_data_loader, mock_config, mock_device, mocker):
    # Patch optimizer and scheduler retrieval
    mocker.patch('src.training.train.get_object', side_effect=[torch.optim.SGD, torch.optim.lr_scheduler.StepLR])

    # Run the model evaluation
    train_losses, val_losses, test_avg_loss, test_outputs, best_model = train_and_evaluate(
        mock_model, mock_data_loader, mock_data_loader, mock_data_loader, mock_config, mock_device
    )

    # Ensure the average loss on the test set is non-negative
    assert test_avg_loss >= 0

    # Verify the length of test outputs matches the test data loader length
    assert len(test_outputs) == len(mock_data_loader)
