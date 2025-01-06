# src/training/train.py
import importlib
import logging

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from src.training.losses import LossRegistry

logger = logging.getLogger(__name__)

def get_object(path):
    """Dynamically import a class or function."""
    module_name, object_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)

def recursive_dict_to_device(input, device):
    # check if input is a tensor and is already in device
    if isinstance(input, torch.Tensor):
        if input.device == device:
            return input
        return input.to(device)
    for key in input.keys():
        input[key] = recursive_dict_to_device(input[key], device)
    return input
def calculate_metrics(all_outputs, all_unprocessed_targets, data_context, split, step=None):
    # Combine all batches
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_unprocessed_targets = np.concatenate(all_unprocessed_targets, axis=0)

    # Recover original features for outputs
    recovered_outputs = data_context.target_generator.recover_original_features(all_outputs, data_context.feature_processors)

    # Flatten arrays for easier processing
    outputs_flat = recovered_outputs.flatten()
    targets_flat = all_unprocessed_targets.flatten()

    # Calculate binary classifications
    outputs_binary = (outputs_flat != 0).astype(int)
    targets_binary = (targets_flat != 0).astype(int)

    # Calculate false positive and false negative rates
    false_positive_rate = np.sum((targets_binary == 0) & (outputs_binary != 0)) / np.sum(targets_binary == 0)
    false_negative_rate = np.sum((targets_binary != 0) & (outputs_binary == 0)) / np.sum(targets_binary != 0)

    # Calculate MAE for non-zero targets 
    positive_mae = np.mean(np.abs(outputs_flat[targets_binary != 0] - targets_flat[targets_binary != 0]))

    # Calculate percentage error (only for non-zero target and non-zero output)
    mask = (targets_flat != 0) & (outputs_flat != 0)
    pe = np.mean(np.abs((targets_flat[mask] - outputs_flat[mask]) / targets_flat[mask])) * 100

    mlflow.log_metric(f"{split}_false_positive_rate", false_positive_rate, step=step)
    mlflow.log_metric(f"{split}_false_negative_rate", false_negative_rate, step=step)
    mlflow.log_metric(f"{split}_positive_mae", positive_mae, step=step)
    mlflow.log_metric(f"{split}_percentage_error", pe, step=step)

    logger.info(f"{split}_false_positive_rate: {false_positive_rate:.4f}")
    logger.info(f"{split}_false_negative_rate: {false_negative_rate:.4f}")
    logger.info(f"{split}_positive_mae: {positive_mae:.4f}")
    logger.info(f"{split}_percentage_error: {pe:.4f}")
    return {
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'positive_mae': positive_mae,
        'percentage_error': pe
    }

def evaluation_loop(model, data_context, loss, config, device, split, epoch=None, train_mode=False):
    """
    Note on batch structure:
    Each batch from the data loader is a dictionary with the following keys:
    - 'user_ids': List of user IDs
    - 'inputs': Dict[str, torch.Tensor] containing input sequences
    - 'targets': Dict[str, torch.Tensor] containing target sequences
    - 'unprocessed_targets': Dict[str, torch.Tensor] containing unprocessed target sequences
    The exact structure of the tensors depends on the SequenceGenerator used.
    """
    data_loader = data_context.get_dataloader(split)
    total_loss = 0
    all_outputs = []
    all_unprocessed_targets = []
        
    for batch in tqdm(data_loader, desc=f"{split} mode", leave=False):
        inputs = recursive_dict_to_device(batch['inputs'], device)
        outputs = model(inputs)
        detached_outputs = outputs.detach().cpu()
        all_outputs.append(detached_outputs.numpy())
        targets = batch['targets']['sequences'].to(device)
        unprocessed_targets = batch['unprocessed_targets']['sequences']
        all_unprocessed_targets.append(unprocessed_targets)
        loss_tensor = loss(outputs, targets)
        if train_mode and hasattr(model, 'backward'):
            model.backward(loss_tensor)
        total_loss += loss_tensor.item() if hasattr(loss_tensor, 'item') else loss_tensor
    
    avg_loss = total_loss / len(data_loader)
    mlflow.log_metric(f"{split}_loss", avg_loss, step=epoch)
    
    metrics = calculate_metrics(all_outputs, all_unprocessed_targets, data_context, split, step=epoch)
    
    if split == 'test':
        return avg_loss, all_outputs
    return avg_loss

def evaluate_model(model, data_context, loss, config, device, split="val", epoch=None):
    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad():
        return evaluation_loop(model, data_context, loss, config, device, split, epoch)

def train_and_evaluate(model, data_context, config, device):
    """Train the model with early stopping and evaluate on test set."""
    if device and hasattr(model, 'to'):
        model.to(device)
    
    optimizer = None
    if config.train.optimizer:
        optimizer_class = get_object(config.train.optimizer.name)
        optimizer = optimizer_class(model.parameters(), **config.train.optimizer.params)
    
    scheduler = None
    if optimizer:
        scheduler_class = get_object(config.train.scheduler.name)
        scheduler = scheduler_class(optimizer, **config.train.scheduler.params)

    loss = LossRegistry.get(config.train.loss)(config.train.loss)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(config.train.num_epochs):
        # Training
        if hasattr(model, 'train'):
            model.train()
        if optimizer:
            optimizer.zero_grad()
        avg_loss = evaluation_loop(model, data_context, loss, config, device, "train", epoch, train_mode=True)
        if optimizer:
            optimizer.step()
        train_losses.append(avg_loss)

        # Validation
        val_avg_loss  = evaluate_model(model, data_context, loss, config, device, "val", epoch)
        val_losses.append(val_avg_loss)

        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_avg_loss)

        # Early stopping check
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_model = model.copy() if hasattr(model, 'copy') else model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.train.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Test best model
    logger.info("Evaluating best model on test set")
    test_avg_loss, test_outputs = evaluate_model(best_model, data_context, loss, config, device, "test")
          
    return train_losses, val_losses, test_avg_loss, test_outputs, best_model
