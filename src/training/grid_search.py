import logging
import os

import git
import hydra
import mlflow
import numpy as np
import optuna
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.config.config_manager import ConfigManager
from src.config.grid_search_config import GridSearchConfig, TrialConfig
from src.data.data_context import DataContext
from src.models.models import create_model
from src.training.train import train_and_evaluate

CWD = os.getcwd()
TEMP_OUT_DIR = os.path.join(CWD, 'outputs', 'tmp')

def log_git_info(logger):
    """Log the current git head and any uncommitted changes."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Current git HEAD: {sha}")
    
    if repo.is_dirty():
        logger.info("There are uncommitted changes:")
        for item in repo.index.diff(None):
            logger.info(f"  {item.a_path}")

def sample_hyperparameters(trial, hyperparameters):
    """Sample hyperparameters based on the hyperparameter configuration."""
    sampled_params = {}

    def sample_param(name, config):
        if isinstance(config, dict) and 'type' in config:
            if config['type'] == 'int':
                return trial.suggest_int(name, config['min'], config['max'])
            elif config['type'] == 'float':
                return trial.suggest_float(name, config['min'], config['max'])
            elif config['type'] == 'loguniform':
                return trial.suggest_float(name, config['min'], config['max'], log=True)
            elif config['type'] == 'categorical':
                return trial.suggest_categorical(name, config['options'])
        elif isinstance(config, dict):
            return {k: sample_param(f"{name}.{k}", v) for k, v in config.items()}
        return config

    # Sample parameters for model, train, and data configurations
    for category in ['model', 'train', 'data']:
        if category in hyperparameters.dict():
            sampled_params[category] = sample_param(category, hyperparameters.dict()[category])

    # Dynamically generate hidden dimensions for concat_layer
    if 'model' in sampled_params and 'concat_layer' in sampled_params['model']:
        n_layers = sampled_params['model']['concat_layer'].pop('n_layers', 1)
        hidden_dim_config = hyperparameters.dict()['model']['concat_layer']['hidden_dim']
        
        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f'hidden_dim_{i}', hidden_dim_config['min'], hidden_dim_config['max'])
            hidden_dims.append(dim)

        sampled_params['model']['concat_layer']['hidden_dims'] = hidden_dims
        # Remove the temporary parameter if it exists
        sampled_params['model']['concat_layer'].pop('hidden_dim', None)

    return sampled_params

def update_config_with_params(config_dict, params):
    """Recursively update configuration dictionary with sampled parameters."""
    for key, value in params.items():
        if isinstance(value, dict):
            if key not in config_dict:
                config_dict[key] = {}
            update_config_with_params(config_dict[key], value)
        else:
            config_dict[key] = value

def adjust_d_model_and_nhead(params):
    """Adjust d_model to ensure it's divisible by 2 and nhead."""
    if 'model' in params and 'd_model' in params['model'] and 'transformer' in params['model'] and 'nhead' in params['model']['transformer']:
        d_model = params['model']['d_model']
        nhead = params['model']['transformer']['nhead']
        if nhead % 2 == 0:
            least_common_multiple = nhead
        else:
            least_common_multiple = nhead * 2
        # Ensure d_model is divisible by nhead and 2
        d_model = (d_model // least_common_multiple) * least_common_multiple
        
        # Update d_model
        params['model']['d_model'] = d_model
    return params

def recursive_tensor_to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    else:
        return {k: recursive_tensor_to_numpy(v) for k, v in input.items()}

def objective(trial, config, data_context, device, best_val_loss):
    """Optuna objective function for hyperparameter optimization."""
    with mlflow.start_run(nested=True):
        # Sample hyperparameters
        sampled_hyperparams = sample_hyperparameters(trial, config.hyperparameters)
        sampled_hyperparams = adjust_d_model_and_nhead(sampled_hyperparams)
        trial_dict = config.trial.dict()
        # Update the trial configuration with sampled parameters
        for hyperparam_type, hyperparams in sampled_hyperparams.items():
            update_config_with_params(trial_dict[hyperparam_type], hyperparams)

        # Convert the updated trial configuration to TrialConfig
        trial_config = TrialConfig(**trial_dict)
        
        # Initialize model
        model = create_model(trial_config.model, data_context)

        # Train and evaluate model
        train_losses, val_losses, test_loss, test_outputs, best_model = train_and_evaluate(
            model, data_context, trial_config, device
        )

        # Log metrics
        flattened_params = {f"{category}.{k}": v for category, params in sampled_hyperparams.items() for k, v in params.items()}
        mlflow.log_params(flattened_params)
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("current_best_val_loss", min(val_losses))
        mlflow.log_metric("test_loss", test_loss)

        for epoch, loss in enumerate(train_losses):
            mlflow.log_metric("train_loss", loss, step=epoch)
        for epoch, loss in enumerate(val_losses):
            mlflow.log_metric("val_loss", loss, step=epoch)

        # Check if the current validation loss is the best so far
        current_val_loss = min(val_losses)
        if current_val_loss < best_val_loss[0]:
            best_val_loss[0] = current_val_loss        
            train_loader = data_context.get_dataloader('train')
            # Save the best model with signature
            batch = next(iter(train_loader))

            # Infer the signature using the input and target
            signature = mlflow.models.infer_signature(recursive_tensor_to_numpy(batch['inputs']), recursive_tensor_to_numpy(batch['targets']['sequences']))
            
            mlflow.pytorch.log_model(best_model, "best_model", signature=signature)
            # Save and log the test outputs for later error analysis
            test_outputs_path = os.path.join(TEMP_OUT_DIR, "best_test_outputs.npy")
            np.save(test_outputs_path, test_outputs)
            mlflow.log_artifact(test_outputs_path)

    return current_val_loss

def load_previous_best(study_name, storage):
    """Load the best validation loss and test outputs from previous trials if they exist."""
    best_val_loss = [float('inf')]
    try:
        # Try to load existing study
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_trial = study.best_trial
        best_val_loss = [best_trial.value]
    except KeyError:
        # Create new study if not exists
        study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage)
    except ValueError:
        study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage, load_if_exists=True)
    return study, best_val_loss

def run_study(config, data_context, device, logger):
    """Run Optuna study for hyperparameter optimization."""
    study_name = config.experiment
    
    # Set up Optuna storage
    if config.optuna.storage.type == 'local':
        storage = f"sqlite:///{config.optuna.storage.path}"
    elif config.optuna.storage.type == 'database':
        storage = config.optuna.storage.url
    else:
        raise ValueError(f"Invalid Optuna storage type: {config.optuna.storage.type}")
    
    study, best_val_loss = load_previous_best(study_name, storage)
    study.optimize(lambda trial: objective(trial, config, data_context, device, best_val_loss),
                   n_trials=config.n_trials)

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    return study


@hydra.main(config_path="../../conf", config_name="config", version_base="1.4")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Use ConfigManager to apply defaults and validate
    validated_config = ConfigManager.load_and_validate_config(config_dict, GridSearchConfig)
    ConfigManager.setup_logging(validated_config.logger)
    # Set up logging based on the validated configuration
    logger = logging.getLogger(__name__)

    # Log git information
    log_git_info(logger)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create DataContext
    data_context = DataContext(validated_config.data)
    original_cwd = get_original_cwd()
    data_file_path = os.path.join(original_cwd, validated_config.data.file_path)
    data_context.prepare_data(data_file_path)

    # Set up MLflow
    if validated_config.mlflow.storage.type == 'local':
        mlflow.set_tracking_uri(f"file:{validated_config.mlflow.storage.path}")
    elif validated_config.mlflow.storage.type == 'database':
        mlflow.set_tracking_uri(validated_config.mlflow.storage.url)
    else:
        raise ValueError(f"Invalid MLflow storage type: {validated_config.mlflow.storage.type}")

    mlflow.set_experiment(validated_config.experiment)
    
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        # Log parent run information
        mlflow.log_params({
            "config": validated_config.dict()
        })

        # Run hyperparameter search
        study = run_study(validated_config, data_context, device, logger)

        # Log the best parameters and metrics
        mlflow.log_metric("best_val_loss", study.best_value)
        
        # Log best trial's parameters
        best_params = study.best_params
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)

        logger.info("Hyperparameter search completed.")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")

    return study

if __name__ == "__main__":
    main()
