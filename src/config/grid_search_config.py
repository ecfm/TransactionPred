from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, root_validator

from src.config.data_config import DataContextConfig
from src.config.train_config import TrainConfig
from src.config.model_config import NeuralNetModelConfig

class LoggerConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: str = None  # If None, logging will only print to stdout

class StorageConfig(BaseModel):
    type: str
    path: str

class MLflowConfig(BaseModel):
    storage: StorageConfig

class OptunaConfig(BaseModel):
    storage: StorageConfig

class HyperparameterConfig(BaseModel):
    type: str
    min: Optional[Union[float, int]] = None
    max: Optional[Union[float, int]] = None
    options: Optional[List[Any]] = None

    @root_validator(pre=True)
    def validate_fields(cls, values):
        if values.get('type') in ['int', 'float', 'loguniform']:
            if 'min' not in values or 'max' not in values:
                raise ValueError("Numerical hyperparameters must have 'min' and 'max' fields")
        elif values.get('type') == 'categorical':
            if 'options' not in values:
                raise ValueError("Categorical hyperparameters must have an 'options' field")
        else:
            raise ValueError(f"Invalid hyperparameter type: {values.get('type')}")
        return values

class ModelHyperparameterConfig(BaseModel):
    d_model: HyperparameterConfig
    transformer: Dict[str, HyperparameterConfig]
    encoders: Dict[str, Dict[str, HyperparameterConfig]]
    concat_layer: Dict[str, HyperparameterConfig]
    max_seq_length: HyperparameterConfig
    dropout: HyperparameterConfig

class TrainHyperparameterConfig(BaseModel):
    learning_rate: HyperparameterConfig

class HyperparameterConfig(BaseModel):
    model: ModelHyperparameterConfig
    train: TrainHyperparameterConfig

class TrialConfig(BaseModel):
    model: NeuralNetModelConfig
    train: TrainConfig

class GridSearchConfig(BaseModel):
    experiment: str
    n_trials: int
    data: DataContextConfig
    logger: LoggerConfig = LoggerConfig()
    mlflow: MLflowConfig
    optuna: OptunaConfig
    hyperparameters: HyperparameterConfig
    trial: TrialConfig

    class Config:
        extra = "allow"  # This allows for additional fields from Hydra composition
    
    def hyperparameter_items(self):
        return self.hyperparameters.dict(by_alias=True)