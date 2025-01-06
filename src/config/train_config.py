from typing import Dict, Union

from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    name: str
    params: Dict[str, Union[float, int]]

class SchedulerConfig(BaseModel):
    name: str
    params: Dict[str, Union[float, int, str]]

class TrainConfig(BaseModel):
    num_epochs: int
    patience: int | None = None
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None
    loss: str
