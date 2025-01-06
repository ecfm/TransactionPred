from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator


class FeatureProcessorConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    groupby: Optional[bool] = False

class SequenceConfig(BaseModel):
    type: str
    features: List[str]
    separate_brand_amount: bool = False
    
class SplitConfig(BaseModel):
    train: float
    val: float
    test: float
    overlap: float

class CutoffConfig(BaseModel):
    in_start: str
    train: Dict[str, str]
    val: Dict[str, str]
    test: Dict[str, str]

class DataLoaderConfig(BaseModel):
    batch_size: int = 32
    num_workers: int = 0

class Time2VecConfig(BaseModel):
    in_features: int
    out_features: int
    means: str = Field(..., regex="^(sine|cosine)$")

class DataContextConfig(BaseModel):
    time_interval: str
    file_path: str
    splits: SplitConfig
    input: SequenceConfig
    output: SequenceConfig
    feature_processors: Dict[str, FeatureProcessorConfig]
    data_loader: DataLoaderConfig
    cutoffs: CutoffConfig

    @root_validator
    def validate_feature_processors(cls, values):
        feature_processors = values.get('feature_processors', {})
        
        # Check the "brand_to_id" processor if it exists
        brand_processor_config = feature_processors.get('brand')
        if brand_processor_config and brand_processor_config.type == "brand_to_id":
            params = brand_processor_config.params
            top_n = params.get("top_n")
            freq_threshold = params.get("freq_threshold")
            
            if top_n is not None and freq_threshold is not None:
                raise ValueError("Only one of 'top_n' and 'freq_threshold' can be set for 'brand_to_id' processor")
        
        return values
