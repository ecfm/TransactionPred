from typing import Optional, List, Union, Dict, Literal
from pydantic import BaseModel, Field

class BaseCoderConfig(BaseModel):
    type: str
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    activation: Optional[str] = None

class LinearCoderConfig(BaseCoderConfig):
    type: Literal['linear'] = 'linear'

class MLPCoderConfig(BaseCoderConfig):
    type: Literal['mlp'] = 'mlp'
    hidden_dims: List[int] = Field(..., description="List of hidden dimensions for each layer")
    final_activation: Optional[str] = None

class EmbeddingCoderConfig(BaseCoderConfig):
    type: Literal['embedding'] = 'embedding'
    num_embeddings: Optional[int] = None

class NoOpCoderConfig(BaseCoderConfig):
    type: Literal['no_op'] = 'no_op'

class RTDLNumEmbeddingsCoderConfig(BaseCoderConfig):
    type:str
    d_embedding: int = Field(..., description="Dimension of the embedding vector")
    activation: Optional[str] = None
    n_bins: Optional[int] = Field(default=48,
                            description="Default number of bins for discretization")


CoderConfig = Union[
    LinearCoderConfig,
    EmbeddingCoderConfig,
    MLPCoderConfig,
    NoOpCoderConfig,
    RTDLNumEmbeddingsCoderConfig
]

class TransformerConfig(BaseModel):
    nhead: int
    num_encoder_layers: int
    dropout: float

class NeuralNetModelConfig(BaseModel):
    type: str
    d_model: int
    encoders: Dict[str, CoderConfig]
    decoder: CoderConfig
    concat_layer: CoderConfig
    max_seq_length: int
    transformer: TransformerConfig
    positional_encoding: Optional[str] = None
    dropout: Optional[float] = None
