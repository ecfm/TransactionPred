from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.registry import create_registry
from src.config.model_config import LinearCoderConfig, EmbeddingCoderConfig, MLPCoderConfig
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

CoderRegistry = create_registry()


@CoderRegistry.register("no_op")
class NoOpCoder(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.dim() == 2:
            x = x.unsqueeze(-1)
        return x


class FeedFowardBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_dim = config.output_dim

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        original_shape = x.shape[:-1]
        x_reshaped = x.reshape(-1, x.shape[-1])
        output = self._forward(x_reshaped)
        return output.reshape(*original_shape, self.output_dim)


@CoderRegistry.register("embedding")
class EmbeddingCoder(nn.Module):
    def __init__(self, config: EmbeddingCoderConfig):
        super().__init__()
        self.embed = nn.Embedding(config.num_embeddings, config.output_dim)

    def forward(self, x):
        return self.embed(x)


# Base encoder and decoder classes
@CoderRegistry.register("linear")
class LinearCoder(FeedFowardBase):
    def __init__(self, config: LinearCoderConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_dim, config.output_dim)
        self.activation = getattr(F, config.activation) if config.activation else None

    def _forward(self, x):
        x = self.linear(x)
        return self.activation(x) if self.activation else x


@CoderRegistry.register("mlp")
class MLPCoder(FeedFowardBase):
    def __init__(self, config: MLPCoderConfig):
        super().__init__(config)

        layers = []
        input_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(getattr(nn, config.activation)())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, config.output_dim))

        if config.final_activation:
            layers.append(getattr(nn, config.final_activation)())

        self.mlp = nn.Sequential(*layers)

    def _forward(self, x):
        return self.mlp(x)



@CoderRegistry.register("rtdl_num_embeddings")
class RTDLNumEmbeddingsProcessor(nn.Module):
    def __init__(self, bins: List[torch.Tensor], d_embedding: int, activation: bool):
        super(RTDLNumEmbeddingsProcessor, self).__init__()


        self.embedding_layer = PiecewiseLinearEmbeddings(
            bins=bins,
            d_embedding=d_embedding,
            activation=activation
        )

        self.activation = nn.ReLU() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_before_activation = self.embedding_layer(x)  # Capture the output before activation
        if self.activation:
            x = self.activation(output_before_activation)  # Apply ReLU activation
        else:
            x = output_before_activation

        return x