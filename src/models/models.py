import math
import warnings
from typing import Union

import torch
import torch.nn as nn
from rtdl_num_embeddings import compute_bins

from src.config.model_config import (
    CoderConfig,
    EmbeddingCoderConfig,
    LinearCoderConfig,
    NeuralNetModelConfig,
    RTDLNumEmbeddingsCoderConfig,
)
from src.models.baselines import ModelRegistry
from src.models.encoders_decoders import CoderRegistry, RTDLNumEmbeddingsProcessor

warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool")

def create_model(config, data_context):
    return ModelRegistry.get(config.type)(config, data_context)

# Base model class
class NeuralNetModelBase(nn.Module):
    def __init__(self, config: NeuralNetModelConfig, data_context):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.input_features = data_context.config.input.features

        # Initialize encoders for each input feature
        self.encoders = nn.ModuleDict({
            feature: self._create_encoder(feature, encoder_config, data_context)
            for feature, encoder_config in config.encoders.items()
        })
        # TODO: this assumes one decoder, and the existence of brand
        decoder_config = config.decoder
        decoder_config.input_dim = self.d_model
        decoder_config.output_dim = len(data_context.feature_processors['brand'].id_to_value)
        self.decoder = CoderRegistry.get(decoder_config.type)(decoder_config)

        # Initialize concat layer
        total_input_dim = sum(coder.output_dim for coder in self.config.encoders.values())
        concat_config = config.concat_layer.copy()
        concat_config.input_dim = total_input_dim
        concat_config.output_dim = self.d_model
        self.concat_layer = CoderRegistry.get(concat_config.type)(concat_config)

        # Initialize the main model (to be implemented by subclasses)
        self.model = None

    def _create_encoder(self, feature: str, coder_config: CoderConfig, data_context) -> Union[
        RTDLNumEmbeddingsProcessor, CoderConfig]:
        if isinstance(coder_config, EmbeddingCoderConfig):
            # Set the number of embeddings based on the feature processor
            coder_config.num_embeddings = len(data_context.feature_processors[feature].id_to_value)

        elif isinstance(coder_config, RTDLNumEmbeddingsCoderConfig):
            # Retrieve feature data from the feature processor and convert to tensor
            feature_processor = data_context.feature_processors[feature]
            feature_data = torch.tensor(feature_processor.get_feature_data(), dtype=torch.float)

            # Determine the number of bins
            num_samples = feature_data.shape[0]
            n_bins = min(coder_config.n_bins, max(2, num_samples - 1))  # Ensure n_bins is at least 2

            # Compute bins from the feature data
            bins = compute_bins(feature_data, n_bins=n_bins)

            coder_config.output_dim = coder_config.d_embedding

            # Create and return RTDLNumEmbeddingsProcessor with computed bins
            return RTDLNumEmbeddingsProcessor(
                bins=bins,
                d_embedding=coder_config.d_embedding,
                activation=coder_config.activation
            )

        # For other coder configurations, use CoderRegistry to create the encoder
        return CoderRegistry.get(coder_config.type)(coder_config)

    def forward(self, inputs):
        encoded = {
            feature: self.encoders[feature](inputs['sequences'][feature])
            for feature in self.input_features
        }
        combined = torch.cat(list(encoded.values()), dim=-1)
        model_input = self.concat_layer(combined)

        if isinstance(self.model, nn.TransformerEncoder):
            if isinstance(self.pos_encoder, nn.Embedding):
                positions = torch.arange(model_input.size(1), device=model_input.device).unsqueeze(0)
                model_input = model_input + self.pos_encoder(positions)
            else:
                model_input = self.pos_encoder(model_input)
            model_output = self.model(model_input, src_key_padding_mask=inputs['masks'])
        else:
            raise ValueError(f"Unsupported model type: {self.model}")

        # TODO: currently using the last timestep (in ContinuousTimeSequenceGenerator, a special end of sequence token is added), try different pooling methods
        decoded_output = self.decoder(model_output[:, -1, :] )
        return decoded_output

@ModelRegistry.register("transformer-encoder")
class TransformerEncoderModel(NeuralNetModelBase):
    def __init__(self, config: NeuralNetModelConfig, data_context):
        super().__init__(config, data_context)

        self.pos_encoder = self._create_positional_encoding(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.transformer.nhead,
            dropout=config.transformer.dropout,
            batch_first=True,
            norm_first=True
        )
        self.model = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer.num_encoder_layers
        )

    def _create_positional_encoding(self, config: NeuralNetModelConfig):
        if config.positional_encoding is None:
            return nn.Identity()
        elif config.positional_encoding == 'sinusoidal':
            return SinusoidalPositionalEncoding(self.d_model, config.transformer.dropout)
        elif config.positional_encoding == 'learned':
            return nn.Embedding(config.max_seq_length, self.d_model)
        else:
            raise ValueError(f"Unknown positional encoding type: {config.positional_encoding}")

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
