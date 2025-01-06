import torch
import pandas as pd
from src.utils.registry import create_registry

ModelRegistry = create_registry()
class MockPyTorchModel:
    def __init__(self, config, data_context):
        self.device = 'cpu'
        self.config = config
        self.data_context = data_context

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.forward(x)

    def train(self, mode=True):
        # No-op for non-neural network models
        pass

    def eval(self):
        # No-op for non-neural network models
        pass

    def parameters(self):
        # Non-neural network models don't have parameters in the PyTorch sense
        return []

    def state_dict(self):
        # Subclasses should implement this if they have state to save
        return {}

    def load_state_dict(self, state_dict):
        # Subclasses should implement this if they have state to load
        pass

@ModelRegistry.register('baseline.avg')
class AverageModel(MockPyTorchModel):

    def forward(self, x):
        user_ids = x['user_ids']
        inputs = x['inputs']
        in_df = self.data_context.input_generator.collated_sequences_to_df(user_ids, inputs)
        
        for feature, processor in self.data_context.feature_processors.items():
            if feature in in_df.columns:
                in_df[feature] = processor.inverse_transform(in_df[feature])
        if 'brand' in in_df.columns:
            in_df = in_df[in_df['brand'] != '<EOS>']
        duration_delta = pd.to_timedelta(self.data_context.config.time_interval)
        
        # for each user, sum up the transaction amount of each brand within each duration, then calculate the average amount per brand
        return x