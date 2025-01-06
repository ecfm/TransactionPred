from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.config.data_config import SequenceConfig
from src.utils.registry import create_registry

SequenceGeneratorRegistry = create_registry()

class SequenceGenerator:
    def __init__(self, config: SequenceConfig):
        self.config = config
        self.features = config.features

    def generate(self, data: pd.DataFrame, start, end, feature_processors: Dict, process_features=None) -> List[Tuple[Any, List]]:
        if 'date' in self.features:
            feature_processors['date'].set_sos_token(start)
            feature_processors['date'].set_eos_token(end)
        data = data[(data['date'] >= pd.to_datetime(start)) & (data['date'] < pd.to_datetime(end))]
        # Apply feature processing
        processed_data = data.copy()
        if process_features is None:
            process_features = self.features
        for feature, processor in feature_processors.items():
            if feature in self.features and feature in process_features:
                processed_data[feature] = processor.transform(data, feature)
        return processed_data

    def collate(self, sequences: List[Dict], to_tensor=True) -> Dict:
        raise NotImplementedError

    def recover_original_features(self, processed_sequences: Dict, feature_processors: Dict) -> Dict:
        raise NotImplementedError
    
    def collated_sequences_to_df(self, user_ids: List, collated_sequences: Dict) -> pd.DataFrame:
        raise NotImplementedError


@SequenceGeneratorRegistry.register('continuous_time')
class ContinuousTimeSequenceGenerator(SequenceGenerator):
    def __init__(self, config: SequenceConfig):
        super().__init__(config)

    def generate(self, user_ids, data: pd.DataFrame, start: str, end: str, feature_processors: Dict, process_features=None) -> List[Dict[str, List]]:
        processed_data = super().generate(data, start, end, feature_processors, process_features)
        sequences = []
        
        for user_id in user_ids:
            group = processed_data[processed_data['user_id'] == user_id]
            seq = {feature: [feature_processors[feature].get_sos_token()] for feature in self.features}
            
            for _, row in group.iterrows():
                for feature in self.features:
                    seq[feature].append(row[feature] if feature in row else None)
            
            # Add a special end of sequence token to indicate the end-time of the sequence
            for feature in self.features:
                seq[feature].append(feature_processors[feature].get_eos_token())
            sequences.append(seq)
        return sequences

    def collate(self, sequences: List[Dict[str, List]], to_tensor=True) -> Dict:
        """
        Collate a list of sequences into a batch.

        Args:
            sequences (List[Dict[str, List]]): A list of dictionaries, each containing feature names and their corresponding lists of values.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'sequences': list of Tensors of shape (batch_size, max_seq_length, feature_dim)
                - 'masks': Tensor of shape (batch_size, max_seq_length)

        Note:
            - The 'sequences' tensor contains the padded sequences, where each entry is a list of feature values
              in the order of self.features.
            - The 'masks' tensor is a boolean mask where True indicates a valid entry and False indicates padding.
        """
        max_len = max(len(seq[self.features[0]]) for seq in sequences)

        collated_sequences = {feature: [] for feature in self.features}
        masks = []
        
        for feature_dict in sequences:
            mask = [0] * len(feature_dict[self.features[0]]) + [1] * (max_len - len(feature_dict[self.features[0]]))
            masks.append(mask)
            
            for feature in self.features:
                feature_seq = feature_dict[feature]
                pad_item = feature_seq[-1]  # Use the last item (EOS token) for padding
                padded_seq = feature_seq + [pad_item] * (max_len - len(feature_seq))
                collated_sequences[feature].append(padded_seq)
        
        if to_tensor:
            for feature in self.features:
                # Determine dtype
                if all(isinstance(x, (int, np.integer)) for x in collated_sequences[feature][0]):
                    dtype = torch.long
                elif all(isinstance(x, float) for x in collated_sequences[feature][0]):
                    dtype = torch.float
                else:
                    dtype = torch.float  # Default to float if mixed or other types

                feature_tensor = torch.tensor(collated_sequences[feature], dtype=dtype)                
                collated_sequences[feature] = feature_tensor
            
            masks = torch.tensor(masks, dtype=torch.bool)
            
        return {
            'sequences': collated_sequences,
            'masks': masks
        }
    
    def recover_original_features(self, processed_sequences: Dict, feature_processors: Dict) -> Dict:
        recovered_sequences = {feature: [] for feature in self.features}
        
        for feature in self.features:
            processor = feature_processors[feature]
            feature_data = processed_sequences['sequences'][feature]
            
            if isinstance(feature_data, torch.Tensor):
                feature_data = feature_data.cpu().numpy()
            
            recovered_data = processor.inverse_transform(pd.Series(feature_data.flatten()))
            recovered_sequences[feature] = recovered_data.values.reshape(feature_data.shape)
        
        return {
            'sequences': recovered_sequences,
            'masks': processed_sequences['masks']
        }

    def collated_sequences_to_df(self, user_ids: List, collated_sequences: Dict) -> pd.DataFrame:
        sequences = collated_sequences['sequences']
        masks = collated_sequences['masks']
        
        data = []
        for i, user_id in enumerate(user_ids):
            seq_length = (~masks[i]).sum().item() if isinstance(masks, torch.Tensor) else sum(1 for m in masks[i] if not m)
            for j in range(seq_length):
                row = {'user_id': user_id}
                for feature in self.features:
                    feature_data = sequences[feature][i]
                    if isinstance(feature_data, torch.Tensor):
                        row[feature] = feature_data[j].item()
                    else:
                        row[feature] = feature_data[j]
                data.append(row)
        
        df = pd.DataFrame(data)
        return df
     
    
class BaseMultiHotSequenceGenerator(SequenceGenerator):
    def __init__(self, config: SequenceConfig):
        super().__init__(config)
        self.id_to_value = None
    
    def generate(self, user_ids, data: pd.DataFrame, start: str, end: str, feature_processors: Dict, process_features=None) -> List[Tuple[Any, List]]:
        processed_data = super().generate(data, start, end, feature_processors, process_features)
        self.id_to_value = feature_processors['brand'].id_to_value
        
        sequences = []
        for user_id in user_ids:
            group = processed_data[processed_data['user_id'] == user_id]
            seq = self._create_sequence(group)
            sequences.append(seq)
        
        return sequences

    def _create_sequence(self, group):
        raise NotImplementedError

    def collate(self, sequences: List, to_tensor=True) -> Dict:
        if to_tensor:
            stacked_sequences = torch.stack([torch.tensor(seq) for seq in sequences])
        else:
            stacked_sequences = np.stack(sequences)
        return {
            'sequences': stacked_sequences[:, 0, :]
        }

    def recover_original_features(self, processed_sequences: Dict, feature_processors: Dict) -> Dict:
        recovered_sequences = processed_sequences.copy()
        
        processor = feature_processors['brand']
        recovered_sequences['sequences'] = processor.inverse_transform(pd.Series(recovered_sequences['sequences'].flatten())).values.reshape(recovered_sequences['sequences'].shape)
        
        return recovered_sequences

@SequenceGeneratorRegistry.register('multi_hot_brand_only')
class BrandOnlyMultiHotSequenceGenerator(BaseMultiHotSequenceGenerator):
    def _create_sequence(self, group):
        vector = [0] * len(self.id_to_value)
        for _, row in group.iterrows():
            if row['brand'] < len(self.id_to_value):
                vector[row['brand']] = 1
        return [vector]

@SequenceGeneratorRegistry.register('multi_hot_brand_amount')
class BrandAmountMultiHotSequenceGenerator(BaseMultiHotSequenceGenerator):
    
    def _create_sequence(self, group, prediction_interval=1, input_interval=1):
        vector = np.zeros(len(self.id_to_value))
        #we add the interval adjustment to handle the difference in prediction interval and input interval
        interval_adjustment = prediction_interval / input_interval
        for _, row in group.iterrows():
            brand_id = row['brand']
            amount = row['amount']
            if brand_id < len(self.id_to_value):
                vector[brand_id] += amount * interval_adjustment
        return [vector.tolist()]
    
    def recover_original_features(self, processed_sequences: Dict, feature_processors: Dict) -> Dict:
        self.amount_processor = feature_processors['amount']
        recovered_sequences = processed_sequences.copy()
        flattened = recovered_sequences.flatten()
        # For other processors, we can use inverse_transform
        recovered_amounts = self.amount_processor.inverse_transform(pd.Series(flattened)).values

        recovered_sequences = recovered_amounts.reshape(recovered_sequences.shape)
        return recovered_sequences

@SequenceGeneratorRegistry.register('multi_hot_separate')
class SeparateMultiHotSequenceGenerator(BaseMultiHotSequenceGenerator):
    def _create_sequence(self, group, prediction_interval=1, input_interval=1):
        brand_vector = [0] * len(self.id_to_value)
        amount_vector = [0] * len(self.id_to_value)
        #we add the interval adjustment to handle the difference in prediction interval and input interval
        interval_adjustment = prediction_interval / input_interval
        for _, row in group.iterrows():
            keys_list = list(self.id_to_value.keys())
            brand_index = keys_list.index(row['brand'])
            if row['brand'] < len(self.id_to_value):
                brand_vector[brand_index] = 1
                amount_vector[brand_index] += row['amount'] * interval_adjustment
        return [brand_vector, amount_vector]

    def collate(self, sequences: List) -> Dict:
        stacked_sequences = torch.stack([torch.tensor(seq) for seq in sequences])
        return {
            'brand_sequences': stacked_sequences[:, 0, :],
            'amount_sequences': stacked_sequences[:, 1, :]
        }

    def recover_original_features(self, processed_sequences: Dict, feature_processors: Dict) -> Dict:
        recovered_sequences = processed_sequences.copy()
        
        brand_processor = feature_processors['brand']
        amount_processor = feature_processors['amount']
        
        recovered_sequences['brand_sequences'] = brand_processor.inverse_transform(pd.Series(recovered_sequences['brand_sequences'].flatten())).values.reshape(recovered_sequences['brand_sequences'].shape)
        recovered_amounts = self.amount_processor.inverse_transform(pd.Series(recovered_sequences['amount_sequences'].flatten())).values
        recovered_sequences['amount_sequences'] = recovered_amounts.reshape(recovered_sequences['amount_sequences'].shape)
        
        return recovered_sequences
