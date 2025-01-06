import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config.data_config import DataContextConfig
from src.data.cache_data_handler import CachedDataHandler
from src.data.feature_processors import FeatureProcessorRegistry
from src.data.sequence_generators import SequenceGeneratorRegistry

logger = logging.getLogger(__name__)

class TransactionSequenceDataset(Dataset):
    def __init__(self, sequences: Dict[str, List]):
        self.input_sequences = sequences['input'] # List[Dict[str, List]], where each Dict corresponds to a User's transactions
        self.target_sequences = sequences['target']
        self.unprocessed_target_sequences = sequences['unprocessed_target']
        self.user_ids = sequences['user_id']

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return {
            'input': self.input_sequences[idx], # Dict[str, List] where each str is a feature name
            'target': self.target_sequences[idx],
            'unprocessed_target': self.unprocessed_target_sequences[idx],
            'user_id': self.user_ids[idx]
        }
    
class DataContext:
    def __init__(self, config: DataContextConfig):
        self.config = config
        self.feature_processors = self._init_feature_processors()
        self.input_generator = SequenceGeneratorRegistry.get(config.input.type)(config.input)
        self.target_generator = SequenceGeneratorRegistry.get(config.output.type)(config.output)
        self.train_overlap_users = None
        self.data_loaders = None
        
        self.cached_data_handler = CachedDataHandler()

    def _init_feature_processors(self):
        feature_processor_registry = FeatureProcessorRegistry()
        return {
            feature: feature_processor_registry.get(processor.type)(processor.params)
            for feature, processor in self.config.feature_processors.items()
        }

    def prepare_data(self, file_path: str):
        """
        Process the data and save it in DataContext. If a cached version exists and if it is up to date, it will be loaded.
        
        Args:
            file_path (str): The path to the file containing the raw data.
        
        Returns:
            None
        """
        cached_data_path, loaded_data = self.cached_data_handler.load_processed_data(self.config, file_path)
        if loaded_data:
            self.data_loaders = loaded_data['data_loaders']
            self.train_overlap_users = loaded_data['train_overlap_users']
            self.feature_processors = loaded_data['feature_processors'] 
            logger.info(f"Loaded cached data from {cached_data_path}")
        else:
            self._process_and_save_data(file_path, cached_data_path)

    def _process_and_save_data(self, file_path: str, cached_data_path: str):
        logger.info(f"Processing and saving data to {cached_data_path}")
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = self._preprocess_data(df)
        train_df, valid_df, test_df, self.train_overlap_users = self._split_data(df)
        sequences = self._generate_processed_sequences(train_df, valid_df, test_df)
        self.data_loaders = self._create_data_loaders(sequences)
        data_to_save = {
            'config': self.config.dict(),
            'data_loaders': self.data_loaders,
            'train_overlap_users': self.train_overlap_users,
            'feature_processors': self.feature_processors,
        }
        self.cached_data_handler.save_processed_data(cached_data_path, data_to_save)               

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['user_id', 'date'])
        bins = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq=self.config.time_interval)
        df['time_interval'] = pd.cut(df['date'], bins=bins, include_lowest=True)
        agg_df = df.groupby(['user_id', 'time_interval', 'brand']).agg({
            'amount': 'sum',
            'date': 'first'
        }).reset_index()
        agg_df['time_interval'] = agg_df['time_interval'].astype(str)
        return agg_df

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List]:
        user_ids = df['user_id'].unique()
        train_users, temp_users = train_test_split(user_ids, test_size=self.config.splits.val + self.config.splits.test, random_state=42)
        valid_only_users, test_only_users = train_test_split(temp_users, test_size=self.config.splits.test / (self.config.splits.val + self.config.splits.test), random_state=42)
        
        overlap_in_train_ratio = self.config.splits.train * self.config.splits.overlap
        train_only_users, train_overlap_users = train_test_split(train_users, test_size=overlap_in_train_ratio, random_state=42)
        
        valid_users = list(valid_only_users) + list(train_overlap_users)
        test_users = list(test_only_users) + list(train_overlap_users)
        
        train_df = df[df['user_id'].isin(train_users)]
        valid_df = df[df['user_id'].isin(valid_users)]
        test_df = df[df['user_id'].isin(test_users)]
        
        return train_df, valid_df, test_df, list(train_overlap_users)

    def _generate_processed_sequences(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, List[Tuple]]]:
        cutoff_config = self.config.cutoffs
        
        input_train_df = train_df[(train_df['date'] >= cutoff_config.in_start) & (train_df['date'] < cutoff_config.train['target_start'])]
        
        for feature, processor in self.feature_processors.items():
            processor.fit(input_train_df, feature)
        
        sequences = {}
        for split, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            user_ids = split_df['user_id'].unique().tolist()
            input_sequences = self._generate_input_sequences(split, split_df, user_ids)
            target_sequences = self._generate_target_sequences(split, split_df, user_ids)
            unprocessed_target_sequences = self._generate_target_sequences(split, split_df, user_ids, process_features=['brand'])
            
            sequences[split] = {
                'input': input_sequences,
                'target': target_sequences,
                'unprocessed_target': unprocessed_target_sequences,
                'user_id': user_ids
            }
        return sequences

    def _generate_input_sequences(self, split: str, df: pd.DataFrame, user_ids: List) -> List[Tuple]:
        return self.input_generator.generate(
            user_ids=user_ids,
            data=df,
            start=self.config.cutoffs.in_start,
            end=getattr(self.config.cutoffs, split)['target_start'],
            feature_processors=self.feature_processors
        )

    def _generate_target_sequences(self, split: str, df: pd.DataFrame, user_ids: List, process_features=None) -> List[Tuple]:
        return self.target_generator.generate(
            user_ids=user_ids,
            data=df,
            start=getattr(self.config.cutoffs, split)['target_start'],
            end=pd.to_datetime(getattr(self.config.cutoffs, split)['target_start']) + pd.Timedelta(self.config.time_interval),
            feature_processors=self.feature_processors,
            process_features=process_features
        )

    def _create_data_loaders(self, sequences: Dict[str, Dict[str, List]]) -> Dict[str, DataLoader]:
        data_loaders = {}
        for split, split_sequences in sequences.items():
            dataset = TransactionSequenceDataset(split_sequences)
            data_loaders[split] = DataLoader(
                dataset,
                batch_size=self.config.data_loader.batch_size,
                shuffle=(split == 'train'),
                collate_fn=self._collate_fn,
                num_workers=self.config.data_loader.num_workers
            )
        return data_loaders

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        Args:
            batch (List[Dict[str, Any]]): A list of dictionaries, each containing 'user_id', 'input', 'target', and 'unprocessed_target'.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'user_ids' (List[Any]): List of user IDs.
                - 'inputs' (Dict[str, torch.Tensor]): Collated input sequences.
                - 'targets' (Dict[str, torch.Tensor]): Collated target sequences.
                - 'unprocessed_targets' (Dict[str, torch.Tensor]): Collated unprocessed target sequences.
        """
        return {
            'user_ids': [item['user_id'] for item in batch],
            'inputs': self.input_generator.collate([item['input'] for item in batch]),
            'targets': self.target_generator.collate([item['target'] for item in batch]),
            'unprocessed_targets': self.target_generator.collate([item['unprocessed_target'] for item in batch], to_tensor=False)
        }

    def get_dataloader(self, split: str) -> DataLoader:
        """
        Get the DataLoader for a specific split.

        Args:
            split (str): The data split ('train', 'val', or 'test').

        Returns:
            DataLoader: A DataLoader that yields batches as dictionaries.
        """
        return self.data_loaders[split]

    def get_train_overlap_users(self) -> List:
        return self.train_overlap_users

    def get_feature_processors(self) -> Dict:
        return self.feature_processors