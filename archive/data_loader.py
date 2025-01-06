import torch
from torch.utils.data import Dataset, DataLoader
from feature_processors import ProcessorRegistry
import os
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_and_filter_data(self, file_path):
        # Load data from CSV
        df = pd.read_csv(file_path)
        
        # Apply filters based on config
        for filter_name, filter_config in self.config['filters'].items():
            df = self.apply_filter(df, filter_name, filter_config)
        
        return df

    def split_data(self, df):
        user_ids = df['user_id'].unique()
        # Split users into train, valid, test
        train_users, temp_users = train_test_split(user_ids, test_size=self.config['val']+self.config['test'], random_state=42)
        valid_only_users, test_only_users = train_test_split(temp_users, test_size=self.config['test'], random_state=42)

        overlap_in_train_ratio = self.config['train']*self.config['overlap']
        # Further split train users for overlap
        train_only_users, train_overlap_users = train_test_split(train_users, test_size=overlap_in_train_ratio, random_state=42)
        groups = df.groupby('user_id')
        train_df = groups[groups['user_id'].isin(train_users)]
        valid_users = valid_only_users + train_overlap_users
        valid_df = groups[groups['user_id'].isin(valid_users)]
        test_users = test_only_users + train_overlap_users
        test_df = groups[groups['user_id'].isin(test_users)]
        return train_df, valid_df, test_df

    def apply_cutoffs(self, sequences, cutoff):
        return {k: [s for s in v if s[0] < cutoff] for k, v in sequences.items()}

    def process_data(self, file_path):
        df = self.load_and_filter_data(file_path)
        train_df, valid_df, test_df = self.split_data(df)

        train_sequences = self.apply_cutoffs(train_sequences, self.config['train_cutoff'])
        valid_sequences = self.apply_cutoffs(valid_sequences, self.config['valid_cutoff'])
        test_sequences = self.apply_cutoffs(test_sequences, self.config['test_cutoff'])

        return train_sequences, valid_sequences, test_sequences

class SpendingHistoryDataset(Dataset):
    def __init__(self, sequences, config):
        self.sequences = list(sequences.values())
        self.config = config
        self.time_processor = ProcessorRegistry.processors[config['time_processor']](**config['time_processor_args'])
        self.merchant_processor = ProcessorRegistry.processors[config['merchant_processor']](**config['merchant_processor_args'])
        self.amount_processor = ProcessorRegistry.processors[config['amount_processor']](**config['amount_processor_args'])
        self.data = self.preprocess_data()

    def preprocess_data(self):
        processed_data = []
        for sequence in self.sequences:
            times, merchants, amounts = zip(*sequence)
            processed_times = self.time_processor.process(times)
            processed_merchants = self.merchant_processor.process(merchants)
            processed_amounts = self.amount_processor.process(amounts)
            processed_data.append((processed_times, processed_merchants, processed_amounts))
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataLoaderWrapper:
    def __init__(self, sequences, config, cache_dir='./cache'):
        self.sequences = sequences
        self.config = config
        self.cache_dir = cache_dir
        self.dataset = None
        self.data_loader = None
        self.initialize()

    def initialize(self):
        cache_file = self.get_cache_filename()
        if os.path.exists(cache_file):
            self.load_processed_data(cache_file)
        else:
            self.process_and_save_data(cache_file)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle'],
            collate_fn=collate_fn,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', False)
        )

    def get_cache_filename(self):
        config_hash = hashlib.md5(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"processed_data_{config_hash}.pt")

    def process_and_save_data(self, cache_file):
        self.dataset = SpendingHistoryDataset(self.sequences, self.config)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(self.dataset.data, cache_file)

    def load_processed_data(self, cache_file):
        processed_data = torch.load(cache_file)
        self.dataset = SpendingHistoryDataset({}, self.config)  # Create empty dataset
        self.dataset.data = processed_data  # Load processed data

    def get_data_loader(self):
        return self.data_loader