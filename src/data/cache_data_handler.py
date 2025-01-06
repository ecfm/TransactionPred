# cached_data_handler.py

import os
import pickle
import hashlib
import logging
from typing import Dict, Tuple
import json

logger = logging.getLogger(__name__)

class CachedDataHandler:
    def __init__(self):
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct paths relative to the current file
        self.data_module_src_path = current_dir
        self.processed_data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'processed'))
        self.src_files_hash = self._calculate_src_files_hash()

    def _calculate_src_files_hash(self) -> str:
        """Calculate a combined hash of all Python files in the data module source directory."""
        hasher = hashlib.md5()
        for root, _, files in os.walk(self.data_module_src_path):
            for file in sorted(files):  # Sort to ensure consistent order
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
        return hasher.hexdigest()

    def save_processed_data(self, file_path: str, data_to_save: Dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_to_save['src_files_hash'] = self.src_files_hash
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_processed_data(self, config: Dict, raw_data_file_path: str) -> Dict:
        """.
        This function first calculates the hashes for the configuration and data using the `get_hashes` method.
        The path to the cached data is constructed with the calculated hashes. If the cached data file:
            Exists: checks if the source data has been modified:
                Modified: return None
                NOT modified: loads the cached data using the `load_processed_data` method.
            NOT exist, return None
        """
        config_hash, data_hash = self.get_hashes(config.dict(), raw_data_file_path)
        cached_data_path = os.path.join(self.processed_data_path, f"{config_hash}_{data_hash}.pkl")
        if (not os.path.exists(cached_data_path)) or self.is_data_module_src_modified(cached_data_path):
            return cached_data_path, None
        with open(cached_data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        return cached_data_path, loaded_data

    def is_data_module_src_modified(self, cached_data_path: str) -> bool:
        with open(cached_data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        saved_src_files_hash = loaded_data.get('src_files_hash', '')
        if saved_src_files_hash != self.src_files_hash:
            logger.info("Data processing code has been modified. Cached data is outdated. Re-processing data.")
            return True
        return False

    def get_hashes(self, config: Dict, file_path: str) -> Tuple[str, str]:
        # Get config hash
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Get data hash
        hasher = hashlib.md5()
        chunk_size = 500 * 1024 * 1024  # 500MB chunks
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        data_hash = hasher.hexdigest()
        return config_hash, data_hash
