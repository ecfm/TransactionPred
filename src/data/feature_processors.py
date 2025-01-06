import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

class FeatureProcessorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(processor_class):
            cls._registry[name] = processor_class
            return processor_class
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

class BaseFeatureProcessor:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.fitted = False

    def fit(self, df: pd.DataFrame, feature: str):
        raise NotImplementedError

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        raise NotImplementedError
    
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError
    
    def get_sos_token(self):
        raise NotImplementedError    

    def get_eos_token(self):
        raise NotImplementedError

@FeatureProcessorRegistry.register('no_op')
class NoOpFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        if 'name' in params:
            if params['name'] == 'time':
                self.sos_token = params['start']
                self.eos_token = params['end']
            elif params['name'] == 'brand':
                self.sos_token = '<SOS>'
                self.eos_token = '<EOS>'
            elif params['name'] == 'amount':
                self.sos_token = -1
                self.eos_token = 2
        else:
            raise NotImplementedError

    def fit(self, df: pd.DataFrame, feature: str):
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        return df[feature]

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        return series
    
    def get_sos_token(self):
        return self.sos_token

    def get_eos_token(self):
        return self.eos_token
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Optional
import numpy as np
from src.data.feature_processors import FeatureProcessorRegistry, BaseFeatureProcessor

@FeatureProcessorRegistry.register('min_max_scaler')
class MinMaxFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, params: Dict[str, Any], groupby: bool = False):
        super().__init__(params)
        self.groupby = groupby
        if self.groupby:
            self.scalers = {} # A dictionary to store the MinMaxScaler object for each feature
        else:
            self.scaler = MinMaxScaler(**params)
        self.sos_token = -1
        self.eos_token = 2

    def fit(self, df: pd.DataFrame, feature: str, groupby_feature: str = None):
        if self.groupby:
            unique_groups = df[groupby_feature].unique()
            for group in unique_groups:
                group_data = df[df[groupby_feature] == group][[feature]]
                scaler = MinMaxScaler(**self.params)
                scaler.fit(group_data)
                self.scalers[group] = scaler
        else:
            self.scaler.fit(df[[feature]])
        self.fitted = True

    def transform(self, df: pd.DataFrame =None, feature: str='amount', groupby_feature: str = None) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if groupby_feature and len(df)>0:
            transformed_values = pd.Series(index=df.index, dtype=float)
            for brand, scaler in self.scalers.items():
                mask = df[groupby_feature] == brand
                transformed_values[mask] = pd.Series(scaler.transform(df[mask][[feature]]).flatten(), index=df[mask].index)
            return transformed_values
        else:
            return pd.Series(self.scaler.transform(df[[feature]]).flatten(), index=df.index)

    def inverse_transform(self, series: pd.Series, df: pd.DataFrame =None, groupby_feature: str = None) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before inverse_transform")
        
        if groupby_feature:
            inverse_transformed_values = pd.Series(index=series.index, dtype=float)
            for brand, scaler in self.scalers.items():
                mask = df[groupby_feature] == brand
                inverse_transformed_values[mask] = pd.Series(scaler.inverse_transform(series[mask].values.reshape(-1, 1)).flatten(), index=series[mask].index)
            return inverse_transformed_values
        else:
            return pd.Series(self.scaler.inverse_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)

    def get_sos_token(self):
        return self.sos_token

    def get_eos_token(self):
        return self.eos_token

@FeatureProcessorRegistry.register('brand_to_id')
class BrandToIdProcessor(BaseFeatureProcessor):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.value_to_id = {'<SOS>': 0, '<EOS>': 1, '<UNK>': 2}
        self.id_to_value = {v: k for k, v in self.value_to_id.items()}
        self.sos_token = self.value_to_id['<SOS>']
        self.eos_token = self.value_to_id['<EOS>']
        self.unk_token = self.value_to_id['<UNK>']
        self.top_n = params.get("top_n", None)
        self.freq_threshold = params.get("freq_threshold", None)
    
    def _filter_top_brands(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Filters the DataFrame to include only the top brands."""
        if self.top_n is not None or self.freq_threshold is not None:
            brand_counts = df[feature].value_counts()
            if self.top_n is not None:
                top_brands = brand_counts.nlargest(self.top_n).index
            else:
                threshold = len(df) * self.freq_threshold
                top_brands = brand_counts[brand_counts >= threshold].index
            df = df[df[feature].isin(top_brands)]
        return df

        self.top_n = params.get("top_n", None)
        self.freq_threshold = params.get("freq_threshold", None)

    def _filter_top_brands(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Filters the DataFrame to include only the top brands."""
        if self.top_n is not None or self.freq_threshold is not None:
            brand_counts = df[feature].value_counts()
            if self.top_n is not None:
                top_brands = brand_counts.nlargest(self.top_n).index
            else:
                threshold = len(df) * self.freq_threshold
                top_brands = brand_counts[brand_counts >= threshold].index
            df = df[df[feature].isin(top_brands)]
        return df

    def fit(self, df: pd.DataFrame, feature: str):

        filtered_df = self._filter_top_brands(df, feature)
        
        unique_brands = filtered_df[feature].unique()

        start_index = max(self.value_to_id.values()) + 1
        self.value_to_id.update({brand: i + start_index for i, brand in enumerate(unique_brands)})
        self.id_to_value.update({i + start_index: brand for i, brand in enumerate(unique_brands)})
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        return df[feature].map(self.value_to_id).fillna(self.unk_token).astype(int)
    
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before inverse_transform")
        return pd.Series([self.id_to_value[i] for i in series])
    
    def get_sos_token(self):
        return self.sos_token

    def get_eos_token(self):
        return self.eos_token
    
    def get_unk_token(self):
        return self.unk_token


@FeatureProcessorRegistry.register('time_delta')
class TimeDeltaEncoder(BaseFeatureProcessor):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.earliest_date = None
        self.sos_token = None
        self.eos_token = None

    def fit(self, df: pd.DataFrame, feature: str):
        # No fitting required for date encoding
        self.earliest_date = df[feature].min()
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        return (df[feature] - self.earliest_date).dt.days

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        return self.earliest_date + pd.to_timedelta(series, unit='d')

    def single_transform(self, date: str):
        return (pd.to_datetime(date) - self.earliest_date).days
    
    def set_sos_token(self, start_date: str):
        self.sos_token = self.single_transform(start_date)

    def set_eos_token(self, end_date: str):
        self.eos_token = self.single_transform(end_date)

    def get_sos_token(self):
        if self.sos_token is None:
            raise ValueError("Special token (start_cutoff) has not been set")
        return self.sos_token

    def get_eos_token(self):
        if self.eos_token is None:
            raise ValueError("Special token (end_cutoff) has not been set")
        return self.eos_token
  
@FeatureProcessorRegistry.register('Time2Vec')
class Time2VecEncoder(nn.Module):
    def __init__(self, in_features, out_features, means):
        super(Time2VecEncoder, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        if means == "sine":
            self.f = torch.sin
        elif means == "cosine":
            self.f = torch.cos

    @staticmethod
    def t2v(tau, f, w, b, w0, b0, arg=None):
        if arg:
            v1 = f(torch.matmul(tau, w) + b, arg)
        else:
            v1 = f(torch.matmul(tau, w) + b)
        v2 = torch.matmul(tau, w0) + b0
        return torch.cat([v1, v2], -1)

    def forward(self, tau):
        return self.t2v(tau, self.f, self.w, self.b, self.w0, self.b0)
