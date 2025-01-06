import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

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
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted = False

    def fit(self, df: pd.DataFrame, feature: str):
        raise NotImplementedError

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        raise NotImplementedError

@FeatureProcessorRegistry.register('min_max_scaler')
class MinMaxFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler = MinMaxScaler(**kwargs)

    def fit(self, df: pd.DataFrame, feature: str):
        self.scaler.fit(df[[feature]])
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        return pd.Series(self.scaler.transform(df[[feature]]).flatten(), index=df.index)

@FeatureProcessorRegistry.register('quantile_transformer')
class QuantileFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformer = QuantileTransformer(**kwargs)

    def fit(self, df: pd.DataFrame, feature: str):
        self.transformer.fit(df[[feature]])
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        return pd.Series(self.transformer.transform(df[[feature]]).flatten(), index=df.index)

@FeatureProcessorRegistry.register('binning')
class BinningFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, n_bins=10, strategy='uniform', **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.strategy = strategy
        self.bins = None

    def fit(self, df: pd.DataFrame, feature: str):
        if self.strategy == 'uniform':
            self.bins = np.linspace(df[feature].min(), df[feature].max(), self.n_bins + 1)
        elif self.strategy == 'quantile':
            self.bins = np.percentile(df[feature], np.linspace(0, 100, self.n_bins + 1))
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        return pd.cut(df[feature], bins=self.bins, labels=False)

class GroupedQuantileProcessorRegistry:
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

@GroupedQuantileProcessorRegistry.register('brand')
@GroupedQuantileProcessorRegistry.register('user')
@GroupedQuantileProcessorRegistry.register('brand_user')
class GroupedQuantileProcessor(BaseFeatureProcessor):
    def __init__(self, n_quantiles=10, group_by='brand', **kwargs):
        super().__init__(**kwargs)
        self.n_quantiles = n_quantiles
        self.group_by = group_by
        self.quantiles = {}

    def fit(self, df: pd.DataFrame, feature: str):
        if self.group_by == 'brand_user':
            grouped = df.groupby(['brand', 'user_id'])
        else:
            grouped = df.groupby(self.group_by)
        
        self.quantiles = grouped[feature].quantile(np.linspace(0, 1, self.n_quantiles + 1))
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        
        def assign_quantile(row):
            if self.group_by == 'brand_user':
                group_quantiles = self.quantiles.loc[row['brand'], row['user_id']]
            elif self.group_by == 'brand':
                group_quantiles = self.quantiles.loc[row['brand']]
            else:  # user
                group_quantiles = self.quantiles.loc[row['user_id']]
            return pd.cut([row[feature]], bins=group_quantiles, labels=False)[0]

        return df.apply(assign_quantile, axis=1)

@FeatureProcessorRegistry.register('brand_to_id')
class BrandToIdProcessor(BaseFeatureProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brand_to_id = {}

    def fit(self, df: pd.DataFrame, feature: str):
        unique_brands = df[feature].unique()
        self.brand_to_id = {brand: idx for idx, brand in enumerate(unique_brands)}
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        return df[feature].map(self.brand_to_id).fillna(-1).astype(int)

@FeatureProcessorRegistry.register('grouped_quantile')
class GroupedQuantileFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, group_by='brand', **kwargs):
        super().__init__(**kwargs)
        self.group_by = group_by
        self.processor = GroupedQuantileProcessorRegistry.get(group_by)(**kwargs)

    def fit(self, df: pd.DataFrame, feature: str):
        self.processor.fit(df, feature)
        self.fitted = True

    def transform(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        return self.processor.transform(df, feature)