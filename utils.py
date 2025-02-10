import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=['Ano ingresso', 'origem']):
        self.feature_to_drop = feature_to_drop
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            return df.drop(columns=self.feature_to_drop)
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=['Fase', 'IAA', 'IEG', 'IPS', 'IDA', 'Matem', 'Portug', 'Inglês', 'IPV', 'IAN', 'Idade', 'IPP', 'Defasagem', 'INDE']):
        self.min_max_scaler = min_max_scaler
        self.scaler = MinMaxScaler()
    
    def fit(self, df, y=None):
        self.scaler.fit(df[self.min_max_scaler])
        return self
    
    def transform(self, df):
        if set(self.min_max_scaler).issubset(df.columns):
            df[self.min_max_scaler] = self.scaler.transform(df[self.min_max_scaler])
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['Gênero', 'Instituição de ensino']):
        self.OneHotEncoding = OneHotEncoding
        self.encoder = OneHotEncoder(handle_unknown='ignore')
    
    def fit(self, df, y=None):
        self.encoder.fit(df[self.OneHotEncoding])
        return self
    
    def transform(self, df):
        if set(self.OneHotEncoding).issubset(df.columns):
            encoded_df = pd.DataFrame(self.encoder.transform(df[self.OneHotEncoding]).toarray(),
                                      columns=self.encoder.get_feature_names_out(self.OneHotEncoding),
                                      index=df.index)
            df = df.drop(columns=self.OneHotEncoding)
            return pd.concat([df, encoded_df], axis=1)
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='Status_entrada'):
        self.target_col = target_col  
        self.oversampler = SMOTE(sampling_strategy='minority')
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        if self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            X_bal, y_bal = self.oversampler.fit_resample(X, y)
            return pd.concat([pd.DataFrame(X_bal, columns=X.columns), pd.Series(y_bal, name=self.target_col)], axis=1)
        else:
            raise ValueError(f"A coluna alvo '{self.target_col}' não está no DataFrame.")

def pipeline(df):
    pipe = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('one_hot_encoding', OneHotEncodingNames()),
        ('min_max_scaler', MinMax()),
        ('oversample', Oversample())
    ])
    return pipe.fit_transform(df)
