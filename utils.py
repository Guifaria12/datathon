from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

class DropFeatures(BaseEstimator,TransformerMixin):
def __init__(self,feature_to_drop = ['Ano ingresso', 'origem']):
        self.feature_to_drop = feature_to_drop
def fit(self,df):
        return self
def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class MinMax(BaseEstimator,TransformerMixin):
def __init__(self,min_max_scaler  = ['Fase', 'IAA', 'IEG', 'IPS', 'IDA', 'Matem', 'Portug', 'Inglês', 'IPV', 'IAN', 'Idade', 'IPP', 'Defasagem', 'INDE']):
        self.min_max_scaler = min_max_scaler
def fit(self,df):
        return self
def transform(self,df):
        if (set(self.min_max_scaler ).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler] = min_max_enc.fit_transform(df[self.min_max_scaler ])
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class OneHotEncodingNames(BaseEstimator,TransformerMixin):
def __init__(self,OneHotEncoding = ['Gênero', 'Instituição de ensino']):

        self.OneHotEncoding = OneHotEncoding

def fit(self,df):
        return self

def transform(self,df):
        if (set(self.OneHotEncoding).issubset(df.columns)):
            # função para one-hot-encoding das features
def one_hot_enc(df,OneHotEncoding):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[OneHotEncoding])
                # obtendo o resultado dos nomes das colunas
                feature_names = one_hot_enc.get_feature_names_out(OneHotEncoding)
                # mudando o array do one hot encoding para um dataframe com os nomes das colunas
                df = pd.DataFrame(one_hot_enc.transform(df[self.OneHotEncoding]).toarray(),
                                  columns= feature_names,index=df.index)
                return df

            # função para concatenar as features com aquelas que não passaram pelo one-hot-encoding
def concat_with_rest(df,one_hot_enc_df,OneHotEncoding):
                # get the rest of the features
                outras_features = [feature for feature in df.columns if feature not in OneHotEncoding]
                # concaternar o restante das features com as features que passaram pelo one-hot-encoding
                df_concat = pd.concat([one_hot_enc_df, df[outras_features]],axis=1)
                return df_concat

            # one hot encoded dataframe
            df_OneHotEncoding = one_hot_enc(df,self.OneHotEncoding)

            # retorna o dataframe concatenado
            df_full = concat_with_rest(df, df_OneHotEncoding,self.OneHotEncoding)
            return df_full

        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class Oversample(BaseEstimator, TransformerMixin):
def __init__(self, target_col='Status_entrada'):
        self.target_col = target_col  # Define a coluna alvo

def fit(self, df, y=None):
        return self  # Como não treinamos nada, apenas retorna a instância

def transform(self, df):
        if self.target_col in df.columns:
            oversample = SMOTE(sampling_strategy='minority')

            # Separar features e target
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]

            # Aplicar SMOTE
            X_bal, y_bal = oversample.fit_resample(X, y)

            # Criar novo DataFrame balanceado
            df_bal = pd.concat([pd.DataFrame(X_bal, columns=X.columns), pd.Series(y_bal, name=self.target_col)], axis=1)

            return df_bal
        else:
            raise ValueError(f"A coluna alvo '{self.target_col}' não está no DataFrame.")

class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Mau' in df.columns:
            # função smote para superamostrar a classe minoritária para corrigir os dados desbalanceados
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'Status_entrada'], df['Status_entrada'])
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print("O target não está no DataFrame")
            return df
