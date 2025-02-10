#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

#carregando os dados 
url_2022 = 'https://docs.google.com/spreadsheets/d/1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=csv&gid=90992733'
url_2023 = 'https://docs.google.com/spreadsheets/d/1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=csv&gid=555005642'
url_2024 = 'https://docs.google.com/spreadsheets/d/1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=csv&gid=215885893'

base_dados_2022 = pd.read_csv(url_2022)
base_dados_2023 = pd.read_csv(url_2023)
base_dados_2024 = pd.read_csv(url_2024)

base_dados_2022['origem'] = 'Base2022'
base_dados_2023['origem'] = 'Base2023'
base_dados_2024['origem'] = 'Base2024'

base_completa = pd.concat([base_dados_2022, base_dados_2023, base_dados_2024], ignore_index=True)
base_completa.sort_values(by='origem', inplace=True, ascending=False)
base_completa.drop_duplicates(keep='first', inplace=True)
base_completa.loc[base_completa['Ano ingresso'] == 2024, 'Status_entrada'] = 'Novato'
base_completa.loc[(base_completa['Ano ingresso'] != 2024) & (base_completa['origem'] == 'Base2024') , 'Status_entrada'] = 'Veterano'
base_completa['Status_entrada'].fillna('Desistente', inplace=True)

import numpy as np

base_completa.replace('#DIV/0!', np.nan, inplace=True)
base_completa.replace('INCLUIR', np.nan, inplace=True)
base_completa = base_completa.drop(columns=['Nº Av', 'RA', 'Avaliador1', 'Avaliador2', 'Data de Nasc', 'Nome Anonimizado', 'Fase Ideal', 'Avaliador3', 'Ativo/ Inativo', 'Ativo/ Inativo.1', 'Escola', 'Destaque IDA', 'Destaque IPV', 'Avaliador4', 'Nome', 'Destaque IEG', 'Rec Av1', 'Fase ideal', 'Atingiu PV', 'Indicado', 'Ano nasc', 'Cg', 'Cf', 'Avaliador3', 'Rec Psicologia' ,'Ct', 'Rec Av3' , 'Rec Av2', 'Turma', 'Data de Nasc', 'Avaliador6', 'Destaque IPV.1', 'Avaliador5', 'Rec Av4'])

# Substituir as vírgulas por pontos e converter para float
base_completa[['Ing', 'Inglês']] = base_completa[['Ing', 'Inglês']].replace({',': '.'}, regex=True).astype(float)
base_completa['Inglês'] = base_completa[['Ing', 'Inglês']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Ing'], inplace=True)

# Substituir as vírgulas por pontos e converter para float
base_completa[['Mat', 'Matem']] = base_completa[['Mat', 'Matem']].replace({',': '.'}, regex=True).astype(float)
base_completa['Matem'] = base_completa[['Mat', 'Matem']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Mat'], inplace=True)

# Substituir as vírgulas por pontos e converter para float
base_completa[['Por', 'Portug']] = base_completa[['Por', 'Portug']].replace({',': '.'}, regex=True).astype(float)
base_completa['Portug'] = base_completa[['Por', 'Portug']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Por'], inplace=True)

base_completa.loc[base_completa['origem'] == 'Base2024', 'Pedra'] = base_completa['Pedra 2024']
base_completa.loc[base_completa['origem'] == 'Base2023', 'Pedra'] = base_completa['Pedra 23']
base_completa.loc[base_completa['origem'] == 'Base2022', 'Pedra'] = base_completa['Pedra 22']

base_completa.drop(columns=['Pedra 20', 'Pedra 21', 'Pedra 2024', 'Pedra 23', 'Pedra 22', 'Pedra 2023'], inplace=True)

base_completa.loc[base_completa['origem'] == 'Base2024', 'INDE'] = base_completa['INDE 2024']
base_completa.loc[base_completa['origem'] == 'Base2023', 'INDE'] = base_completa['INDE 2023']
base_completa.loc[base_completa['origem'] == 'Base2022', 'INDE'] = base_completa['INDE 22']

base_completa.drop(columns=['INDE 22', 'INDE 2023', 'INDE 23', 'INDE 2024'], inplace=True)

############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> HACKTON </h1>", unsafe_allow_html = True)

st.warning(' TESTE 2 **ENVIAR** no final da página.')

# Função para calcular a idade a partir de uma data no formato dd/mm/aaaa
def calcular_idade(data, ano_referencia):
    if isinstance(data, str):  # Verifica se o valor é uma string
        try:
            # Converte a string para data
            data = pd.to_datetime(data, format='%d/%m/%Y')
            return ano_referencia - data.year
        except Exception as e:
            return None  # Retorna None se não for possível converter
    return data  # Se já for um int, retorna o valor sem alteração

# Aplicando a lógica para as duas colunas
base_completa['Idade 22'] = base_completa['Idade 22'].apply(calcular_idade, ano_referencia=2022)
base_completa['Idade'] = base_completa.apply(
    lambda row: calcular_idade(row['Idade'], 2024) if row['origem'] == 'Base2024' else calcular_idade(row['Idade'], 2023),
    axis=1
)
base_completa['Idade'] = base_completa[['Idade 22', 'Idade']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Idade 22'], inplace=True)

base_completa['Defasagem'] = base_completa[['Defas', 'Defasagem']].sum(axis=1, skipna=True)
base_completa.drop(columns=['Defas'], inplace=True)

# Separando os dados em treino e teste
def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(dados, 0.2)

#Criando novo cliente
cliente_predict_df = pd.DataFrame([novo_cliente],columns=teste_df.columns)

#Concatenando novo cliente ao dataframe dos dados de teste
teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)
# Mapeamento das classificações semelhantes
mapeamento = {
    'Privada - Programa de apadrinhamento': 'Privada',
    'Privada *Parcerias com Bolsa 100%': 'Privada',
    'Privada - Pagamento por *Empresa Parceira': 'Privada',
    'Rede Decisão': 'Privada',
    'Escola Pública': 'Pública',
    'Escola JP II': 'Pública',
    'Bolsista Universitário *Formado (a)': 'Bolsista',
    'Nenhuma das opções acima': 'Outros',
    'Concluiu o 3º EM': 'Outros'
}

# Aplicar o mapeamento para a coluna 'Instituição de ensino'
base_completa['Instituição de ensino'] = base_completa['Instituição de ensino'].replace(mapeamento)

# Verificar as classificações únicas após o agrupamento
base_completa['Instituição de ensino'].unique()

# Mapeamento das classificações semelhantes
mapeamento_genero = {
    'Menina': 'Feminino',
    'Menino': 'Masculino'
}

# Aplicar o mapeamento para a coluna 'Gênero'
base_completa['Gênero'] = base_completa['Gênero'].replace(mapeamento_genero)

# Verificar as classificações únicas após o agrupamento
base_completa['Gênero'].unique()

import re

# Função para extrair apenas números da coluna
def extrair_numero(fase):
    # Verifica se a fase é "ALFA" e retorna 0
    if str(fase).upper() == 'ALFA':
        return 0
    # Verifica se a fase é um texto como 'FASE X' e retorna o número
    match = re.search(r'\d+', str(fase))
    if match:
        return int(match.group())
    return fase  # Caso não tenha número, retorna o valor original

# Aplicando a função na coluna 'Fase'
base_completa['Fase'] = base_completa['Fase'].apply(extrair_numero)

# Verificando os valores únicos na coluna após a transformação
print(base_completa['Fase'].unique())

# Substituindo os valores de percentual para o formato desejado
base_completa['IPS'] = base_completa['IPS'].replace({r'%': '', r',': '.'}, regex=True).astype(float) / 100

# Para garantir que o valor decimal tenha vírgula
base_completa['IPS'] = base_completa['IPS'].apply(lambda x: str(x).replace('.', ','))


# Colunas que você deseja converter para float
colunas_para_float = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'IPP', 'INDE']  # Substitua pelos nomes das suas colunas

# Substituir as vírgulas por pontos e converter para float
for coluna in colunas_para_float:
    base_completa[coluna] = base_completa[coluna].replace({',': '.'}, regex=True).astype(float)

# Verificando o tipo das colunas após a conversão
print(base_completa[colunas_para_float].dtypes)

base_completa = base_completa[base_completa['Status_entrada'] != 'Novato']

base_completa['Status_entrada'] = base_completa['Status_entrada'].replace({'Desistente': 1, 'Veterano': 0})

base_completa.drop(columns=['Pedra'], inplace = True)
base_completa.dropna(inplace=True)


#Pipeline
def pipeline(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('min_max_scaler', MinMax()),
        ('oversample', Oversample())
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

#Aplicando a pipeline
base_completa_tratada = pipeline(base_completa)

X = base_completa_tratada.drop(columns=['Status_entrada'])
y = base_completa_tratada['Status_entrada']


#Predições 
if st.button('Enviar'):
   joblib.dump(modelo_logistico, 'logistico.joblib')
 
