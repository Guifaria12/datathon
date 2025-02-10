import pandas as pd
import streamlit as st
import numpy as np

st.subheader ("""
**Faculdade de Informática e Administração Paulista – FIAP**

**Grupo 47: Cristiano de Araujo Santos Filho, Eduardo Vilela Silva, Guilherme de Faria
Rodrigues, Marcelo Pereira Varesi e Vitória Pinheiro de Mattos**
""")
st.header("**Introdução**")

st.write("""
A DEFINIR
""")

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
