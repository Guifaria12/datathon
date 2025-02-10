import pandas as pd
import streamlit as st

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
