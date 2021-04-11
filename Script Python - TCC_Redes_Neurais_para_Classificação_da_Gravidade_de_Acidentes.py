#!/usr/bin/env python
# coding: utf-8

# #### <center> PONTIFÍCIA UNIVERSIDADE CATÓLICA DE MINAS GERAIS </center>

# #### <center> NÚCLEO DE EDUCAÇÃO A DISTÂNCIA </center>

# #### <center> Pós-graduação Lato Sensu em Ciência de Dados e Big Data </center>

# # <center> Modelos de classificação da gravidade de acidentes em rodovias federais brasileiras através de algoritmos de redes neurais </center>

# #### <center> Ramon Batista de Araújo </center>

# #### <center> Belo Horizonte, 2021 </center>

# ### Resumo
# 
# Os acidentes de trânsito são um problema sério de saúde pública no planeta (MÁSILKOVÁ, 2017). Segundo dados da PRF (Polícia Rodoviária Federa), em 2016 acoteceram por volta de 96 mil acidentes, com 87 mil pessoas feridas e 6.398 óbitos, somente em rodovias federais brasileiras. Além disso, esses acidentes geraram mais de 12,3 bilhões de reais em custos para os cofres brasileiros. (BRASIL, 2018). De acorodo com o último relatório de década da OMS (Organização Mundial de Saúde) acidentes de trânsito é a 8ª principal causa de mortes no mundo e a principal entre pessoas de 5 a 29 anos. São 1,35 milhões de vidas perdidas por ano em acidentes de trânsito (WORLD HEALTH ORGANIZATION, 2018).
# 
# Técnicas como machine learning podem extrair conhecimento, auxiliando-os pesquisadores e gestores da área em tomadas de decisões. Os algoritmos de aprendizado de máquina de redes neurais são capazes de classificar a gravidade de um acidente de trânsito, como usado por diversos profissionais em todo mundo. Assim sendo, este estudo tem como objetivo classificar a gravidade dos acidentes de trânsito em rodovias federais brasileiras utilizando de redes neurais. Além disso, complementar os trabalhos já realizados descritos no relatório desse projeto, incluindo na análise novos atributos como a marca, idade e a potência do motor do veículo. 
# 
# Esse estudo comparou quatro modelos de redes neurais, modelo com dados desbalanceados, com dados balanceados, modelo otimizado desbalanceado e modelo otimizado balanceado, conforme procedimento abaixo.

# ## Importação das bibliotecas

# In[1]:


#Instalação das bibliotecas (se necessário)
# !pip install pandas
# !pip install numpy
# !pip install holidays
# !pip install imblearn
# !pip install seaborn
# !pip install matplotlib
# !pip install sklearn


# In[2]:


#Importação das bibliotecas e módulos

#Tratamentos dos dados
import pandas as pd
import numpy as np

#Balancemaneto
from imblearn.under_sampling import NearMiss

#Datas
from datetime import datetime

#Feriados
from pandas.tseries import holiday
import holidays

#Gráficos
import seaborn as sns
import matplotlib as mpl  
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

#Seleção de Variáveis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


#Modelo de Redes Neurais
from sklearn.neural_network import MLPClassifier

#Avaliação do Modelo
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#Otimizando Modelo
from sklearn.model_selection import GridSearchCV


# ## Importando os datasets

# ### Dados de acidentes da Polícia Rodoviária Federal 

# In[3]:


#Acidentes de 2017
df17 = pd.read_csv('acidentes2017.csv', sep=';')

df17.head()


# In[4]:


#Acidentes de 2018
df18 = pd.read_csv('acidentes2018.csv', sep=';')

df18.head()


# In[5]:


#Acidentes de 2019
df19 = pd.read_csv('acidentes2019.csv', sep=';')

df19.head()


# In[6]:


#Acidentes de 2020
df20 = pd.read_csv('acidentes2020.csv', sep=';')

df20.head()


# ### Dados das características do veículo pelo Renavam 

# In[7]:


#Características dos Veículos
dfpot = pd.read_csv('potencia.csv', sep='"",""', encoding='utf-8')

dfpot.head()


# ## Processamento dos dados de acidentes 

# ### Informações Acidentes 2017

# In[8]:


#Informações do dataset Acidentes 2017
df17.info()


# In[9]:


#Conferência de valores únicos dos Acidentes 2017
df17.nunique()


# ### Informações Acidentes 2018

# In[10]:


#Informações do dataset Acidentes 2018
df18.info()


# In[11]:


#Conferência de valores únicos dos Acidentes 2018
df18.nunique()


# ### Informações Acidentes 2019

# In[12]:


#Informações do dataset Acidentes 2019
df19.info()


# In[13]:


#Conferência de valores únicos dos Acidentes 2019
df19.nunique()


# ### Informações Acidentes 2020

# In[14]:


#Informações do dataset Acidentes 2020
df20.info()


# In[15]:


#Conferência de valores únicos dos Acidentes 2020
df20.nunique()


# ### Concatenção e tratamentos dos datasets de acidentes

# In[16]:


#Concatenando datasets de Acidentes
df = pd.concat([df17, df18, df19, df20])

df.head()


# In[17]:


#Informações do dataset de acidentes
df.info()


# In[18]:


#Remoção de valores ausentes 
df = df.dropna()

df.info()


# In[19]:


#Remoção de valores duplicados
df = df.drop_duplicates()

df.info()


# In[20]:


#Conferência de valores únicos dos Acidentes 2020
df.nunique()


# In[21]:


#Backup DatraFrame
df2 = df


# ### Explorando os atributos 

# In[22]:


#Verificando varívael id
df2['id'].value_counts()


# In[23]:


#Verificando varívael pesid
df2['pesid'].value_counts()


# In[24]:


#Verificando varívael data
df2['data_inversa'].value_counts()


# In[25]:


#Verificando varívael horário
df2['horario'].value_counts()


# In[26]:


#Verificando varívael UF
df2['uf'].value_counts()


# In[27]:


#Gráfico por Estado
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,5)) #Tamanho
sns.countplot(x='uf', #variável
              order=df2['uf'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel('UF',fontsize=15)
plt.ylabel('Quantidade de acidentes',fontsize=15)
plt.title('Quantidades de acidentes por Estado',fontsize=18)
plt.savefig('quant_acid_uf.svg', format='svg')


# In[28]:


#Verificando varívael BR
df2['br'].value_counts()


# In[29]:


#Convertendo em inteiro
df2['br'] = df2['br'].astype(int)


# In[30]:


#Gráfico por BR
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,5)) #Tamanho
sns.countplot(x='br', #variável
              order=df2['br'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel('BR',fontsize=15)
plt.ylabel('Quantidade de acidentes',fontsize=15)
plt.title('Quantidades de acidentes por BR',fontsize=18)
plt.xlim(0,15)
plt.savefig('quant_acid_br.svg', format='svg')


# In[31]:


#Verificando varívael Km
df2['km'].value_counts()


# In[32]:


#Verificando varívael Cidade
df2['municipio'].value_counts()


# In[33]:


#Gráfico por Município
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,7)) #Tamanho
sns.countplot(x='municipio', #variável
              order=df2['municipio'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel('Município',fontsize=15)
plt.ylabel('Quantidade de acidentes',fontsize=15)
plt.title('Os 15 municípios com mais acidentes',fontsize=20)
plt.xticks(fontsize=10,rotation=13)
plt.xlim(0,15)
plt.savefig('quant_acid_municipio.svg', format='svg')


# In[34]:


#Verificando varívael Causa Principal
df2['causa_principal'].value_counts()


# In[35]:


#Selecionando somente causas principais
df2 = df2.loc[df2['causa_principal'] == 'Sim']

df2.info()


# In[36]:


#Valores únicos
df2.nunique()


# In[37]:


#Verificando varívael Causa do Acidente
df2['causa_acidente'].value_counts()


# In[38]:


#Gráfico das Causas de Acidentes
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,5)) #Tamanho
sns.countplot(x='causa_acidente', #variável
              order=df2['causa_acidente'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel("Causas",fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('As 10 principais causas de acidentes',fontsize=15)
plt.xticks(fontsize=11,rotation=30)
plt.xlim(0,10)
plt.savefig('causa.svg', format='svg')


# In[39]:


#Verificando varívael Ordem do Acidente
df2['ordem_tipo_acidente'].value_counts()


# In[40]:


#Verificando varívael Tipo de acidente
df2['tipo_acidente'].value_counts()


# In[41]:


#Gráfico dos tipos de acidente
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,5)) #Tamanho
sns.countplot(x='tipo_acidente', #variável
              order=df2['tipo_acidente'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel("Tipos de Acidentes",fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('Os principais tipos de acidentes',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.xlim(0,10)
plt.savefig('tipo.pdf', format='pdf')


# In[42]:


#Verificando varívael Classificação do acidente
df2['classificacao_acidente'].value_counts()


# In[43]:


#Gráfico da Classificação do acidente
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,5)) #Tamanho
sns.countplot(x='classificacao_acidente', #variável
              order=df2['classificacao_acidente'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel("Classificação",fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('Classificação dos Acidentes',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('classificacao.svg', format='svg')


# In[44]:


#Verificando varívael fase do dia
df2['fase_dia'].value_counts()


# In[45]:


#Gráfico da Fase do Dia
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,5)) #Tamanho
sns.countplot(x='fase_dia', #variável
              order=df2['fase_dia'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel("Fase do dia",fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('Fase do dia x Quantidade',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('fase.svg', format='svg')


# In[46]:


#Verificando varívael Sentido da Via
df2['sentido_via'].value_counts()


# In[47]:


#Verificando varívael Trçado da Via
df2['tracado_via'].value_counts()


# In[48]:


#Removendo valores não informados
df2 = df2.loc[df2['tracado_via'] != 'Não Informado']

df2.info()


# In[49]:


df2.nunique()


# In[50]:


#Valores traçado da via
df2['tracado_via'].value_counts()


# In[51]:


#Gráfico do Traçado da Via
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(18,8)) #Tamanho
sns.countplot(x='tracado_via', #variável
              order=df2['tracado_via'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel("Traçado da via",fontsize=15)
plt.ylabel('Quantidade de acidentes',fontsize=15)
plt.title('Traçado da via x Quantidade de acidentes',fontsize=20)
plt.xticks(fontsize=12,rotation=13)
plt.savefig('tracado.svg', format='svg')


# In[52]:


#Verificando varívael Uso do Solo
df2['uso_solo'].value_counts()


# In[53]:


#Verificando id do veículo
df2['id_veiculo'].value_counts()


# In[54]:


#Verificando tipo de veículo
df2['tipo_veiculo'].value_counts()


# In[55]:


#Selecionando somente automóveis
df2 = df2.loc[df2['tipo_veiculo'] == 'Automóvel']

df2.info()


# In[56]:


#Valores unicos
df2.nunique()


# In[57]:


#Conferindo
df2['tipo_veiculo'].value_counts()


# In[58]:


#Verificando ano de fabricação
df2['ano_fabricacao_veiculo'].value_counts()


# In[59]:


#Verificando tipo de envolvido
df2['tipo_envolvido'].value_counts()


# In[60]:


#Selecionando somente condutor
df2 = df2.loc[df2['tipo_envolvido'] == 'Condutor']

df2.info()


# In[61]:


#Valores unicos
df2.nunique()


# In[62]:


#conferindo
df2['tipo_envolvido'].value_counts()


# In[63]:


#Verificando Estado fisíco
df2['estado_fisico'].value_counts()


# In[64]:


#Gráfico do Gravidade dos Acidentes
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,5)) #Tamanho
sns.countplot(x='estado_fisico', #variável
              order=df2['estado_fisico'].value_counts().index, #ordem descrente
              data=df2, #dataframe
              palette=cores,) #paleta de cores
plt.xlabel("Estado físico",fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('Gravidade do acidente',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('gravidade.svg', format='svg')


# In[65]:


#Verificando a idade do condutor
df2['idade'].describe()


# In[66]:


#Boxplot idade
idade = df2['idade']

cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(6,8)) #Tamanho
sns.boxplot(y=idade, palette=cores)
plt.xlabel("Boxplot",fontsize=10)
plt.ylabel('Idade',fontsize=10)
plt.title('Boxplot da Idade do Condutor',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('boxplot_idade.svg', format='svg')


# In[67]:


#Seleção de idade acima de 18 anos e abaixo de 76 anos
df2 = df2.loc[(df2['idade'] >=18) & (df2['idade'] <=76)]

df2['idade'].describe()


# In[68]:


#Boxplot idade entre 18 a 76 anos
idade2= df2['idade']

cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(6,8)) #Tamanho
sns.boxplot(y=idade2,palette=cores)
plt.xlabel("Boxplot",fontsize=10)
plt.ylabel('Idade',fontsize=10)
plt.title('Boxplot da Idade do Condutor entre 18 a 76 anos')
plt.xticks(fontsize=10,rotation=13)
plt.savefig('boxplot_idade2.svg', format='svg')


# In[69]:


#Backup de Dataset tratado
df2.info()


# In[70]:


#Valores unicos
df2.nunique()


# In[71]:


#Verificando sexo do condutor
df2['sexo'].value_counts()


# In[72]:


#Verificando nº de ilesos
df2['ilesos'].value_counts()


# In[73]:


#Verificando nº de feridos leves
df2['feridos_leves'].value_counts()


# In[74]:


#Verificando nº de feridos graves
df2['feridos_graves'].value_counts()


# In[75]:


#Verificando nº de mortos
df2['mortos'].value_counts()


# In[76]:


#Backup

df3 = df2


# ## Análise Exploratória dos dados de acidentes

# ### Ano de Fabricação

# In[77]:


#Estatísticas do ano de fabricação
df3['ano_fabricacao_veiculo'].describe()


# In[78]:


#Boxplot do ano de fabricação
ano = df3['ano_fabricacao_veiculo']

cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(6,8)) #Tamanho
sns.boxplot(y=ano, palette=cores)
plt.xlabel("Boxplot",fontsize=15)
plt.ylabel('Ano de Fabricação',fontsize=15)
plt.title('Boxplot do Ano de Fabricação',fontsize=20)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('boxplot_ano.svg', format='svg')


# In[79]:


#Seleção de veículos fabricados após 1956
df4 = df3.loc[df3['ano_fabricacao_veiculo'] > 1956]

df4['ano_fabricacao_veiculo'].describe()


# In[80]:


#Boxplot do ano de fabricação após 1956
ano = df4['ano_fabricacao_veiculo']

cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(6,8)) #Tamanho
sns.boxplot(y=ano, palette=cores)
plt.xlabel("Boxplot",fontsize=10)
plt.ylabel('Ano de Fabricação',fontsize=10)
plt.title('Boxplot do Ano de Fabricação após 1956',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('boxplot_ano2.svg', format='svg')


# In[81]:


df4.info()


# ### Definindo a idade do veículo 
# 
# Definindo a idade do veículo pela data do acidente e ano de fabricação

# #### Data do acidente

# In[82]:


#Explorando a varíavel data_inversa
df4['data_inversa'].head()


# In[83]:


#Convertendo em datetime pandas
df4['data_inversa'] = pd.to_datetime(df4['data_inversa'])

df4['data_inversa'].head()


# In[84]:


#Criano nova coluna somente com o ano do acidente
df4['data_ano']  = df4['data_inversa']


# In[85]:


#Selecionando apenas o ano
df4['data_ano'] = (df4['data_ano'].dt.year)

df4['data_ano'].head()


# #### Ano de Fabricação

# In[86]:


#Explorando a varíavel ano de fabricação
df4['ano_fabricacao_veiculo'].head()


# In[87]:


#Convertendo em números inteiros
df4['ano_fabricacao_veiculo'] = df4['ano_fabricacao_veiculo'].astype(int)
df4['ano_fabricacao_veiculo'].head()


# #### Idade do veículo

# In[88]:


#Nova coluna com a idade do veículo, subtraindo ano do acidente pelo ano de fabricação
df4['idade_veiculo'] = df4['data_ano'] - df4['ano_fabricacao_veiculo'] 
df4['idade_veiculo'].head()


# In[89]:


#Backup dataset com idade do veículo

df5 = df4

df5.head()


# ### Definindo os feriados 

# In[90]:


#Zerando os ids
df5= df5.reset_index()
tam = df5.shape[0]
df5['id'] = range(tam)
df5 = df5.drop(columns=['index'])
df5.head()


# In[91]:


#Conferindo a existência de feriados na data_inversa
df5['data_inversa'][0] in holidays.Brazil()


# In[92]:


#Criando nova coluna zerada de feriados
df5['Feriado'] = 0

df5['Feriado'].value_counts()


# In[93]:


#Lista de feriados no Brasil
feriado = holidays.Brazil()

#Lista com o tamanho do dataframe
tam_list = list(range(df5.shape[0]))

#Substituindo os valores zerados de feriados pela existencia de feriados na lista de feriados
for i in tam_list:
    df5['Feriado'][i] = df5['data_inversa'][i] in feriado
    
#Conferindo coluna Feriados
df5['Feriado'].head()


# In[94]:


#Quantidade de Feriados
df5['Feriado'].value_counts()


# In[95]:


####Renomeando atributos
#Feriado
df5['Feriado'] = df5['Feriado'].replace({True: 'Feriado'})

#Quando Dia Normal
df5['Feriado'] = df5['Feriado'].replace({False: 'Dia Normal'})

df5['Feriado'].value_counts()


# In[96]:


#Backup dataset com feriados
df6 = df5

df6.head()


# ### Preparando os dados para receber o dataset de potência 

# In[97]:


#Explorando os dados de ano
df6["ano_fabricacao_veiculo"].head()


# In[98]:


#Convertendo em string
df6['ano_fabricacao_veiculo'] = df6['ano_fabricacao_veiculo'].astype(str)
df6["ano_fabricacao_veiculo"].head()


# In[99]:


#Criando nova coluna com as colunas marca e ano de fabricação
df6["marca_ano"] = df6["marca"] + " " + df6["ano_fabricacao_veiculo"]

df6.head()


# ## Processamento do dataset das características dos veículos

# In[100]:


#Explorando o dataset
dfpot.head()


# In[101]:


#Explorando o dataset
dfpot.info()


# In[102]:


#Novo dataframe selecionando colunas que serão utilizadas
dfpot2 = dfpot.iloc[:,[1,2, 4]]

dfpot2.head()


# In[103]:


#Renomeando as colunas conforme o dataset de acidentes
dfpot2 = dfpot2.rename(columns={'Marca Modelo': 'marca',
                                'Ano Fabricação Veículo': 'ano_fabricacao_veiculo',
                                'Potência Veículo – Frota Atual': 'potencia',})


# In[104]:


#Explorando o dataset
dfpot2.info()


# In[105]:


#Removendo valores ausentes
dfpot2 = dfpot2.dropna()

dfpot2.info()


# In[106]:


#Removendo valores duplicados
dfpot2 = dfpot2.drop_duplicates()

dfpot2.info()


# In[107]:


#Estatísticas do dataset
dfpot2.describe()


# ### Explorando a Potência

# In[108]:


#Histograma da potência
cv = dfpot2['potencia']

fig = plt.figure(figsize=(8,6)) #Tamanho
plt.hist(cv, bins = 20, ec = "k", alpha = .6, color = '#df2020')
plt.xlabel("Potência (cv)",fontsize=10)
plt.title('Histograma da Potência',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('hist_pot.svg', format='svg')


# In[109]:


#Boxplot da potência
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(8,6)) #Tamanho
sns.boxplot(y=cv, palette=cores)
plt.xlabel("Potência (cv)",fontsize=10)
plt.ylabel('Boxplot',fontsize=10)
plt.title('Boxplot da Potência',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('boxplot_pot.svg', format='svg')


# In[110]:


#Selecionando potências entre 60 a 300 cv
dfpot2 = dfpot2.loc[(dfpot2['potencia'] >=60) & (dfpot2['potencia'] <=300)]

dfpot2['potencia'].describe()


# In[111]:


#Boxplot da potência
cv = dfpot2['potencia']

cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(8,6)) #Tamanho
sns.boxplot(y=cv, palette=cores)
plt.xlabel("Potência (cv)",fontsize=10)
plt.ylabel('Boxplot',fontsize=10)
plt.title('Boxplot da Potência entre 60 a 300 cv',fontsize=15)
plt.xticks(fontsize=10,rotation=13)
plt.savefig('boxplot_pot2.svg', format='svg')


# In[112]:


#Backup dataset com seleção de potência

dfpot3 = dfpot2

dfpot3.head()


# ### Preparando dados para concatenação com dataset de acidentes

# In[113]:


#Explorando os dados de ano
dfpot3["ano_fabricacao_veiculo"].head()


# In[114]:


#Convertendo em string
dfpot3['ano_fabricacao_veiculo'] = dfpot3['ano_fabricacao_veiculo'].astype(str)
dfpot3["ano_fabricacao_veiculo"].head()


# In[115]:


#Criando nova coluna com as colunas marca e ano de fabricação
dfpot3["marca_ano"] = dfpot3["marca"] + " " + dfpot3["ano_fabricacao_veiculo"]

dfpot3.head()


# In[116]:


#Novo dataset com as colunas que serão concatenadas
dfpot4 = dfpot3.iloc[:,[3,2]]

dfpot4.head()


# In[117]:


#Explorando dataset
dfpot4['marca_ano'].value_counts()


# In[118]:


#Explorando dataset
dfpot4.info()


# In[119]:


#Removendo duplicados
dfpot4 = dfpot4.drop_duplicates()

dfpot4.info()


# In[120]:


#Novo dataset agrupado pela média de potência
dfpot5 = dfpot4.groupby(['marca_ano']).mean()
dfpot5.head()


# In[121]:


#Resetando indice
dfpot5= dfpot5.reset_index()
dfpot5.head()


# ## Concatenação dos datasets de acidentes e características dos veículos 

# In[122]:


#Backups
df_aci = df6
df_pot = dfpot5


# In[123]:


#Explorando dataset acidentes
df_aci.info()


# In[124]:


#Explorando dataset potência
df_pot.info()


# In[125]:


#Concatenando datasets pela marca e ano
df = pd.merge(df_aci, df_pot, on=['marca_ano'], how='left')
df.head()


# In[126]:


#Explorando dados
df.info()


# In[127]:


#Removendo valores ausentes
df = df.dropna()

df.info()


# ## Tratamento do dataset final 

# ### Definindo os classificadores

# In[128]:


#Quantidade de valores ilesos
df.ilesos.value_counts()


# In[129]:


#Quantidade de valores de feridos leves
df.feridos_leves.value_counts()


# In[130]:


#Quantidade de valores de feridos graves
df.feridos_graves.value_counts()


# In[131]:


#Quantidade de valores de mortos
df.mortos.value_counts()


# In[132]:


#Definindo a Gravidade pela soma dos feridos graves e mortos

df['Gravidade'] = df['feridos_graves']+df['mortos']

df['Gravidade'] = df['Gravidade'].replace({1: 'Grave'})

df['Gravidade'] = df['Gravidade'].replace({0: 'Não Grave'})

df['Gravidade'].value_counts()


# In[133]:


#Gráfico
fig = plt.figure(figsize=(8,4)) #Tamanho
sns.countplot(x='Gravidade', #variável
              order=df['Gravidade'].value_counts().index, data=df)
plt.xlabel('Gravidade',fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('Gráfico da Gravidade dos acidentes',fontsize=15)
plt.savefig('gravidade2.svg', format='svg')


# In[134]:


df.head()


# ### Seleção de variáveis

# In[135]:


df.info()


# In[136]:


#Remoção de colunas desnecessárias
df2 = df.iloc[:,[ 3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 17, 18, 19, 22, 26, 27, 38, 39, 41, 42]]
df2.head()


# In[137]:


#Convertendo em variáveis categóricas
df2 = df2.astype("category")
df2.info()


# In[138]:


#Fatorizando as varíaveis 
df3 = df2.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

#Criando Gráfico de Correlação etre variáveis
cores = sns.light_palette("red",30,reverse=True) #Cor
fig = plt.figure(figsize=(15,15)) #Tamanho
df3cor = sns.heatmap(df3.corr()) 
plt.title('Gráfico de Correlação',fontsize=15)
plt.savefig('correlação.svg', format='svg')


# In[139]:


#Novo dataset com varíveis fatorizadas
dfML = df2.apply(lambda x : pd.factorize(x)[0])


# In[140]:


#Separando o dataset em input e output
X = dfML.drop(['Gravidade'], axis=1)
y = dfML['Gravidade']


# In[141]:


# Extração de Variáveis com Testes Estatísticos Univariados (Teste qui-quadrado)
test = SelectKBest(chi2, k=12)
fit = test.fit(X, y)
features = fit.transform(X)
print(features)


# In[142]:


#Sumarizando as varíveis
fit.get_support(indices=True)

cols = fit.get_support(indices=True)
dfML2 = dfML.iloc[:,cols]
dfML2.info()


# ## Machine Learning

# In[143]:


#Separando as varíveis 
X = dfML2
y = dfML['Gravidade']


# In[144]:


#Criando os conjuntos de dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[145]:


# Padronização dos dados
scaler = StandardScaler()
scaler.fit(X_train)


# In[146]:


# Aplicando a padronização aos dados
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[147]:


# Criação do modelo
mlp = MLPClassifier(hidden_layer_sizes = (10,10,10))
mlp.fit(X_train, y_train)


# In[148]:


#Fazendo as previsões e construindo a Confusion Matrix
predictions = mlp.predict(X_test)

#Confusion Matrix
print(confusion_matrix(y_test,predictions))


# In[149]:


#Acurácia
print(accuracy_score(y_test,predictions))


# In[150]:


# Imprimindo o relatório
print("Relatório de Classificação:\n", classification_report(y_test, predictions, digits=4))
print("AUC: {:.4f}\n".format(roc_auc_score(y_test, predictions)))


# ## Balanceando os dados 

# In[151]:


dfMLb = dfML


# In[152]:


sns.countplot(x='Gravidade', #variável
              order=dfMLb['Gravidade'].value_counts().index, data=dfMLb)
plt.xlabel('Gravidade',fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('Gráfico Desbalanceado',fontsize=15)
plt.savefig('gravidadedes.svg', format='svg')


# In[153]:


#Separando as varíaveis preditoras
X2 = dfMLb.drop("Gravidade", axis = 1)
y2 = dfMLb.Gravidade


# In[154]:


#Aplicando balanceamento
nr = NearMiss()
X2, y2 = nr.fit_sample(X2, y2)


# In[155]:


sns.countplot(x=y2, data=dfMLb)
plt.xlabel('Gravidade',fontsize=10)
plt.ylabel('Quantidade de acidentes',fontsize=10)
plt.title('Gráfico Balanceado',fontsize=15)
plt.savefig('gravidadebal.svg', format='svg')


# In[156]:


#Conferindo valores
y2.value_counts()


# ## Aplicando Redes Neurais no dataset balanceado

# In[157]:


#Criando os conjuntos de dados de treino e de teste
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2)


# In[158]:


# Criação do modelo
mlp = MLPClassifier(hidden_layer_sizes = (10,10,10))
mlp.fit(X2_train, y2_train)


# In[159]:


#Fazendo as previsões e construindo a Confusion Matrix
predictions = mlp.predict(X2_test)

#Confusion Matrix
print(confusion_matrix(y2_test,predictions))


# In[160]:


#Acurácia
print(accuracy_score(y2_test,predictions))


# In[161]:


# Imprimindo o relatório
print("Relatório de Classificação:\n", classification_report(y2_test, predictions, digits=4))
print("AUC: {:.4f}\n".format(roc_auc_score(y2_test, predictions)))


# ## Otimizando Modelo

# ### Dados desbalanceados

# In[162]:


# Construindo o modelo do classificador
mlp2 = MLPClassifier(max_iter=100)


# In[163]:


#Valores do parâmetros a serem testados
param_grid = {'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],}


# In[164]:


#Aplicando o GridShearch balanceados
grid_search = GridSearchCV(mlp2, param_grid = param_grid)
grid_search.fit(X, y)


# In[166]:


#Imprimindo o melhor parâmetro dados balanceados
print ('Melhores parâmetros encontrados: \ n', grid_search.best_params_)


# In[165]:


#Fazendo as previsões e construindo a Confusion Matrix dados balanceados
predictions_grid = grid_search.predict(X_test)

#Confusion Matrix dados balanceados
print(confusion_matrix(y_test,predictions_grid))


# In[167]:


#Acurácia
print(accuracy_score(y_test,predictions_grid))


# In[168]:


# Imprimindo o relatório dados balanceados
print("Relatório de Classificação:\n", classification_report(y_test, predictions_grid, digits=4))
print("AUC: {:.4f}\n".format(roc_auc_score(y_test, predictions_grid)))


# ### Dados balanceados

# In[169]:


#Aplicando o GridShearch dados balanceados
grid_search2 = GridSearchCV(mlp2, param_grid = param_grid)
grid_search2.fit(X2, y2)


# In[171]:


#Imprimindo o melhor parâmetro dados desbalanceados
print ('Melhores parâmetros encontrados: \ n', grid_search2.best_params_)


# In[170]:


#Fazendo as previsões e construindo a Confusion Matrix dados balanceados
predictions_grid2 = grid_search2.predict(X2_test)

#Confusion Matrix dados balanceados
print(confusion_matrix(y2_test,predictions_grid2))


# In[172]:


#Acurácia
print(accuracy_score(y2_test,predictions_grid2))


# In[173]:


# Imprimindo o relatório dados balanceados
print("Relatório de Classificação:\n", classification_report(y2_test, predictions_grid2, digits=4))
print("AUC: {:.4f}\n".format(roc_auc_score(y2_test, predictions_grid2)))

