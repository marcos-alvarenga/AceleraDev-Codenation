#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[4]:


black_friday.info()


# In[5]:


black_friday.nunique()


# In[6]:


#Quantidade NA's
df_info = pd.DataFrame ({'type': black_friday.dtypes,
                        'NA': black_friday.isna().sum(),
                        'NA%': (black_friday.isna().sum() / black_friday.shape[0]) * 100})


# In[7]:


df_info


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[8]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[9]:


#Quantidade Homens e Mulheres
black_friday.Gender.value_counts()


# In[10]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday[ (black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35') ].shape[0]
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[11]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[12]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[13]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return (black_friday.shape[0] - black_friday.dropna().shape[0])/(black_friday.shape[0])
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[14]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return black_friday.isna().sum().max()
q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[15]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return int(black_friday['Product_Category_3'].dropna().mode())
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[16]:


def q8():
    # Retorne aqui o resultado da questão 8.
    normalization = ((black_friday['Purchase']) - black_friday['Purchase'].min())/((black_friday['Purchase'].max()) - (black_friday['Purchase'].min()))
    return normalization.mean()
q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[17]:


def q9():
    # Retorne aqui o resultado da questão 9.
    padronization = ((black_friday['Purchase'] - black_friday['Purchase'].mean())/(black_friday['Purchase'].std()))
    return int(padronization.between(-1, 1).sum())
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[18]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return black_friday['Product_Category_2'].isna().equals(black_friday['Product_Category_2'].isna())
q10()

