#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:00:21 2019

@author: lucas
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Criação da instância da classe GaussianNB
classificador = GaussianNB()
# Criação do modelo de classificação utilizando Naive-Bayes
classificador.fit(previsores_treinamento, classe_treinamento)

# NOTA: A linha acima vai criar uma tabela de probabilidades do NB de acordo
# com os dados previsores de treinamento e indicando a qual classe eles são
# por isso, passamos como 2ª parâmetro a classe treinamento.

# Neste momento vamos chamar o método predict() que irá prever  o resultado
# os dados previsores_teste de teste de acordo com a tabela de probabilidades
# que foi criada anteriormente.
previsoes = classificador.predict(previsores_teste)

# Neste momento iremos comparar os resultados previstos de acordo com o
# algoritmo NB que está armazenado na variável "previsoes" com o resultado
# já conhecido que está na variável "classe_teste".
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# EXPLICANDO A MATRIZ DE CONFUSÃO:
# O resultado retornado da função confusion_matrix() será uma matriz, onde
# as linhas serão os valores da classe_teste, ou seja, os valores corretos
# e as colunas são os valores da variável prevista (previsoes)
# a intersecção da linha e coluna apresentará os valores onde o algortimo
# acertou, ou seja, na linha 0 e coluna 0 terá os valores onde a classe 0
# é o valor 0 e a coluna 0 os valores previstos, com isso, o valor da
# intersecção será a quantidade de acertos do algoritmo
# já a linha 0 e coluna 1 terá os valores onde era correto o valor 0
# mas o algoritmo previu que era 1, com isso, esse valor contido na célula
# são a quantidade de erros do algoritmo.
# Portanto, a matriz de confusão é utilizada para visualizar os acertos e
# os erros.
