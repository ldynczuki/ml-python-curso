#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:41:39 2019

@author: lucas
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

labelenconder_previsores = LabelEncoder()
# Aqui utilizamos direto o OneHotEnconder, mas poderíamos fazer o LabelEnconder
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                        [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

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

# ATENÇÃO: O resultado da precisão deste algoritmo foi de 0.47...
# este é um resultado ruim. O que o professor fez em seguida foi, ele
# apagou as variáveis criadas, reexecutou as etapas acima, com exceção do
# StandardScaler (escalonamento dos atributos) com isso o resultado aumentou
# para 0.79...
# Portanto, é importante reexecutar os processos retirando alguns dos pré
# processamentos, para verificar se o resultado de accurary muda ou não
# No script "naive_bayes_census.py" o resultado não mudou
# Ou seja, essa é uma área experimental, onde é testado os atributos
# e procedimentos.
# Portanto, nesta base de dados, fazer o escalonamento em atributos dummy
# acabou gerando uma baixa precisão dos da predição.