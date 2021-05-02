#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:07:56 2019

@author: lucas
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

labelenconder_previsores = LabelEncoder()
# Realiza o LabelEncoder
# previsores[:, 1] = labelenconder_previsores.fit_transform(previsores[:, 1])
# previsores[:, 3] = labelenconder_previsores.fit_transform(previsores[:, 3])
# previsores[:, 5] = labelenconder_previsores.fit_transform(previsores[:, 5])
# previsores[:, 6] = labelenconder_previsores.fit_transform(previsores[:, 6])
# previsores[:, 7] = labelenconder_previsores.fit_transform(previsores[:, 7])
# previsores[:, 8] = labelenconder_previsores.fit_transform(previsores[:, 8])
# previsores[:, 9] = labelenconder_previsores.fit_transform(previsores[:, 9])
# previsores[:, 13] = labelenconder_previsores.fit_transform(previsores[:, 13])

# LabelEncoder em conjunto com o OneHotEncoder
# OneHotEnconder é utilizado para variáveis categóricas nominais
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                        [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# LabelEncoder para variáveis categóricas
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)


# CÓDIGO ABAIXO É UM TEMPLATE PARA SER REUTILIZADO FUTURAMENTE
# importação da biblioteca
# Criando a instância do classificador()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)