# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

base = pd.read_csv('risco-credito2.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Transformação de variáveis categóricas em numéricas
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

# Nota: para o atributo classe, normalmente o primeiro valor encontrado
# será o 0, nesta base de dados, o risco alto é o 0.

# Criando instância do classificador de Regressão Logística
classificador = LogisticRegression()
classificador.fit(previsores, classe)

# Retorna o valor do coeficiente b0
print(classificador.intercept_)

# Retorna o coeficiente de cada atributo previsor, de acordo com sua ordem
print(classificador.coef_)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
print(resultado)

# Armazena a probabilidade da predição de cada uma das classes
resultado2 = classificador.predict_proba([[0, 0, 1, 2], [3, 0, 0, 0]])
print(resultado2)

# NOTA: veja que é retornado uma matriz 2 por 2, ou seja, temos 2 novos
# registros classificados (linhas 0 e 1)
# nas colunas, temos as PROBABILIDADES de ser da classe 0 (alto) ou da classe
# 1 (baixo)
# Veja no resultado da primeira linha, temos 2 colunas
# esse registro teve 0.1856032% de probabilidade de ser risco 0 (alto)
# e 0.8143968% de probabilidade de ser risco 1 (baixo)
# Portanto, a maior probabilidade irá prevalecer, portanto, será classificado
# como risco 0 (baixo)
# O mesmo vale pro segundo registro, ou seja, a segunda linha, onde a maior
# probabilidade é de ser risco 0 (alto), que é de 0.90683435%
