# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Criação das variáveis da base, previsores e classe
base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
# Nota: Não se esqueça de utilizar .values

# Realização do pré-processamento dos dados
# Transformando atributos categóricos em numerais
# Obs: algoritmo NB não aceita atributos do tipo str
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
# NOTA: Para o algoritmo Naive-Bayes não é necessário transformar categórico
# da classe para numérico. Já para os previsores é necessário.

# Instanciando a classe GaussianNB()
classificador = GaussianNB()
# a linha abaixo irá fazer o treinamento do algoritmo, no caso NB.
classificador.fit(previsores, classe)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
# a linha abaixo irá a predição (cálculos probabilísticos do NB)
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])

# visualizando algumas propriedades do naive-bayes
# classes_ informa os valores existentes do atributo classe
print(classificador.classes_)
# class_count_ retorna a quantidade de registros existente em cada classe.
print(classificador.class_count_)
# class_pior_ retorna a probabilidade de cada classe
print(classificador.class_prior_)

# Veja que estamos passando 2 matrizes com novos registros para a predição
# Como fizemos manualmente na aula teórica, a 1ª matriz tem risco baixo
# e a segunda matriz tem risco moderado.
# Veja a variável "resultado".
