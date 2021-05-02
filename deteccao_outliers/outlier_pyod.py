"""
Objetivo é detectar outliers utilizando a biblioteca PyOD
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN


base = pd.read_csv('credit-data.csv')

# Neste exemplo, iremos tratar os dados ausentes "nan" excluindo-os
# não é a melhor maneira, em outros exemplos substuímos pelo valor da média
base = base.dropna()

# Criação dos detectores de outliers com KNN
detector = KNN()
detector.fit(base.iloc[:, 1:4])  # não iremos passar o atributo com id do cliente para não treinar com ele

# o método .labels_ retorna uma lista com valores 0 e 1, onde 0 não é um outlier e 1 é outlier
previsoes = detector.labels_

# método decision_scores_ retorna o nível de confianca das previsoes da detecção de outliers
confianca_previsoes = detector.decision_scores_


outliers = []

# iremos fazer append na lista vazia acima inserindo os valores que foram previstos como outliers
for i in range(len(previsoes)):
    # print(previsoes[i])
    if previsoes[i] == 1:
        outliers.append(i)  # irá adicionar o id dos registros que foram detectados como outliers pelo algoritmo
        

# Criação de um novo DataFrame, onde será filtada do DataFrame original "base"
# onde as linhas serão os valores dos id que capturamos anteriormente como outliers e pegaremos todas as colunas
lista_outliers = base.iloc[outliers, :]
