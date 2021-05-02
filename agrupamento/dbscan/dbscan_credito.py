"""
As previsões com valores -1 são os ruídos de previsões
portanto, altere os parâmetros no objeto e refaça o treinamento e previsões
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np


base = pd.read_csv('credit-card-clients.csv', header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

# iremos utilizar 2 atributos previsores neste exemplo
X = base.iloc[:,[1, 25]].values

# Escalonamento dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Criação do objeto dbscan da classe DBSCAN
dbscan = DBSCAN(eps=0.37, min_samples=4)
previsoes = dbscan.fit_predict(X)  # fit_predict() realiza o treinamento do modelo em conjunto com as previsoes, retornando os valores de previsão

unicos, quantidade = np.unique(previsoes, return_counts = True)  # variveis unicos recebe um array com a quantidade de grupos criados e quantidade recebe a quantidade de eleemntos em cada grupo


# X[previsoes==0, 0] significa que será retornado na posição das linhas os resultados das previsões igual a 0 e assim sucessivamente.
plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'orange', label = 'Cluster 2')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('Limite do Cartão de Crédito')
plt.ylabel('Gastos do Cartão de Crédito')
plt.legend()


lista_clientes = np.column_stack((base, previsoes))  # np.column_stack() faz a união das variáveis, ou seja, retornará a união da base e previsoes
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]  # retorna o ordenamento de todas as linhas da coluna 26
