import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np


base = pd.read_csv('credit-card-clients.csv', header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

# primeiramente, iremos utilizar apenas 2 atributos previsores, todas as linhas e as colunas localizados nos índices da coluna 1 e 25
X = base.iloc[:, [1, 25]].values

# Escalonamento dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# GERAÇÃO DO GRÁFICO DENDROGRAMA PARA DETERMINAR A QTD DE CLUSTERS
dendrograma = dendrogram(linkage(X, method='ward'))

# hc = Hiearchical Cluster
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  # affinity é o método do cálculo da distância
previsoes = hc.fit_predict(X)  # fit_predict() é o método que faz o treinamento do modelo e já realiza a previsão, retornando o resultado para dentro da variável, não precisando realizar o treinamento separado da previsão.

plt.scatter(X[previsoes==0, 0], X[previsoes==0, 1], s=100, c='red', label='Cluster 1')  # retorna as linhas que sejam do cluster 0 = Cluster 1
plt.scatter(X[previsoes==1, 0], X[previsoes==1, 1], s=100, c='blue', label='Cluster 1')  # retorna as linhas que sejam do cluster 1 = Cluster 2
plt.scatter(X[previsoes==2, 0], X[previsoes==2, 1], s=100, c='green', label='Cluster 1')  # retorna as linhas que sejam do cluster 2 = Cluster 3
plt.xlabel('Limite do Cartão de Crédito')
plt.ylabel('Gastos do Cartão de Crédito')
plt.legend()
