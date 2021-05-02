"""
Lembre-se: os valores da variável x terão 2 colunas, pois são os valores
do eixo X (coluna 0) e do eixo Y (coluna 1)
e os valores da variável y são os clusters existentes

NESTE SCRIPT IREMOS COMPARAR OS ALGORITMOS ABAIXO
KMEANS X HIERARQUICO X DBSCAN
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets
import numpy as np


# É chamado de "make_moons" porque o gráfico gerado será como luas
x, y = datasets.make_moons(n_samples=1500, noise=0.09)  # esta função irá gerar valores aleatórios

plt.scatter(x[:, 0], x[:, 1], s=5)

# array de cores
cores = np.array(['red', 'blue'])

# KMEANS
kmeans = KMeans(n_clusters=2)
previsoes_kmeans = kmeans.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[previsoes_kmeans])


# HIEARCHICAL CLUSTER
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
previoes_hc = hc.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[previoes_hc])


# DBSCAN
dbscan = DBSCAN(eps=0.1)
previsoes_dbscan = dbscan.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[previsoes_dbscan])


"""
OBSERVAÇÃO: O DBSCAN SE ADAPTA MELHOR COM ESTE TIPO DE BASE DE DADOS
"""
