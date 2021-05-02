"""
Para o algoritmo de agrupamento hierárquico, será gerado um gráfico dendrograma
onde serão apresentados gráficos com os registros e as distâncias euclidianas
devemos traçar uma linha horizontal no sentido em que tenha menos linhas parelelas próximas
ou seja, neste exemplo, iremos traçar a linha entre as azuis
a quantidade de clusters será a quantidade de intersecções da linha horizontal
cruzando com as linhas que teve a interseccção (das linhas azuis)
neste nosso exemplo, teremos 3 clusters
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


x = [20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  # variável IDADE
y = [1000, 1200, 2900,1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]  # variável SALÁRIO

plt.scatter(x, y)

# criando variável com formato matriz (numpy array)
base = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])


# Algoritmos de agrupamento geralmente necessitam Escalonar os dados
scaler = StandardScaler()
base = scaler.fit_transform(base)

# GERAÇÃO DO GRÁFICO DENDROGRAMA PARA A ESCOLHA DA QTD DE CLUSTERS
dendograma = dendrogram(linkage(base, method='ward'))  # plotagem do gráfico dendrograma utilizando o método Linkage para linkar os registros nos clusters
plt.title('Dendrograma')
plt.xlabel('Pessoas')
plt.ylabel('Distância Euclidiana')

# criação do objeto hc (hierarch cluster)
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  # affinity é o método para calcular a distância
previsoes = hc.fit_predict(base)  # estamos utilizando o método fit de treinar o modelo junto com predict que já irá prever, ou seja, estamos treinando e ao mesmo tempo prevendo

# o comando: base[previsoes==0,0] irá retornar linhas no eixo X os valores do cluster 0, ou seja, o 1 para a coluna 0
# o comando: base[previsoes==0,1] irá retornar linhas no eixo Y os valores do cluster 0, ou seja, o 1 para a coluna 1
plt.scatter(base[previsoes==0, 0], base[previsoes==0, 1], s=100, c='red', label='Cluster 1')  # s é o tamanho das bolunhas e c é a cor

# base[previsoes==1,0] significa que iremos retornar todas as linhas previstas para o cluster 2 ou seja, o 1 do valor
plt.scatter(base[previsoes==1, 0], base[previsoes==1, 1], s=100, c='blue', label='Cluster 2')  # s é o tamanho das bolunhas e c é a cor

# base[previsoes==1,0] significa que iremos retornar todas as linhas previstas para o cluster 2 ou seja, o 1 do valor
plt.scatter(base[previsoes==2, 0], base[previsoes==2, 1], s=100, c='green', label='Cluster 3')  # s é o tamanho das bolunhas e c é a cor

plt.xlabel('Idade')
plt.ylabel('Salário')
plt.legend()
