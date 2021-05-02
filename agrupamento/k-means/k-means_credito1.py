import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('credit-card-clients.csv', header=1)


# Primeiramente, iremos realizar o agrupamento utilizando apenas 2 atributos
# Os atributos "BILL_AMT*" são entendidos como valores gastos no cartão para cada mês
# portanto, iremos somá-los e jogando para a variável "BILL_TOTAL"
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6'] 

# Como disse, iremos utilizar 2 atributos previsores, que serão "LIMIT_BAL" e "BILL_TOTAL"
X = base.iloc[:, [1, 25]].values

# Aplicando o escalonamento nos dados de previsão
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Cálculo > within-cluster sum of square
wcss = []

"""
CÁLCULO PARA A DEFINÇÃO DO NÚMERO DE CLUSTERS
"""
# ELBOW METHOD
# iremos iterar 10x e iremos visualizar o gráfico escolher o valor entre 1 e 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)  # o número de cluster irá alterar conforme a iteração for realizada
    kmeans.fit(X)  # treinamento do modelo
    wcss.append(kmeans.inertia_)  # para cada iteração, será armazenada na lista wcss os valores do cálculo de distância feito pelo kmeans e iremos escolher o menor valor para numero de cluster

# os valores armazenados na lista wcss serão os valores resultantes do cálculo da distância 
# devemos escolher a quantidade de clusters de acordo com o menor valor
# obs: index 0 da lista é referente a 1 cluster e possui valor de 60000.0 e assim sucesivamente.

plt.plot(range(1, 11), wcss)  # será gerado gráfico com a quantidade de clusters e seus respectivos valores wcss no eixo y
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')

"""
REALIZAÇÃO DO AGRUPAMENTO - ALGORITMO KMEANS
"""
kmeans = KMeans(n_clusters=4, random_state=0)
previsoes = kmeans.fit_predict(X)  # o método fit_predict() faz o treinamento e já faz a previsão de qual grupo cada registro de X pertence

# Lembrando que o índice começa do 0 mas é referente ao Cluster 1
# X[previsoes==0, 0] significa que queremos as linhas que são do cluster 1 que foi previsto anteriormente e a coluna 0 pois será o eixo x
# X[previsoes==0, 1] significa que queremos as linhas que são do cluster 1 que foi previsto anteriormente e a coluna 1 pois será o eixo y
plt.scatter(X[previsoes==0, 0], X[previsoes==0, 1], s=100, c='red', label='Cluster 1')  # s é o tamanho das bolinhas e c a cor
# agora queremos os elementos do cluster 2
plt.scatter(X[previsoes==1, 0], X[previsoes==1, 1], s=100, c='orange', label='Cluster 2') 
# agora queremos os elementos do cluster 3
plt.scatter(X[previsoes==2, 0], X[previsoes==2, 1], s=100, c='green', label='Cluster 3') 
# agora queremos os elementos do cluster 4
plt.scatter(X[previsoes==3, 0], X[previsoes==3, 1], s=100, c='blue', label='Cluster 4')  

plt.xlabel('Limite do Cartão de Crédito')
plt.ylabel('Valores Gastos do Cartão de Crédito')
plt.legend()  # insere a legenda no gráfico gerado (Cluster 1, 2, 3 e 4)

lista_clientes = np.column_stack((base, previsoes))  # essa função serve para unir as variáveis base e previsoes e jogar na variável
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]  # comando para ordenar a nova lista unida de acordo com os valores das coluna (26) que é a coluna de previsoes (última)
