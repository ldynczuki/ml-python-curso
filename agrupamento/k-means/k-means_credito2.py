import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('credit-card-clients.csv', header=1)

# Agora, iremos trabalhar com mais atributos do que o exemplo 1
# Os atributos "BILL_AMT*" são entendidos como valores gastos no cartão para cada mês
# portanto, iremos somá-los e jogando para a variável "BILL_TOTAL"
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6'] 

# Agora, iremos utilizar mais previsores do que no exemplo 1
X = base.iloc[:, [1, 2, 3, 4, 5, 25]].values

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

# Quando temos mais de 2 atributos previsores dos clusters, não temos como utilizar o matplot para gerar gráficos
# pois não tem como visualizar mais de 2 atributos (eixo x e y) 

lista_clientes = np.column_stack((base, previsoes))  # essa função serve para unir as variáveis base e previsoes e jogar na variável
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]  # comando para ordenar a nova lista unida de acordo com os valores das coluna (26) que é a coluna de previsoes (última)


"""
Portanto, quando temos mais de 2 atributos previsores não temos o recurso do gráfico dos clusters como no exemplo 1
com isso, a análise se torna muito mais manual do que visual
"""