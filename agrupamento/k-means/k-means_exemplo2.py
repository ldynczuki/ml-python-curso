"""
Lembre-se: os valores para a variável y no algoritmo de agrupamento (cluster)
será representado pelas classes (grupos/clusters) que os valores de x serão classificados
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# n_samples significa quantas amostras queremos gerar
# centers significa centroids, veja que serão gerados valores para y de 0 a 3
# random_state é o parâmetro que faz com que a geração dos dados seja igual
x, y = make_blobs(n_samples=200, centers=4)

# veja que para gerar o gráfico precisamos de valores para o eixo x e y
# neste caso, estamos usando todas as linhas com a coluna 0 da variável x para o eixo x
# e todas as linhas com a coluna 1 para o eixo y
plt.scatter(x[:, 0], x[:, 1])  # geração de gráfico com os registros de x nas colunas 0 e 1

kmeans = KMeans(n_clusters=4)  # criação do objeto kmeans da classe KMeans()
kmeans.fit(x)  # treinamento do modelo, passando os dados de x

previsoes = kmeans.predict(x)  # realizando a predição dos dados de x

plt.scatter(x[:, 0], x[:, 1], c = previsoes)  # c é o parâmetro de cores
