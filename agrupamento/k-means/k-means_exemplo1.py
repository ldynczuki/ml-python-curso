import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Criação de variáveis para a simulação do k-means
x = [20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  # idade
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]  # salário

plt.scatter(x, y)  # geração de um gráfico com os dados existentes (x, y)

# Criação da variável base que será do tipo array numpy que é formato de matriz
base = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])


# O algoritmo K-means necessita que os dados sejam escolanados
scaler = StandardScaler()
base = scaler.fit_transform(base)

# criando objeto da classe KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(base)  # treinamento do algoritmo k-means

# armazenando os valores (x, y) dos centroides (3) gerados pelo algoritmo
centroides = kmeans.cluster_centers_

# armazenando os nomes dados para os labels
# Veja que será retornando uma quantidade igual dos registros, ou seja,
# cada rótulo (no mesmo índice) indicará a qual classe (cluster) determinado
# registro pertence
rotulos = kmeans.labels_


# criamos uma lista com 3 cores para os 3 clusters existentes
cores = ['g.', 'r.', 'b.']


for i in range(len(x)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize=15)  # plotando gráfico com os registros do array

plt.scatter(centroides[:, 0], centroides[:, 1], marker='x')  # geração gráfico com os valores dos centroides
    