import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


x = [20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  # variável IDADE
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]  # variável SALÁRIO

plt.scatter(x, y)

# criação da variável base com os dados de x,y no formato de matriz (array numpy)
# cada registro (intersecção x, y) é como uma pessoa, que possui X idade e Y salário
base = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])


# Nos algoritmos que calculam a distância de pontos, é importante o escalonamento
scaler = StandardScaler()
base = scaler.fit_transform(base)

# criação do objeto da classe DBSCAN
# os valores dos parâmetros abaixo deverão ser testados e modificados até encontrar os clusters para cada registro, se retornar -1, altere o valor até encontrar todos
dbscan = DBSCAN(eps=0.95, min_samples=2)  # eps: distância máxima (threshold) entre dois pontos para ser agrupado em uma mesma classe, min_samples: é a quantidade mínima de vizinhos parar formar um cluster (denso)
dbscan.fit(base)  # treinamento do modelo

previsoes = dbscan.labels_  # se retornar valores -1 significa que o modelo não conseguiu agrupar em nenhum cluster

# criação da lista de cores com os códigos das cores
# como vimos, foram criados 3 clusters, sabendo disso, criamos a lista de 3 cores
cores = ['g.', 'r.', 'b.']

for i in range(len(base)):
    plt.plot(base[i][0], base[i][1], cores[previsoes[i]], markersize=15)
