"""
Importante!
quando utilizamos regressão com redes neurais, a função de ativação para a camada de SAÍDA DEVE SER a linear.
A função de ativação "linear" não fará nenhum cálculo com o valor, como é feito
na classificação com redes neurais podemos utilizar as funções de ativação "step function",
sigmoid, relu, softmax (quando temos mais de 2 classes) na camada de saída
Isso ocorre pelo fato de que na classificação com redes neurais, queremos classificar
já em regressão, queremos predizer valores, por isso, a função de ativação linear não realizará cálculos
Portanto, para a camada de saída na regressão com redes neurais, utilize a função de ativação "LINEAR"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1].values  # utiliza-se para retornar matriz
y = base.iloc[:, 1:2].values  # utiliza-se para retornar matriz


# É NECESSÁRIO REALIZAR O ESCALONAMENTO PARA REDES NEURAIS!!!!
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Criação da rede neural
regressor = MLPRegressor()
regressor.fit(X, y)

score = regressor.score(X, y)

plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

# Realizando previsão
previsao = scaler_y.inverse_transform(regressor.predict(scaler_X.transform(np.array(40).reshape(1, -1))))


"""
É importante lembrar que utilizamos apenas na primeira vez o "fit_transform" para que o método
adapte a sua base de dados, e nas próximas utilize apenas "transform"
se utilizar mais de uma vez o fit_transform, os valores ficarão em escalas diferentes e irá atrapalhar o resultado do algoritmo
"""
