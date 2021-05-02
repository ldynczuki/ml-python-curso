import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('plano-saude2.csv')

# realizando a separação entre atributos previsores e classe
X = base.iloc[:, 0:1].values  # é utilizado 0:1 para gerar matriz
y = base.iloc[:, 1:2].values  # é utilizado 0:1 para gerar matriz

"""
Iremos realizar 3 testes no SVR, utilizando kerneis diferentes:
1 - Kernel Linear
2 - Kernel Polinomial
3 - Kernel rbf

Obs: para o kernel rbf é preciso fazer o escalonamento!
"""

# KERNEL LINEAR
regressor_linear = SVR(kernel='linear')
regressor_linear.fit(X, y.ravel())

plt.scatter(X, y)  # gerando gráfico com os dados da base
plt.plot(X, regressor_linear.predict(X), color='red')  # gerando predição e plotando a linha da regressão

score_linear = regressor_linear.score(X, y)  # verificando o score do modelo SVR com kernel linear


# KERNEL POLY
#teste valores diferentes no parâmetro degree
regressor_poly = SVR(kernel='poly', degree=2, gamma='auto')  # lembrando os conceitos de regressão polinomial, utilizamos o parâmetro 2 para ter os valores originais e elevados ao quadrado (2)
regressor_poly.fit(X, y.ravel())

plt.scatter(X, y)  # gerando gráfico com os dados da base
plt.plot(X, regressor_poly.predict(X), color='red')  # gerando predição e plotando a linha da regressão

score_poly = regressor_poly.score(X, y)  # verificando o score do modelo SVR com kernel polinomial


# KERNEL rbf

# para o kernel rbf iremos realizar um pré-processamento nos dados
# Escalonamento dos dados
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


regressor_rbf = SVR(kernel='rbf', gamma='auto')  # criação do regressor
regressor_rbf.fit(X, y.ravel())  # treinamento do modelo


plt.scatter(X, y)  # gerando gráfico com os dados da base
plt.plot(X, regressor_rbf.predict(X), color='red')  # gerando predição e plotando a linha da regressão

score_rbf = regressor_rbf.score(X, y)  # verificando o score do modelo SVR com kernel rbf


"""
Realizando previsões dos 3 kerneis
"""
# Como os modelos foram treinados com valores escalonados, é preciso prever com valores escalonados
previsao1 = scaler_y.inverse_transform(regressor_linear.predict(scaler_X.transform(np.array(40).reshape(1, -1))))
previsao2 = scaler_y.inverse_transform(regressor_poly.predict(scaler_X.transform(np.array(40).reshape(1, -1))))
previsao3 = scaler_y.inverse_transform(regressor_rbf.predict(scaler_X.transform(np.array(40).reshape(1, -1))))
