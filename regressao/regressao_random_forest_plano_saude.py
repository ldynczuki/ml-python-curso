import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

regressor = RandomForestRegressor(n_estimators=10)  # criação do regressor
regressor.fit(X, y)  # treinamento do regressor de árvore de decisão
score = regressor.score(X, y)  # verificando o score do modelo


# criação de um array numpy que irá iniciar com menor valor de X e irá até o maior de X com incremento de 0.1 em 0.1
X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1, 1)  # incluindo uma coluna para utilizar no comando plt.plot()
plt.scatter(X, y)  # geração do gráfico com a disposição dos dados
plt.plot(X_teste, regressor.predict(X_teste), color='red')  # realizando a previsão no gráfico
plt.title('Regressão com Random Forest')
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao = regressor.predict(np.array(40).reshape(1, -1))
