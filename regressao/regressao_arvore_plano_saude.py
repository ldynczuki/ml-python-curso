import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

regressor = DecisionTreeRegressor()
regressor.fit(X, y)  # treinamento do regressor de árvore de decisão
score = regressor.score(X, y)  # verificando o score do modelo


plt.scatter(X, y)  # geração do gráfico com a disposição dos dados
plt.plot(X, regressor.predict(X), color='red')  # realizando a previsão no gráfico
plt.title('Regressão com árvore de decisão')
plt.xlabel('Idade')
plt.ylabel('Custo')


"""
A maneira como é criada o gráfico acima, não é exatamente como é feito por trás dos "panos" com
regressão em árvore de decisão, pois não temos uma linha contínua como foi exibida anteriormente
Abaixo, iremos criar uma variável nova denominada "X_teste" e iremos utilizar ela para criar
o novo gráfico com a visualização correta de uma árvore de decisão.
Árvores de decisão são chamadas de modelos não lineares e não contínuos,
por isso é verificado um gráfico com "escadas" com os pontos ligados
"""
# criação de um array numpy que irá iniciar com menor valor de X e irá até o maior de X com incremento de 0.1 em 0.1
X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1, 1)  # incluindo uma coluna para utilizar no comando plt.plot()
plt.scatter(X, y)  # geração do gráfico com a disposição dos dados
plt.plot(X_teste, regressor.predict(X_teste), color='red')  # realizando a previsão no gráfico
plt.title('Regressão com árvore de decisão')
plt.xlabel('Idade')
plt.ylabel('Custo')


regressor.predict(np.array(40).reshape(1, -1))
