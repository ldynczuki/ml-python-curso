import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


base = pd.read_csv('plano-saude2.csv')

# Utilizamos a codificação 0:1 para que X receba em matriz que precisamos
X = base.iloc[:, 0:1].values  # lembrando que pegará apenas a coluna 0
y = base.iloc[:, 1].values


# Regressão Linear Simples
regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)  # verificando o score do modelo

regressor1.predict(np.array(40).reshape(1, -1))  # Previsão de custo de uma pessoa de 40 anos

plt.scatter(X, y)  # visualização do gráfico dos dados da base
plt.plot(X, regressor1.predict(X), color='Red')  # plotando a linha da regressão linear simples
plt.title('Regressão Linear Simples')  # Adicionando título ao gráfico
plt.xlabel('Idade')  # Adicionando label ao eixo X do gráfico
plt.ylabel('Custo')  # Adicionando label ao eixo y do gráfico

"""
Ananlisando o gráfico gerado, percebe-se que a Regressão Linear Simples
não é uma boa escolha para a predição nesta base de dados.
"""


# Regressão Linear Polinominal
"""
A regressão linear polinominal é utilizada para problemas NÃO LINEARMENTE SEPARÁVEIS
obs: mesmo que o nome seja "linear", a regressão polinominal é utilizada para problemas
não linearmente separáveis
"""
poly = PolynomialFeatures(degree=2)  # este parâmetro significa que será elevado ao quadrado, podemos utilizar outros valores de parâmetros, teste para verificar qual gera o melhor resultado de score
X_poly = poly.fit_transform(X)
# Veja que será gerado uma nova variável com 3 colunas, a primeira é a constante
# a segunda coluna é o valor original de X e a terceira coluna é a multiplicação ao quadrado do valor original.

regressor2 = LinearRegression()  # perceba que utilizamos a mesma classe da LinearRegression()
regressor2.fit(X_poly, y)  # veja que o que difere é que passamos a variável X_poly ao invés de X durante o treinamento do modelo

# para criar um modelo de regressão linear polinominal, utilizamos a classe LinearRegression,
# entretanto, devemos anteriormente calcular nosso atributo previsor de forma polinominal

score2 = regressor2.score(X_poly, y)  # verificando o score do modelo


# regressor2.predict(np.array(40).reshape(1, -1))  # OBS: Perceba que irá gerar erro, pois o modelo está esperando o valor de 40 (idade) e o valor de 40 ao quadrado que foi como foi feito o treinamento


regressor2.predict(poly.transform(np.array(40).reshape(1, -1)))  # corrigindo o erro acima

plt.scatter(X, y)  # visualização do gráfico dos dados da base
plt.plot(X, regressor2.predict(poly.fit_transform(X)), color='Red')  # plotando a linha da regressão linear polinominal
plt.title('Regressão Linear Polinominal')  # Adicionando título ao gráfico
plt.xlabel('Idade')  # Adicionando label ao eixo X do gráfico
plt.ylabel('Custo')  # Adicionando label ao eixo y do gráfico
