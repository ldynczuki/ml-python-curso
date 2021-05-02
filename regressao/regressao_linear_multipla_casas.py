"""
A diferença entre Regressão Linear Simples x Regressão Linear Múltipla
é que na simples temos apenas 1 atributo previsor, e a múltipla é que temos
mais de 1 atributo previsor
Como vimos o cálculo da regressão linear, encontramos o b0 que é a constante
e o b1 que é o coeficiente do atributo previsor.
Como a regressão linear simples tem apenas 1 previsor, temos apenas o coeficiente
b1. Já na regressão linear múltipla, teremos um coeficiente para cada atributo previsor

Atributo classe: "price" - coluna 2
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv("house-prices.csv")

# criação do atributo previsor (X) e classe (y)
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values


# dividindo a base de treinamento e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, 
                                                                  test_size=0.3,
                                                                  random_state=0)


# Criação do regressor
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento) # treinamento do regressor (modelo)

# Verificando o score - quanto o modelo se adaptou (quanto maior melhor)
# Podemos verificar que utilizando múltiplos atributos previsores, o valor
# do score (correlação) é forte
score = regressor.score(X_treinamento, y_treinamento) 

# Realizando algumas previsões
previsoes = regressor.predict(X_teste)

# Calculando os erros das previsoes
# Passamos como parâmetro a classe e as previsoes feitas pelo regressor
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes) # o cálculo eleva ao quadrado

# Verificando o score na base de teste
regressor.score(X_teste, y_teste)


# VERIFICANDO OS PARÂMETROS
# Veja que temos apenas um valor para b0 que é a constante de todo o modelo
# e temos um coeficiente para cada atributo previsor
regressor.intercept_
regressor.coef_

len(regressor.coef_)  # verificando a quantidade de coeficiente
