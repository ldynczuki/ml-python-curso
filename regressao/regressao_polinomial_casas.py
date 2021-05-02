"""
É importante armazenar os resultados da regressão linear simlpes, múltipla
e polinomial
para verificar qual é o modelo mais adequado para ser utilizado
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size=0.3,
                                                                  random_state=0)

poly = PolynomialFeatures(degree=4)  # parâmetro significa que irá elevar a potência de 4
X_treinamento_poly = poly.fit_transform(X_treinamento)  # neste momento foi elevado CADA ATRIBUTO a 2ª, 3ª e 4ª potência
# agora, para cada atributo, temos 4 valores, pois elevamos a 4ª potência.
# Faça o teste com outros valores de parâmetros para verificar qual possui o melhor score e resultado do mean_absolute_error

X_teste_poly = poly.transform(X_teste)  # veja que agora so usamos o transform, e não fit_transform, quando já fizemos anteriormente o fit_transform, basta na próxima usar o transform


# Criação do Regressor
regressor = LinearRegression()  # veja que utilizamos a mesma classe da Regressão Linear para fazermos a Regressão Linear Polinomial
regressor.fit(X_treinamento_poly, y_treinamento)  # treinamento do modelo

score = regressor.score(X_treinamento_poly, y_treinamento)  # verificando o score do modelo

previsoes = regressor.predict(X_teste_poly)  # realizando previsões

mae = mean_absolute_error(y_teste, previsoes)  # calculando o erro das previsões
