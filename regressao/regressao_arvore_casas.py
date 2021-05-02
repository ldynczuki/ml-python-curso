import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


base = pd.read_csv("house-prices.csv")

# criação do atributo previsor (X) e classe (y)
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values


# dividindo a base de treinamento e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, 
                                                                  test_size=0.3,
                                                                  random_state=0)

# Criação do regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_treinamento, y_treinamento)  # treinamento do modelo
score = regressor.score(X_treinamento, y_treinamento)  # verificando o score do modelo na base de treinamento

previsoes = regressor.predict(X_teste)  # realizando previsões do modelo

mae = mean_absolute_error(y_teste, previsoes)  # verificando a média dos erros

regressor.score(X_teste, y_teste)  # verificando o score na base de teste
