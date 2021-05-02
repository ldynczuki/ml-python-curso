"""
Neste caso, queremos fazer a previsão dos preços (prices) das casas.
Atributo previsor (coluna 5) - "sqft_living"
Atributo classe (coluna 2) - "price"
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv("house-prices.csv")

# Utilizar o comando 5:6 faz com que o tipo de dado de X seja uma matriz
X = base.iloc[:, 5:6].values
y = base.iloc[:, 2].values

# Para utilizarmos o algoritmo de regressão linear, precisamos que o atributo
# previsor esteja em matriz. Podemos utilizar o 5:6 que ele não irá pegar a
# coluna 6 porque é indice_final-1


X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size=0.3,
                                                                  random_state=0)

# Criação do regressor
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

# Verifica o score do algoritmo de regressão, ou seja, o quanto o atributo
# previsor se adapta ao atributo classe
# Neste caso, o atributo previsor "sqft_living" não se adaptou tão bem porque
# o valor resultante foi de 0.4945
# Ou seja, o quanto o atributo escolhido consegue prever o atributo classe
score = regressor.score(X_treinamento, y_treinamento)


# Vamos gerar um gráfico para "comprovar"/verificar que o atributo previsor
# não é tão bom para prever o atributo classe
plt.scatter(X_treinamento, y_treinamento)
plt.plot(X_treinamento, regressor.predict(X_treinamento), color='Red')


# Realizando algumas previsões
previsoes = regressor.predict(X_teste)

# Analisando os resultados das previsões
resultado = abs(y_teste - previsoes)
resultado.mean()

# Podemos utilizar as funções mean_absolute_error e mean_squared_error
# ao invés de calcular o erro manualmente como fizemos acima
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)


# Gerando gráfico com a base de dados de teste
plt.scatter(X_teste, y_teste)
plt.plot(X_teste, regressor.predict(X_teste), color='Red')

# verificando o score com a base de teste
regressor.score(X_teste, y_teste)
