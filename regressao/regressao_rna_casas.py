import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values  # utiliza-se 2:3 para gerar matriz para poder fazer o escalonamento

# É NECESSÁRIO REALIZAR O ESCALONAMENTO DOS DADOS PARA REDES NEURAIS
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Divisão da base de dados em treinamento e testes
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

# Criação da rede neural
regressor = MLPRegressor(hidden_layer_sizes=(9,9)) # inserimos o parâmetro para criar 2 camadas ocultas, cada uma com 9 neurônios
# O cálculo para o valor de camadas e neurônios já aprendemos anteriormente, que são a quantidade de atributos (16) + quantidade de saídas (1) como será um número é 1
# e então dividimos por 2 que dá 8,5 e arredondamos para cima, com isso criamos 2 camadas ocultas com 9 neurônios cada
regressor.fit(X_treinamento, y_treinamento)

score_treinamento = regressor.score(X_treinamento, y_treinamento)
score_teste = regressor.score(X_teste, y_teste)


previsoes = regressor.predict(X_teste)  # Realizando as previsões

y_teste = scaler_y.inverse_transform(y_teste)  # realizando o inverso do escalonamento, retornando o valor original
previsoes = scaler_y.inverse_transform(previsoes)  # realizando o inverso do escalonamento, retornando o valor original


mae = mean_absolute_error(y_teste, previsoes)
