import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values  
y = base.iloc[:, 2:3].values  # utiliza-se 2:3 para retornar uma matriz

# Escalonamento dos dados
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


# divisão da base entre treinamento e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, 
                                                                  test_size=0.3,
                                                                  random_state=0)


# Criação do regressor
regressor = SVR(kernel='rbf')
regressor.fit(X_treinamento, y_treinamento)  # treinamento do modelo

score_treinamento = regressor.score(X_treinamento, y_treinamento)
score_teste = regressor.score(X_teste, y_teste)

previsoes = regressor.predict(X_teste)  # realizando previsoes com a base de teste

y_teste = scaler_y.inverse_transform(y_teste)  # realizando o escalonamento inverso, ou seja, retornando ao valor original
previsoes = scaler_y.inverse_transform(previsoes)  # realizando o escalonamento inverso, ou seja, retornando ao valor original

mae = mean_absolute_error(y_teste, previsoes)  # calculando a média dos erros das previsões
