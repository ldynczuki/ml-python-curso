import pandas as pd
import numpy as np
import keras
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Nota: Rede neural densa é quando um neurônio se conecta com todos outros
# neurônios da próxima camada (camada a frente)
# Por isso importamos do módulo keras.layers o pacote 'Dense"

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


# Criando a instância do classificador Sequential() - rede neural
classificador = Sequential()

# Configurando a camada oculta da rede neural manualmente
# (camadas ocultas, camada de saída). Cada vez que o parâmetro .add for chamado
# será criado apenas 1 camada oculta por vez. Veja que chamamos objeto Dense
# que significa que os neurônios se conectarão com todos da próxima camada
# Parâmetro "units" => quantos neurônios existem na camada oculta
# Uma regra simples para quantidade de neurônios da camada oculta é somar a
# quantidade de atributos e a quantidade de classes e dividir por 2.
# que no nosso caso são 3 atributos previsores e 1 classe e dividir por 2 é 2
# Parãmetro input_dim significa quantos atributos de entrada serão ligados a
# camada oculta. Devemos configurar isso na primeira camada oculta, pois será
# conectada a camada de entrada
classificador.add(Dense(units=2, activation='relu', input_dim=3))

# Configurando outra camada oculta da rede nerual
# lembre-se que cada comando .add será criado e configurado 1 camada oculta
classificador.add(Dense(units=2, activation='relu'))

# Configurando a camada de saída
# O parâmetro units terá valor 1 porque neste exemplo temos um problema binário
# ou seja,os possíveis resultados são apenas 2
# para a função de ativação, na camada de saída iremos utilizar a sigmoid
classificador.add(Dense(units=1, activation='sigmoid'))

# Compilando a rede neural
# Parãmetro optimizer define como fará o ajuste dos pesos
# Parâmetro "loss" é o cálculo do erro que vimos nas aulas teóricas
classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

# Realizando o treinamento da rede neural Sequential do keras
# Parâmetro batch_size indica de quantos em quantos registros é feita a atualização
# dos pesos, no noss exemplo, a cada 10 registros cálculados os erros, será atualizado os pesos
# epochs é a quantidade de vezes que será realizado o ajuste dos pesos
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10,
                  epochs=100)

# Realizando as previsões na rede neural já treinada
previsoes = classificador.predict(previsores_teste)
# O resultado das previsões será um pouco diferente, pois os valores finais
# são resultados da função sigmoid, por isso perceba que não são valores exatos 0 ou 1
# mas valores entre 0 e 1
# Por isso é necessário fazer outra codificação abaixo
# Abaixo iremos receber na variável previsões o valor True se o valor resultante da sigmoid
# for maior que 0.5 ou False se for menor
# Lembrando que este valor 0.5 é o threshold, ou seja, o limiar/limite
previsoes = (previsoes > 0.5)

# Perceba que podemos realizar a acurácia e a matriz com valores booleanos como o previsoes
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
