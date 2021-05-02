import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense


base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

labelenconder_previsores = LabelEncoder()
# Realiza o LabelEncoder
# previsores[:, 1] = labelenconder_previsores.fit_transform(previsores[:, 1])
# previsores[:, 3] = labelenconder_previsores.fit_transform(previsores[:, 3])
# previsores[:, 5] = labelenconder_previsores.fit_transform(previsores[:, 5])
# previsores[:, 6] = labelenconder_previsores.fit_transform(previsores[:, 6])
# previsores[:, 7] = labelenconder_previsores.fit_transform(previsores[:, 7])
# previsores[:, 8] = labelenconder_previsores.fit_transform(previsores[:, 8])
# previsores[:, 9] = labelenconder_previsores.fit_transform(previsores[:, 9])
# previsores[:, 13] = labelenconder_previsores.fit_transform(previsores[:, 13])


# LabelEncoder em conjunto com o OneHotEncoder
# OneHotEnconder é utilizado para variáveis categóricas nominais
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                        [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# LabelEncoder para variáveis categóricas
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)


# Criando a instância do classificador Sequential() - rede neural
classificador = Sequential()

# CONFIGURAÇÃO DAS CAMADAS OCULTAS E CAMADA DE SAÍDA
classificador.add(Dense(units=55, activation='relu', input_dim=108)) # 1ª camada oculta
classificador.add(Dense(units=55, activation='relu')) # 2ª camada oculta
classificador.add(Dense(units=1, activation='sigmoid')) # camada de saída
# lembrando que se tivéssemos mais que 2 possíveis classes, deveríamos colocar
# a quantidade de possibilidades no units e activation seria "softmax"


# Compilando a rede neural
classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

# Realizando treinamento da rede neural
# Parâmetro batch_size indica de quantos em quantos registros é feita a atualização
# dos pesos, no noss exemplo, a cada 10 registros cálculados os erros, será atualizado os pesos
# epochs é a quantidade de vezes que será realizado o ajuste dos pesos
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10,
                  epochs=100)

# Realizando previsoes com conjunto de testes na rede já treinada
previsoes = classificador.predict(previsores_teste)

previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
