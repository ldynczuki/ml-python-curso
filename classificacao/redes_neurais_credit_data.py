import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

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


# Criando a instância do MLPClassifier()
# MLP significa => Multi Layer Perceptron
# Parâmetro "verbosa" apresenta no console os resultados dos erros aprendidos e
# as iterações (a cada iteração é esperado diminuir o parâmetro loss)
# Parâmetro "tol" é o valor de tolerância que o algoritmo tem para diminuir o
# erro a cada iteração, se passar 10 iterações e não diminuir o valor de "tol"
# o treinamento irá parar mesmo se o valor de max_iter não for alcançado
# O parâmetro hidden_layer_sizes é a quantidade de neurônios nas camadas
# ocultas, se colocarmos (100, 100) significa que teremos 2 camadas ocultas
classificador = MLPClassifier(verbose=True, max_iter=1000,
                              tol=0.000010,
                              solver='adam',
                              hidden_layer_sizes=(100),
                              activation='relu')

# NOTA: Quando executado a primeira vez o treinamento da rede neural, sem
# inserir nenhuma parametrização na linha em que instanciamos o objeto
# MLPClassifier() retornou a mensagem dizendo que o limite máximo de iterações
# foi alcançado e o seu valor otimizado não foi alcançado. Com isso, iremos
# inserir alguns parâmetros na instanciação do objeto classificador e
# executar novamente o  treinamento.
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
