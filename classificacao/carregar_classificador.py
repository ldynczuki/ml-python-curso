"""
Como já temos salvo os classificadores, não precisamos dividir a nossa base
em treinamento e teste, pois os modelos já foram gerados. Com isso, podemos
até utilizar toda a base para prever utilizando os modelos já salvos
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# CARREGANDO OS MODELOS JÁ SALVOS
# Parâmetro rb para realizar a leitura: r de read e b de binary
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))

# REALIZANDO AS PREDIÇÕES APÓS TER CARREGADO OS MODELOS SALVOS
# Utilizamos o método score para retornar o score de acertos do modelo salvo
resultado_svm = svm.score(previsores, classe)
resultado_random_forest = random_forest.score(previsores, classe)
resultado_mlp = mlp.score(previsores, classe)


# REALIZANDO A PREDIÇÃO DE UM NOVO REGISTRO QUALQUER NOS MODELOS CARREGADOS
novo_registro = [[50000, 40, 5000]]
novo_registro = np.asarray(novo_registro)
# O formato está 1 linha e 3 colunas, e devemos alterar para (3, 1)
# Utilizamos parâmetro -1 que significa que não queremos mexer nas linhas
novo_registro = novo_registro.reshape(-1, 1)
# Agora devemos aplicar o escalonamento no novo registro
# E anteriormente fizemos o reshape para conseguirmos fazer o scaler, caso contrário
# todos os valores do novo_registro seriam zerados
novo_registro = scaler.fit_transform(novo_registro)
# Agora devemos voltar para o formato de (1, 3), ou seja, 1 linha e 3 colunas
# Porque já aplicamos o escalonamento
novo_registro = novo_registro.reshape(1, -1)

# O resultado retornado foi 0 - significa que o novo cliente NÃO irá pagar o empréstimo 
# Utilizamos o método predict() porque queremos prever um resultado
resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)
