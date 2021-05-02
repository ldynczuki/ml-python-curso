# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

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


# NOTA: Iremos testar vários tipos de Kernel diferentes para ver os resultados.

# Criando a instância do classificador SVC

# Primeiramente iremos realizar o teste com o kernel 'linear'
# classificador = SVC(kernel='linear', random_state=1)

# Aqui estamos testando o kernel polinomial
# classificador = SVC(kernel='poly', random_state=1)

# Testando o kernel sigmoid
# classificador = SVC(kernel='sigmoid', random_state=1)

# Testando o kernel rbf
# Inserindo o valor 2.0 para o parâmetro de custo 'c'
# lembrando que quanto maior o seu valor, mais lento e melhores resultados.
classificador = SVC(kernel='rbf', random_state=1, C=2.0)

# Realizando o treinamento do modelo SVC para geração da margem máxima
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# O kernel 'rbf' retornou um resultado melhor do que os outros para essa base
