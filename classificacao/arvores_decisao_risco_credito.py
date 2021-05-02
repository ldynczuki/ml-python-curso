# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export

base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# É preciso fazer a transformação de atributos categóricos em numéricos
# caso contrário não é possível executar uma árvore de decisão
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

# Criando instância da classe da árvore de decisão
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)

# Exibe a importância de cada atributo
# A ordem dos valores é de acordo com a ordem dos atributos da base de dados.
print(classificador.feature_importances_)

# Comando utilizado para gerar um arquivo para visualização da árvore de 
# decisão criada anteriormente
export.export_graphviz(classificador,
                       out_file='arvore.dot',
                       feature_names=['historia', 'divida', 'garantias', 'renda'],
                       class_names=['alto', 'moderado', 'baixo'],
                       filled=True,
                       leaves_parallel=True)


# Nota: é preciso ter instalado o graphviz instalado para abertura do doc
# https://www.katrinasiegfried.com/single-post/2018/10/24/Installing-GraphViz-on-Linux-Ubuntu-1804LTS

# Abaixo iremos submeter dois novos registros para nossa árvore para ela
# predizer o resultado de acordo com a árvore de decisão criada com a nossa base de treinamento.

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
print(resultado)


print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
