#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:48:40 2019

@author: lucas
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
# Nota: vai dar um erro caso queira abrir a variável no explorador
# isso ocorre porque é uma limitação da versão do spyder para tipo "object"


# Transformação de variáveis categóricas em variáveis numéricas (discretas)
# Nota: veja que será transformado apenas os atributos categóricos
# Transformação de variáveis categórias nominais em variáveis numéricas
# Perceba que é instanciado o objeto "ColumnTransformer" para realizar a
# transformação colocando valores 0 e 1 ao invés de contagens, pois variáveis
# categóricas nominais não tem ordem, portanto, não pode haver ordem de
# grandeza nos valores numéricos.
# Isso não ocorre para as variáveis categóricas ordinais, que existe ordens
labelenconder_previsores = LabelEncoder()
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                        [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# Transformação de variáveis categórias em variáveis numéricas da "classe"
# Perceba queo o tipo da variável "classe" é categórica ordinal, pois existe
# uma ordem, sendo que quem ganha <=50k está em uma ordem abaixo de >50k.
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

# Escalonamento dos atributos previsores após a transformação das variáveis.
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# VARIÁVEIS DO TIPO DUMMY - Transformação de variáveis categórias NOMINAIS.
# módulo: OneHotEncoder
# Correção da transformação de variáveis categórias nominais.
# Nota: Pelo fato das variáveis categórias nominais não terem ordem, pode
# acontecer na hora de calcular o algoritmo, ser contabilizado de forma errada.
# Portanto, para cada variável categória nominal, é criada uma variável dummy.
# que irá armazenar 1 (se o valor corresponder) e 0 (caso não seja o valor).
# Exemplo: atributo raça é uma categórica nominal, para cada valor possível,
# será criada uma variável dummy, ou seja, para cada possível valor, será
# um atributo. Podemos pensar que os valores serão uma nova tabela de 0 e 1.
