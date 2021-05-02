# -*- coding: utf-8 -*-
"""
Spyder Editor

Script para pré-processamento da base de dados de crédito bancário.
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Importação da base de dados de crédito
base = pd.read_csv('credit-data.csv')

# Exibindo algumas estatísticas da base de dados de crédito
print(base.describe())

"""
TRATAMENTO DE VALORES INCONSISTENTES
"""

# Localizando registros que o atributo "age" é menor que 0
print(base.loc[base['age'] < 0])

"""
 Perceba que temos 3 registros que possuem idade menor que 0
 A seguir serão apresentadas maneiras de corrigir estas inconsistências.
 - Veremos 4 formas:
"""

# 1 - Apagar a coluna 'age' inteira
# Não é indicado realizar a exclusão de toda a coluna
# base.drop('age' , 1, inplace=True)

# 2 - Apagar somente os registros com problema
# Também não é indicado apagar os registros com problema
# base.drop(base[base.age < 0].index, inplace=True)

# 3 - Preencher os valores inconsistentes manualmente
# Também não é muito viável, pois veja nosso exemplo, não temos o nome
# Como saberemos de quem são os registros inconsistentes

# 4 -  Preencher os valores com a média geral do atributo inconsistente
base.mean()
base['age'].mean()

# Veja que acima foi feita o cálculo de média contando com valores negativos
# Não podemos fazer isso, portanto, faremos a média apenas de valores corretos.
base['age'][base.age > 0].mean()

# Atribuindo o valor da média correta na idade dos 3 registros errados.
base.loc[base.age < 0, 'age'] = 40.92

"""
TRATAMENTO DE VALORES FALTANTES
"""

# Verificando se existem valores nulos (faltantes) no atributo 'age'
pd.isnull(base['age'])

# Como o comando anterior mostra todos os resultados, fica um pouco ruim.
# Para isso, veja o outro comando abaixo.
base.loc[pd.isnull(base['age'])]


"""
Para trabalhar com os algoritmos de aprendizagem de máquina
é preciso fazer uma "divisão" dos atributos, ou seja,
teremos os atributos previsores e o atributo "meta" ou "classe"

Teremos uma variável que irá armazenar somente os atributos "previsores"
e teremos uma variável que irá armazenar somente o atributo "meta" ou "classe".

Em nossa base de crédito, o atributo "classe" é o "default"
E todo o restante são os atributos "previsores".
"""

# Criação da variável "previsores" que irá armazenar atributos previsores.
# Veja que não estamos selecionando a coluna 'cliendid'

# Dica: Ao utilizar algoritmos de ML, você pode retirar os atributos ID
# No comando abaixo, estamos selecionando todas as linhas e colunas de 1 a 3.
previsores = base.iloc[:, 1:4].values

# Criação da variável "classe" que irá armazenar atributo "meta" ou "classe".
classe = base.iloc[:, 4].values


# Preenchendo valores faltantes utilizando biblioteca sklearn
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


# Escalonamento dos atributos (Padronização dos atributos na mesma escala)
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
