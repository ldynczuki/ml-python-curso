import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# criando o classificador NB
classificador = GaussianNB()


# CROSS VALIDATION
# Quando utilizamos o cross validation não precisamos realizar manualmente
# a separação das bases de dados de treinamento e base de testes
# o valor de K que definirmos fará a separação da base de dados de treinamento
# e testes e após gerar o resultado, será feito uma nova divisão da base
# com outra "fatia" para treinamento e testes, que é o conceito de
# validação cruzada
# Parâmetro cv é o valor de K que é a quantidade de folds, ou seja,
# quantidade de vezes que será executado, e de forma automática mudando as bases
# cv é de CrossValidation
resultados = cross_val_score(classificador, previsores, classe, cv=10)

# Como temos 10 resultados de predições, devemos tirar a média deles
resultados.mean()

# Podemos também verificar o desvio padrão
# A importância do desvio padrão é se o valor dele for muito alto, signfica que
# pode estar ocorrendo overfitting em seu modelo
resultados.std()


# NOTA:
# Os resultados podem ser vistos na variável "resultados" que serão 10 registros
# com o resultado das predições em cada uma das 10 vezes executadas.