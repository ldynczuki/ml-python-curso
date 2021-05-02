"""
StrafifieldKFold (estratificação) é o processo que vai reorganizar os dados com
o intuito de garantir que cada um dos folds (divisões) da base de dados seja um
bom representante do todo.
Essa abordagem é mais interessante do que simplesmente utilizar o método
cross_val_score (validação cruzada) que vimos no script "validacao_cruzada_cross_val_score.py"
Pois a validação cruzada stratifieldkfold fará um pré-processamento para
garantir que teremos uma boa distribuição dos dados entre as classes.
É recomendado utilizar essa abordagem quando for trabalhar com validação cruzada

Matriz de confusão utilizando validação cruzada: Como antes trabalhamos com 1
resultado, tínhamos uma matriz de confusão dos resultados. Entretanto, como
temos 10 resultados, teremos 10 matrizes de confusão

"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
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


# Abaixo é apresentado como funciona o método zeros() do np
# Temos que criar uma matriz que recebe o shape de previsores para armazenar
# os índices dos previsores e adicionar 1 coluna
# Isso é necessário para a abordagem StratifieldKFold, pois ela requer que nossa classe
# seja uma matriz (tenha linha e coluna), e do jeito que atualmente está (2000, )
# não está com colunas
# Veja que a variável b cria uma matriz com o formato (2000, 1) com 2000
# registros 0 por causa do np.zeros, mas com o shape na posição 0 de previsores
# e 1 coluna.
a = np.zeros(5) 
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0], 1))


# Parâmetro n_splits para dividir em 10 partes
# Parâmetro shuffle True irá garantir a aleatoriedade dos dados
# Parâmetro random_state é a semente geradora, usado quando shuffle = True
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

resultados = []
matrizes = []


# Passamos como parâmetro do kfold.split os previsores e a classe no formato matriz
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0], 1))):
    # print("Índice treinamento: ", indice_treinamento, "Índice teste: ", indice_teste)
    
    # Criação do classificador NB
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

# Para poder verificar a média das 10 execuções, devemos converter o tipo de
# dados do "resultados" que é list e vamos transformar em np array 
resultados = np.asanyarray(resultados)
resultados.mean()
resultados.std()

# Criamos a variável matriz_final para calcular a média das 10 matrizes de confusão
matriz_final = np.mean(matrizes, axis=0)
