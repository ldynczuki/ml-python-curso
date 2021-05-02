"""
Neste momento iremos criar variável para armazenar o nível de confiabilidade
e também iremos criar variáveis para armazenar as probabilidaes dos resultados. 
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))

novo_registro = [[50000, 40, 5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1, 1)
scaler = StandardScaler()
novo_registro = scaler.fit_transform(novo_registro)
novo_registro = novo_registro.reshape(-1, 3)

resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)


# ARMAZENANDO A PROBABILIDADE E CONFIANÇA DOS CLASSIFICADORES
# Criando variáveis para armazenar a proba e confiabilidade dos classificadores
# Caso dêa mensagem de probability=False, devemos colocar isso no modelo salvo, ou seja retreinar o classificador
probabilidade_svm = svm.predict_proba(novo_registro)
confianca_svm = probabilidade_svm.max()

probabilidade_random_forest = random_forest.predict_proba(novo_registro)
confianca_random_forest = probabilidade_random_forest.max()

probabilidade_mlp = mlp.predict_proba(novo_registro)
confianca_mlp = probabilidade_mlp.max()


# Criando variáveis para serem contadores dos resultados das predições
paga = 0
nao_paga = 0

# Variável utilizada para verificar a rejeição de classificador
# de acordo com o valor da confiabilidade do mesmo.
confianca_minima = 0.98


# Verificando a resposta de cada classificador e incrementando nos contadores
if confianca_svm >= confianca_minima:
    if resposta_svm[0] == 1:
        paga += 1
    else:
        nao_paga += 1


if confianca_random_forest >= confianca_minima:
    if resposta_random_forest[0] == 1:
        paga += 1
    else:
        nao_paga += 1


if confianca_mlp >= confianca_minima:
    if resposta_mlp[0] == 1:
        paga += 1
    else:
        nao_paga += 1


if paga > nao_paga:
    print("Cliente pagará o empréstimo.")
elif paga == nao_paga:
    print("Resultado empatado.")
else:
    print("Cliente não pagará o empréstimo.")
