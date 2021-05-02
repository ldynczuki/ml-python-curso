"""
Valor retornado igual a 0 significa que o novo cliente não irá pagar empréstimo
Valor retornado igual a 1 significa que o novo cliente irá pagar empréstimo
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

# Criando variáveis para serem contadores dos resultados das predições
paga = 0
nao_paga = 0


# Verificando a resposta de cada classificador e incrementando nos contadores
if resposta_svm[0] == 1:
    paga += 1
else:
    nao_paga += 1
    

if resposta_random_forest[0] == 1:
    paga += 1
else:
    nao_paga += 1


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
