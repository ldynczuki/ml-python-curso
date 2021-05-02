import pandas as pd
from apyori import apriori


dados = pd.read_csv('mercado.csv', header=None)
transacoes = []

# Inserindo os dados do DataFrame em um formato de lista, separados cada um
for i in range(0, 10):
    transacoes.append([str(dados.values[i, j]) for j in range(0, 4)])


regras = apriori(transacoes, min_support=0.3, min_confidence=0.8, min_lift=2,
                 min_length=2)


resultados = list(regras)
resultados

resultados2 = [list(x) for x in resultados]
resultados2

resultadoFormatado = []

for j in range(0, 3):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])

resultadoFormatado
