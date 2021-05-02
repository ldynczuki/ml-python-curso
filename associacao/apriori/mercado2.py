import pandas as pd
from apyori import apriori


dados = pd.read_csv('mercado2.csv', header=None)
transacoes = []

# Inserindo os dados do DataFrame em um formato de lista, separados cada um
for i in range(0, 7501):
    transacoes.append([str(dados.values[i, j]) for j in range(0, 20)])


# Como estamos trabalhando com base de dados real, o valor do parâmetro
# min_support não pode ser tão alto como estava 0.3, porque este valor significa
# que um determinado conjunto aparece 30% na base de dados,
# vamos fazer um cálculo, que queremos descobrir uma associação de produtos
# que são vendidos 4x na semana (4 x 7 = 28) e então dividimos pelo total de
# transações (7501) ou seja (28 / 7501 = 0.003971067933626436), então = 0.003
# Vamos também mudar o valor de min_confidence e min_lift
    
# Atenção: se o valor de confiança for muito alto, serão gerados regras
# muito "óbvias". Isso ocorre também com o parâmetro min_support (suporte)
regras = apriori(transacoes, min_support=0.003, min_confidence=0.2,
                 min_lift=3.0, min_length=2)

resultados = list(regras)
resultados

resultados2 = [list(x) for x in resultados]
resultados2

resultadoFormatado = []

# iremos iterar nas 5 primeiras regras criadas
for j in range(0, 5):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])

resultadoFormatado