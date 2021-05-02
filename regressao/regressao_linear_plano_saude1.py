"""
REGRESSÃO LINEAR SIMPLES => Existe apenas 1 atributo previsor
REGRESSÃO LINEAR MÚLTIPLA -> Mais de 1 atributo previsor

Variável x são as variáveis independentes (explantatórias)
Variável y é a variável independente (a qual buscamos prever)

A correlação vai de -1 a 1, onde valores -1 e 1 são correlações perfeitas
que significa que existe total correação. Se o for 1 significa que uma variável
aumenta e a outra também, correlação positiva.
Se o valor for -1 significa que a correlação é negativa, onde uma variável
aumenta e a outra diminui.
O valor 0 para correlação significa que não há correlação.

# NOTA:
É muito importante a análise da correlação, pois se tivermos valores muito
baixos (próximos de 0) provavel que seu modelo de regressão linear não vai se
adaptar muito bem a sua base de dados.

Neste exemplo como temos uma correlação de 0.93, é uma correlação forte,
portanto, é um primeiro indicativo que o modelo de regressão linear pode ser
usado.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot


base = pd.read_csv("plano-saude.csv")

X = base.iloc[:, 0].values
y = base.iloc[:, 1].values

# Iremos verificar se existe algum tipo de correlação entre as variáveis x e y
# A palavra correlação vem da estatística e indica quanto uma variável está
# próxima de outra, ou seja, se uma variável está ou não correlacionada.
correlacao = np.corrcoef(X, y)

# Veja o resultado da correlação é uma matriz 2 por 2 onde, a intersecção do 0 com 0
# o resultado é 1, ou seja, a correlação da variável idade com ela mesma é 1, e
# a correlação da variável 0 (idade) com a 1 (custo) é de 0.9390 (93%)
# Quanto maior o valor, melhor, ou seja, maior correlação, por isso na diagonal
# principal temos os valores de 1, que são as correlações das variáveis com
# elas mesmas, que é totalmente correlacionada (100%)

# Iremos alterar a forma (shape) das variáveis para matriz, pois os algoritmos
# do scikit-learn precisam que os parâmetros sejam matrizes
# Usamos o -1 para não mexer nas linhas e o 1 para adicionar uma coluna
X = X.reshape(-1, 1)


# Criação do regressor
regressor = LinearRegression()

# treinamento do regressor
# O treinamento do algoritmo de regressão linear é para encontrar os melhores
# parâmetros do b0 e b1 que são a constante e o coeficiente, respectivamente
regressor.fit(X, y)


# Exibe o valor de b0 (constante)
print(regressor.intercept_)


# Exibe o valor de b1 (coeficiente)
# Temos um coeficiente para cada uma das variáveis previsores
# Como temos apenas 1 variável previsora neste exemplo, será apenas 1 valor
print(regressor.coef_)


# Utiliza a lib matplot para exibir um gráfico com os valores (Reais) de X e y
plt.scatter(X, y)

# Criando linha da regressão linear após encontrar o b0 e b1 (Valores previstos)
# Será adicionado a linha da regressão linear de acordo com a previsão
plt.plot(X, regressor.predict(X), color='red')

# Adicionando título ao gráfico
plt.title("Regressão Linear Simples")
# adicionando nome para os eixos
plt.xlabel("Idade")
plt.ylabel("Custo")


# Realizando a previsão de um novo registro
# previsão pessoa com 40 anos
previsao1 = regressor.predict(np.array(40).reshape(1, -1))
# Podemos reescrever a previsão acima utilizando os parâmetros b0 + b1 * valor X
previsao2 = regressor.intercept_ + regressor.coef_ * 40

# O valor previsto para o custo do plano de saúde foi de 1.915 reais


# Verificando como a correlação está se comportando
# O valor retornado é diferente do valor de coeficiente de correlação encontrado
# anteriormente, o valor do score é o valor de correlação do seu modelo de regressão
# Quanto valores maiores melhor
# Mede quão bom é o seu modelo
score = regressor.score(X, y)


# Valores de residuais são as distâncias que existem entre os pontos (reais) e a linha reta (prevista)
# Passamos como parâmetro o regressor que já foi treinado
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()
