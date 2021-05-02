"""
Neste script não faremos manualmente a criação da rede, camadas e ligações
"""

# Módulo shortcuts indica que não precisamos criar manualmente as etapas
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

# Criação da rede neural
# Veja como é mais simples a criação da rede neural em comparação com o exemplo
# do script "exemplo_pybrain1.py" onde precisamos criar objetos para cada
# camada
# Os parâmetros 2, 3, 1 significam a quantidade de neurônios da camada de
# entrada, oculta e saída, respectivamente.
rede = buildNetwork(2, 3, 1)

# Podemos configurar as camadas da rede no comando buildnetwork como abaixo
# rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer,
#                    hiddenclass = SigmoidLayer, bias = False)


# Visualizando algumas propriedades da rede neural criada acima
# o print abaixo apresenta informações da camada de entrada "in" que é uma
# uma Linear Layer porque não vamos aplicar nenhuma função nesta camada
print(rede['in'])

# exibe propriedade da rede na camada oculta que possui a função de ativação
# Sigmoide
print(rede['hidden0'])

# Visualiza propriedade da rede na camada de saída que não possui a função de
# ativação, ou seja, é uma Linear Layer
print(rede['out'])

# Exibe a propriedade Bias da rede neural criada
print(rede['bias'])


# Criação da base de dados
# O parâmetro 2 significa que teremos 2 atributos previsores e o parâmetro 1
# que teremos 1 classe
base = SupervisedDataSet(2, 1)
# Adicionando exemplos com os valores dos atributos e o seu resultado
# A lógica dos comandos abaixo é a seguinte, se temos os valores 0 e 0 para os
# atributos previsores, o resultado da classe é 0 (como na linha abaixo)
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

# Exibindo os dados da base
print(base)
print(base['input'])
print(base['target'])


# Realizando o treinamento
treinamento = BackpropTrainer(rede, dataset=base, learningrate=0.01,
                              momentum=0.06)


# For para iterar a quantidade de treinamento da rede neural
for i in range(1, 30000):
    erro = treinamento.train()
    # Exibe o erro do treinamento de 1000 em 1000
    if i % 1000 == 0:
        print("Erro: %s" % erro)


# Após o treinamento, podemos fazer uma predição na rede
print(rede.activate([0, 0]))
print(rede.activate([0, 1]))
print(rede.activate([1, 0]))
print(rede.activate([1, 1]))

# Obs: podemos utilizar a função round() para arredondar o valor
rede.

"""
Atenção, o resultado apresentado com notação científica, ou seja, com valores
com o - significa que é a quantidade de zeros que existem, ou seja, quanto
maior o valor após o sinal de menos, maior é a quantidade de zeros e MENOR
é o número
no exemplo de predição que fizemos, passamos os valores [0, 0], portanto,
deverá ser retornado um valor muito próximo de zero, e no exemplo atual
foi retornado o valor [7.40518757e-14] ou seja, 14 zeros antes de 7.
Já os valores que foram passados [0, 1] deve ser retornado valor próximo de 1
 e foi retornado o valor 1.
"""

