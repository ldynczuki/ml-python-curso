"""
Neste script faremos manualmente a criação da rede, das camadas e a ligação
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection


# Criação da rede neural
rede = FeedForwardNetwork()


# CRIAÇÃO DE CAMADAS

# Colocamos o parâmetro 2 que significa a quantidade de neurônios na camada
# Para a camada de entrada utilizamos a classe Linear Layer porque significa
# que não utilizaremos nenhuma função para os valores da camada de entrada
camadaEntrada = LinearLayer(2)

# Colocamso o parâmetro 3 que significa a quantidade de neurônios na camada
# Para a camada oculta utilizamos a classe SigmoideLayer porque os valores
# da camada oculta serão passados pela função de ativação Sigmoide
camadaOculta = SigmoidLayer(3)
# A camada de saída recebe o parâmetro 1 porque terá apenas 1 neurônio
# A camada de saída também passará pela função de ativação Sigmoide
camadaSaida = SigmoidLayer(1)

# Criamos dois objetos do tipos BiasUnit porque temos o bias para a camada
# oculta e outra para a camada de saída, por isso são 2 objetos BiasUnit()
bias1 = BiasUnit()
bias2 = BiasUnit()

# Depois de definidas as camadas, precisamos agora é adicioná-las à rede
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)

# Após a adição das camadas na rede neural, precisamos ligar as camadas
# Criamos um objeto do tipo FullConnection porque um neurônio irá se ligar
# com todos os outros
# Neste objeto abaixo estamos realizando a ligação da camada de Entrada
# na camada Oculta utilizando o objeto FullConnection
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
# Também devemos conectar as bias com a camada oculta e bias para a saída
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

# Chamamos o método sortModules() que definitivamente a rede será construída
rede.sortModules()

# Exibindo algumas informações da rede neural construída
print(rede)
# Exibe os pesos aleatórios que foram criados entre a camada de entrada e Oculta
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)
