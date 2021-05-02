#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 10:13:55 2019

@author: lucas
"""

# Iremos trabalhar com a biblioteca Orange
import Orange

base = Orange.data.Table('credit-data.csv')

# Exibe o nome dos atributos da base
base.domain

# A biblioteca Orange não necessita que façamos o pré-processamento dos dados

# Ao executar a linha abaixo, será exibido um erro "ValueError"
# Isso ocorre porque o classificador CN2 não encontrou qual é o atributo
# classe. Para corrigir, abra a base de dados e coloque c# antes do nome do
# atributo. Neste caso, alterei o nome "risco" para "c#risco" no arquivo
# risco-credito.csv

# Nesta base de dados, credit-data.csv, não vamos utilizar o atributo id
# ou seja, o "clientid", então, no arquivo, colocamos i# antes do nome,
# isto fará com que ignore a coluna "clientid".


# Neste momento, iremos dividir as bases em treinamento e testes
# No Orange é um pouco diferente do que fazemos com o sklearn
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)

# Após dividir a base conforme fizemos anteriormente, agora iremos receber
# os dados em variáveis distintas
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# Verificando o tamanho das bases de treinamento e teste
print(len(base_treinamento))
print(len(base_teste))

# Criando instância do classificador CN2 Leaner
cn2_learner = Orange.classification.rules.CN2Learner()

# Neste momento estamos treinando o nosso modelo, gerando as regras de indução
classificador = cn2_learner(base_treinamento)


# Visualizando as regras geradas anteriormente
for regras in classificador.rule_list:
    print(regras)


# Agora que temos as regras criadas, iremos submeter a nossa base_teste
# Passamos uma lista de classificador, onde podemos ter mais de um classificador ao mesmo tempo
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])

# Neste momento iremos verificar a acurácia do classificador
# Entretanto, o comando abaixo gerou erros, que mesmo no código atualizado não corrige.
print(Orange.evaluation.CA(resultado))
