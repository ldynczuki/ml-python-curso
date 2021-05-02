#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:38:14 2019

@author: lucas
"""

# Neste script iremos utilizar a biblioteca Orange com classificador CN2 Leaner
import Orange

base = Orange.data.Table('risco-credito.csv')

# Exibe algumas propriedades da base importada
base.domain

# Criando instância do algoritmo CN2Leaner da biblioteca Orange
cn2_leaner = Orange.classification.rules.CN2Learner()

# Ao executar a linha abaixo, será exibido um erro "ValueError"
# Isso ocorre porque o classificador CN2 não encontrou qual é o atributo
# classe. Para corrigir, abra a base de dados e coloque c# antes do nome do
# atributo. Neste caso, alterei o nome "risco" para "c#risco" no arquivo
# risco-credito.csv
classificador = cn2_leaner(base)

# Criando um for para iterar quais foram as regras criadas pelo classificador
for regras in classificador.rule_list:
    print(regras)

# Não é necessário chamar o método predict(), apenas os valores
# Estamos passando os mesmos valores referente aos outros algoritmos
# com a diferença é que aqui passamos o valor como está na base original
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'],
                          ['ruim', 'alta', 'adequada', '0_15']])

for i in resultado:
    print(base.domain.class_var.values[i])


# NOTA: Perceba que o classificador CN2 Leaner não solicita que particionamos
# a base em "previsores" e "classe" como fizemos anteriromente.
# O que é solicitado é que a gente indique qual é o atributo classe com o c#
