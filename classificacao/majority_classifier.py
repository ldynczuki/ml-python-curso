#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:35:02 2019

@author: lucas
"""

import Orange
from collections import Counter

base = Orange.data.Table('credit-data.csv')
base.domain

base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

print(len(base_treinamento))
print(len(base_teste))

# Criando instância do classificador Majority Leaner
classificador = Orange.classification.MajorityLearner()

resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])

print(Orange.evaluation.CA(resultado))

print(Counter(str(d.get_class()) for d in base_teste))


# Este algoritmo é denominado base line classifier.
# Ou seja, como ele é simplesmente um "count" da maioria, se o resultado
# de um outro algoritmo de classificação der resultado inferior, é mais
# indicado utilizar este algoritmo base line.

# NOTA:
# Na realidade, o Majority Leaner não chega a ser um "Leaner", ele irá
# classificar os novos registros com base na maioria
# ou seja, se a maioria dos registros é de determinada classe, o próximo
# registro a ser classificado será da mesma classe da maioria
# ele é basicamente um "count", e o que for maior, será a classe dos novos
# registros.

# Reveja a aula 59 - Classificador base - majority leaner (machine A a Z)
