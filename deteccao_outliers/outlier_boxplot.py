"""
O objetivo deste script é tentar detectar outliers nesta base de dados utilizando
o gráfico de boxplot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


base = pd.read_csv('credit-data.csv')

# Neste exemplo, iremos tratar os dados ausentes "nan" excluindo-os
# não é a melhor maneira, em outros exemplos substuímos pelo valor da média
base = base.dropna()

# Outlier - IDADE
# Veja que temos 3 registros com valores negativos para o atributo AGE (IDADE)
plt.boxplot(base.iloc[:, 2], showfliers=True)  # showfliers é TRUE por default que significa que irá apresentar os outliers
outliers_age = base[(base.age < -20)]  # capturando registros pelo atributo age

#  Outlier - Loan (dívida)
# Veja que temos 2 registros de outliers com valores de dívidas muito altas
plt.boxplot(base.iloc[:, 3], showfliers=True)
outliers_loan = base[(base.loan > 13400)]
