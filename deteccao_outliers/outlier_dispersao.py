"""
O objetivo deste script é tentar detectar outliers nesta base de dados utilizando
o gráfico de dispersão
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


base = pd.read_csv('credit-data.csv')

# Neste exemplo, iremos tratar os dados ausentes "nan" excluindo-os
# não é a melhor maneira, em outros exemplos substuímos pelo valor da média
base = base.dropna()

# substituindo os valores com idade negativa pela média da idade da base
base.loc[base.age < 0, 'age'] = 40.92


# Outlier - income x age
plt.scatter(base.iloc[:, 1], base.iloc[:, 2])


# Outlier - income x loan
plt.scatter(base.iloc[:, 1], base.iloc[:, 3])


# Outlier - age x loan
plt.scatter(base.iloc[:, 2], base.iloc[:, 3])


"""
Base: credit-data.csv
Após tratar os dados com valores nan e as datas negativas, não tivemos outliers
"""


"""
Base: census.csv
"""
base_census = pd.read_csv('census.csv')

# Outlier - age x final weight
plt.scatter(base_census.iloc[:, 0], base_census.iloc[:, 2])
