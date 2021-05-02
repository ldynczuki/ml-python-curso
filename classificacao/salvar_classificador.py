"""
Utiliza-se a biblioteca pickle para salvar arquivos no disco!
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# importamos apenas os classificadores acima que tiveram os melhores resultados

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Criando as instâncias dos classificadores e treinando (fit) os classificadores
# Utilizamos parâmetro "probability=True" para que seja possível armazenar a probabilidade da predição
classificadorSVM = SVC(kernel='rbf', C=2.0, probability=True)
classificadorSVM.fit(previsores, classe)

classificadorRandomForest = RandomForestClassifier(n_estimators=40, 
                                                   criterion='entropy')
classificadorRandomForest.fit(previsores, classe)

classificadorMLP = MLPClassifier(verbose=True, max_iter=1000, tol=0.000010,
                                 solver='adam', hidden_layer_sizes=(100),
                                 activation='relu', batch_size=200,
                                 learning_rate_init=0.001)
classificadorMLP.fit(previsores, classe)


# SALVANDO OS CLASSIFICADORES - lib pickle
# Parâmetro wb para efetivar o salvamento em disco do classificador
# w de write e b de binary
pickle.dump(classificadorSVM, open('svm_finalizado.sav', 'wb'))
pickle.dump(classificadorRandomForest, open('random_forest_finalizado.sav', 'wb'))
pickle.dump(classificadorMLP, open('mlp_finalizado.sav', 'wb'))
