import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age']=40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

resultados30 = []

for i in range(30):
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0], 1))):
        #classificador = GaussianNB()
        classificador = DecisionTreeClassifier()
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento]) 
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
   
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)

resultados30 = np.asarray(resultados30)

for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))

