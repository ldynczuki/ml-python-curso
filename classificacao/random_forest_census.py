# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

labelenconder_previsores = LabelEncoder()
# Aqui utilizamos direto o OneHotEnconder, mas poderíamos fazer o LabelEnconder
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                        [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

# Criação da instância da classe RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# Nota:
# O parâmetro n_estimators define a quantidade de árvores na floresta
# altere o valor e reexecute para ver o valor da precisão ser alterado.

# ATENÇÃO: O interessante é apagar as variáveis e reexecutar tirando algumas
# etapas do pré-processamento para tentar verificar os resultados finais.
