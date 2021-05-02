"""
A ideia deste algoritmo é realizar uma comparação, se utilizando todos os atributos
previsores desta base terá melhores resultados do que utilizando o pca para
extrair os melhores atributos previsores

Iremos utilizar o algoritmo de RandomForest sem realizar o OneHotEncoder nos previsores
e sem realizar o LabelEncoder() na classe, fazendo apenas nos previsores
pois, quando estávamos estudando os algoritmos de classificação, foi este algoritmo
com estas definições que tiveram melhores resultados

Kernel PCA é considerado algoritmo de aprendizagem não supervisionado para a
redução de dimensionalidade
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier



base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Labelencoder é utilizado para transformar variáveis categóricas nominais (sem ordem) em variáveis quantitativas contínuas
labelenconder_previsores = LabelEncoder()
previsores[:, 1] = labelenconder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelenconder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelenconder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelenconder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelenconder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelenconder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelenconder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelenconder_previsores.fit_transform(previsores[:, 13])


# PRIMEIRAMENTE NÃO IREMOS REALIZAR O ONEHOTENCODER() PARA FACILITAR O ENTENDIMENTO
# onehotencoder = OneHotEncoder(categorical_features = [1, 3, 5, 6, 7, 8, 9, 13])
# previsores = onehotencoder.fit_transform(previsores).toarray()

# utilizando LabelEncoder() junto do OneHotEncoder()
# onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
#                        [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
# previsores = onehotencorder.fit_transform(previsores).toarray()


# LabelEncoder para variáveis categóricas da classe
# labelencorder_classe = LabelEncoder()
# classe = labelencorder_classe.fit_transform(classe)

# Escalonamento dos dados
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe,
                                                                                              test_size=0.15, random_state=0)

"""
REDUÇÃO DE DIMENSIONALIDADE
"""
# Criação do objeto da classe PCA 
pca = PCA(n_components=6)  # n_componentes é o valor de número de atributos que queremos
previsores_treinamento = pca.fit_transform(previsores_treinamento)
previsores_teste = pca.transform(previsores_teste)  # como utilizamos o fit_transform, o método já se adaptou aos dados, por isso usamos transform apenas, para usar a mesma escala

# Podemos selecionar os atributos que possuem os maiores coeficientes (com isso é somado os valores dos coeficientes) e então é "escolhido os melhores" dos atributos
componentes = pca.explained_variance_  # este método retorna os coeficientes de cada atributo selecionado (escolhido no método durante a criação do objeto da classe PCA)

"""
Perceba que após escolher n_componentes igual a 6, a dimensionalidade das
variáveis previsores_treinamento e previsores_teste reduziu de 14 atributos
(original) para 6 atributos que foi o valor que foi passado no PCA()
"""

# Após realizar a redução de dimensionalidade, iremos executar o algoritmo de classificação
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)  # n_estimators: número de árvores de decisão
classificador.fit(previsores_treinamento, classe_treinamento)  # treinamento do modelo

previsoes = classificador.predict(previsores_teste)  # execução da previsão com o modelo treinado e previsoes_teste

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)


"""
É importante salientar que o valor original do RandomForestClassifier foi de 0.84
utilizando os 14 atributos e utilizando o PCA(n_componentes=6) que diminui dimensionalidade
de 14 atributos para 6.
Veja que, abaixou muito pouco o valor da precisão e por outro lado o algoritmo ficou
muito menos complexo, por causa da diminuição da complexidade.
Então é importante analisar se vale a pena diminuir a dimensionalidade ou não
tudo irá depender dos resultados da precisão e velocidade de execução
"""