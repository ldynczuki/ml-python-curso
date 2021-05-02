import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier


base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# LabelEncoder em conjunto com o OneHotEncoder
# OneHotEnconder é utilizado para variáveis categóricas nominais
labelenconder_previsores = LabelEncoder()
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                        [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# LabelEncoder para variáveis categóricas
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de dados em dados de treino e dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)


# Criando a instância do classificador MLPClassifier()
classificador = MLPClassifier(verbose=True, max_iter=1000, tol=0.000010,
                              solver='adam',
                              hidden_layer_sizes=(100),
                              activation='relu')

# Treinamento da rede neural
classificador.fit(previsores_treinamento, classe_treinamento)

# Predição de novos registros após o treinamento da rede neural
previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# Exibe qual é a função de ativação da camada de saída
print(classificador.out_activation_)

# No scikit-learn não é possível alterar a função de ativação da camada de saída
# será escolhido automaticamente pelo algoritmo
# Neste exemplo, a função é a sigmoide (logistic) porque temos apenas 2 classes
