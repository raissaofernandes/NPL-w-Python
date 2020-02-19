from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

dados_textos = pd.read_csv("C:\\Users\\Raissa\\analise\\brutos\\classificados_textos.csv", sep=";", encoding='ISO-8859-1')

dados_textos.head()

X = dados_textos["Text"].values.astype('U')
y = dados_textos["Voto"].values.astype('U')

vectorizer = CountVectorizer(analyzer = "word")
X_vetor = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vetor, y, test_size = 0.3)

classificador = svm.SVC(C=1.0)

#treinando o modelo
classificador.fit(X_train, y_train)

classificador.predict(X_test)

y_pred = classificador.predict(X_test)

classificador.score(X_test,y_test)

print(classification_report(y_test, y_pred))
