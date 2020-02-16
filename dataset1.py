from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import seaborn as sns

#NaN values
dados_classificador = pd.read_csv("C:\\Users\\Raissa\\analise\\brutos\\classificados_tres.csv", sep=";", encoding='ISO-8859-1')

dados_classificador.head()

dados_classificador.describe()

data = dados_classificador.values
dados_df = pd.DataFrame(data, columns = ['AFederalLuta', 'afederalmeproporcionou', 'balburdia', 'TsunamiDaEducacao', 'MeuFilhoNaoVai', 
                                         'Marolinha30M', 'NaRuaPelaEducacao', 'TiraAMaodoMeuIF', 'Tsunami13A'])
                                         
sns.pairplot(dados_df[['AFederalLuta', 'afederalmeproporcionou', 'balburdia', 'TsunamiDaEducacao', 'MeuFilhoNaoVai',
                       'Marolinha30M', 'NaRuaPelaEducacao', 'TiraAMaodoMeuIF', 'Tsunami13A']])
