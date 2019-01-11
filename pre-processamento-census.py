import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

labelencoder_previsores = LabelEncoder()

cols = [1, 3, 5, 6, 7, 8, 9, 13]
#cols = [0]
for i in cols:
    print(i)
    previsores[:, i] = labelencoder_previsores.fit_transform(previsores[:, i])

onehotencoder = OneHotEncoder(categorical_features=cols)
previsores = onehotencoder.fit_transform(previsores).toarray()

# ajustando classe
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# escalando
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)