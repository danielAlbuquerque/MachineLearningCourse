#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:08:26 2019

@author: danielalbuquerque
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')
base.describe()

base.loc[base['age'] < 0]

#apagarColuna
base.drop('age', 1, inplace=True)

#apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)

#preenhcer valores com a media
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

# preenchendo dados faltantes
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

#
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
