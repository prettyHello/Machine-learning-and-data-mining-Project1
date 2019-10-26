# On importe les librairies dont on aura besoin
#import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import cross_val_score

# On charge le dataset
Training_Set = pd.read_csv('../Data/Adult_train.csv', sep=',')
taille = len(Training_Set)

#on change la colonne de revenu(=salary) en binaire pour prédire
Training_Set['salary_bi'] = Training_Set.apply(lambda row: 1 if '>50K'in row['salary'] else 0, axis=1)

# Remove redundant columns
Training_Set = Training_Set.drop(['salary'], axis=1)
