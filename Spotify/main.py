# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:59:30 2022

@author: Hossein.JvdZ
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("spotify_dataset.csv", low_memory = False)
dataset.drop(dataset.columns[[0,1,2,3,-2]],1 , inplace = True)

X = dataset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(), [-7,-4])], remainder="passthrough")

X_train = np.array(ct.fit_transform(X))

from sklearn.cluster import KMeans

km = KMeans(n_clusters = 10, init = "k-means++", random_state = 5)
y_means = km.fit(X_train)


