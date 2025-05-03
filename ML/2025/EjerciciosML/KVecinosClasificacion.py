import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_clasificacion(datos, k=3):
    y_train = datos["species"]
    X_train = datos.drop( "species", axis = 1 )
    knn = KNeighborsClassifier( n_neighbors = k )
    knn.fit( X_train, y_train )
    return knn