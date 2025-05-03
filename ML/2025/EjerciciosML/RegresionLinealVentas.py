import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Función de regresión lineal
def regresion_ventas(datos):
    X = datos[['TV', 'Radio', 'Periodico']]
    Y = datos['Ventas']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    modelo = LinearRegression()

    modelo.fit(X_train, y_train)

    return modelo