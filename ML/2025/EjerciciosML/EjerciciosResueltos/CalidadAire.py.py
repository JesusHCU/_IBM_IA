import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Función de regresión logística
def regresion_logistica(datos):
    X = datos[['Edad', 'Colesterol']]
    y = datos['Enfermedad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = LogisticRegression()

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    precision = accuracy_score(y_test, y_pred)
    return modelo
