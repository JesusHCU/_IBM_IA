# prediccion_abandono.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # ← Cambio clave
from sklearn.metrics import accuracy_score, classification_report


def entrenar_modelo(data):
    """
    Entrena un Árbol de Decisión usando la última columna como target.
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(random_state=42, max_depth=3)  # Árbol de Decisión
    modelo.fit(X_train, y_train)

    # Evaluación
    y_pred = modelo.predict(X_test)
    print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")
    print("Reporte:\n", classification_report(y_test, y_pred))

    return modelo


def predecir_abandono(modelo, nuevo_estudiante):
    prediccion = modelo.predict([nuevo_estudiante])[0]
    return "Abandona" if prediccion == 1 else "No abandona"


# Ejemplo de uso
if __name__ == "__main__":
    data = pd.DataFrame({
        'edad': [20, 22, 19],
        'horas_estudio': [5, 8, 3],
        'asistencia': [90, 75, 60],
        'abandono': [0, 0, 1]  # Última columna
    })

    modelo = entrenar_modelo(data)
    print("Predicción:", predecir_abandono(modelo, [21, 7, 80]))