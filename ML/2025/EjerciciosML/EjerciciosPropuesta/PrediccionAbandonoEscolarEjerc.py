import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def cargar_datos():
    """
    TODO: Implementa la carga y preparación de datos
    Hint: Crea un DataFrame con datos de estudiantes
    """
    # TODO: Crea un DataFrame con estas columnas:
    # - edad: entre 18 y 25 años
    # - horas_estudio: entre 0 y 10 horas
    # - asistencia: entre 0 y 100%
    # - abandono: 0 (no abandona) o 1 (abandona)
    data = None
    return data


def entrenar_modelo(data):
    """
    TODO: Implementa el entrenamiento del modelo
    Hint: Usa DecisionTreeClassifier para predecir abandono
    """
    # TODO: Separa features (X) y target (y)
    # Hint: El target es la última columna 'abandono'
    X = None
    y = None
    
    # TODO: Divide los datos en train y test
    # Hint: Usa train_test_split con test_size=0.2
    
    # TODO: Crea y entrena el modelo
    # Hint: Usa max_depth=3 para evitar overfitting
    modelo = None
    
    # TODO: Evalúa el modelo
    # Hint: Usa accuracy_score y classification_report
    
    return modelo


def predecir_abandono(modelo, nuevo_estudiante):
    """
    TODO: Implementa la predicción para un nuevo estudiante
    Hint: Usa el modelo entrenado para predecir
    """
    # TODO: Realiza la predicción
    # Hint: Retorna "Abandona" o "No abandona"
    pass


# Código de prueba
if __name__ == "__main__":
    # Crear datos de ejemplo
    print("Preparando datos...")
    datos = pd.DataFrame({
        'edad': [20, 22, 19, 21, 23],
        'horas_estudio': [5, 8, 3, 6, 7],
        'asistencia': [90, 75, 60, 85, 95],
        'abandono': [0, 0, 1, 0, 0]
    })
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    modelo = entrenar_modelo(datos)
    
    # Probar con nuevo estudiante
    nuevo_estudiante = [21, 4, 70]  # [edad, horas_estudio, asistencia]
    print("\nProbando predicción...")
    print("Nuevo estudiante:")
    print(f"- Edad: {nuevo_estudiante[0]} años")
    print(f"- Horas de estudio: {nuevo_estudiante[1]} horas")
    print(f"- Asistencia: {nuevo_estudiante[2]}%")
    
    resultado = predecir_abandono(modelo, nuevo_estudiante)
    print(f"\nPredicción: {resultado}")