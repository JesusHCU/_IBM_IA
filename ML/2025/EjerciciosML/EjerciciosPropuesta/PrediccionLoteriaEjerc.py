import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generar_series(num_series):
    """
    TODO: Genera combinaciones de lotería
    Hint: Genera 'num_series' combinaciones de 6 números entre 1 y 49
    """
    # TODO: Crea una lista para guardar las combinaciones
    series = []
    
    # TODO: Genera cada combinación
    # Hint: Usa np.random.choice(range(1, 50), size=6, replace=False)
    
    # TODO: Retorna las series como un array de NumPy
    return np.array(series)


# Esta función ya está implementada para ti
def simular_resultados(num_series):
    """Simula resultados de lotería (éxito/fracaso)"""
    return [1 if np.random.rand() < 0.1 else 0 for _ in range(num_series)]


def entrenar_modelo():
    """
    TODO: Entrena un modelo para predecir combinaciones ganadoras
    Hint: Usa RandomForestClassifier
    """
    # TODO: Genera datos de entrenamiento
    # Hint: Usa generar_series() y simular_resultados()
    
    # TODO: Crea un DataFrame con las combinaciones
    # Hint: Usa pd.DataFrame con columnas ['num1'...'num6']
    
    # TODO: Divide los datos en train y test
    # Hint: Usa train_test_split
    
    # TODO: Crea y entrena el modelo
    # Hint: Usa RandomForestClassifier(random_state=42)
    
    return modelo


def predecir_mejor_serie(modelo, num_series):
    """
    TODO: Predice la mejor combinación entre varias nuevas
    Hint: Usa el modelo para predecir probabilidades
    """
    # TODO: Genera nuevas combinaciones
    # Hint: Usa generar_series()
    
    # TODO: Obtén las probabilidades de éxito
    # Hint: Usa modelo.predict_proba()
    
    # TODO: Encuentra la mejor combinación
    # Hint: Usa np.argmax()
    
    return mejor_combinacion, mejor_probabilidad


# Código de prueba
if __name__ == "__main__":
    print("Entrenando modelo de predicción...")
    modelo = entrenar_modelo()
    
    print("\nBuscando mejor combinación...")
    mejor_combinacion, probabilidad = predecir_mejor_serie(modelo, 100)
    
    print("\nResultados:")
    print(f"Números sugeridos: {mejor_combinacion}")
    print(f"Probabilidad de éxito: {probabilidad:.2%}")