import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def graficar_datos_estudio(horas, calificaciones):
    """
    TODO: Implementa la función para graficar la relación entre horas de estudio y calificaciones
    Parámetros:
    - horas: lista de horas de estudio
    - calificaciones: lista de calificaciones obtenidas
    """
    # TODO: Configura el tamaño de la figura
    # Hint: Usa plt.figure(figsize=(8, 5))
    
    # TODO: Crea el gráfico de dispersión
    # Hint: Usa plt.plot() con marker='o'
    
    # TODO: Añade etiquetas y título descriptivos
    # Hint: Usa plt.xlabel(), plt.ylabel() y plt.title()
    
    # TODO: Configura la cuadrícula
    # Hint: Usa plt.grid() con alpha=0.6
    
    pass


def analizar_rendimiento(horas, calificaciones):
    """
    TODO: Implementa el análisis de rendimiento
    Hint: Calcula estadísticas básicas y correlación
    """
    # TODO: Convierte las listas a arrays de numpy
    # Hint: Usa np.array()
    
    # TODO: Calcula:
    # 1. Promedio de horas de estudio
    # 2. Promedio de calificaciones
    # 3. Calificación máxima y mínima
    # 4. Correlación entre horas y calificaciones
    
    return {
        "promedio_horas": 0,  # Reemplaza con el cálculo real
        "promedio_calificaciones": 0,  # Reemplaza con el cálculo real
        "max_calificacion": 0,  # Reemplaza con el cálculo real
        "min_calificacion": 0,  # Reemplaza con el cálculo real
        "correlacion": 0  # Reemplaza con el cálculo real
    }


def predecir_calificacion(horas_estudio, modelo):
    """
    TODO: Implementa la predicción de calificaciones
    Hint: Usa el modelo de regresión lineal
    """
    # TODO: Usa el modelo para predecir la calificación
    # Hint: Asegúrate de que horas_estudio tenga el formato correcto
    pass


# Código de prueba
if __name__ == "__main__":
    # Datos de ejemplo
    horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8]
    calificaciones = [55, 60, 65, 70, 75, 80, 85, 90]

    # TODO: 1. Grafica los datos
    graficar_datos_estudio(horas_estudio, calificaciones)
    
    # TODO: 2. Analiza el rendimiento
    resultados = analizar_rendimiento(horas_estudio, calificaciones)
    print("\nAnálisis de Rendimiento:")
    for key, value in resultados.items():
        print(f"{key}: {value}")
    
    # TODO: 3. Crea y entrena un modelo de regresión lineal
    # Hint: Usa LinearRegression() de sklearn
    
    # TODO: 4. Predice la calificación para 10 horas de estudio
    # nueva_prediccion = predecir_calificacion(10, modelo)
    # print(f"\nPredicción para 10 horas de estudio: {nueva_prediccion:.2f}")