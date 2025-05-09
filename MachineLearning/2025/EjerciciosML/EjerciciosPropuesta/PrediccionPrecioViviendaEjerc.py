import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def generar_datos(num_viviendas=200):
    """
    TODO: Genera datos sintéticos de viviendas
    Hint: Usa np.random para crear características realistas
    """
    np.random.seed(42)  # Para reproducibilidad
    
    # TODO: Crea un DataFrame con estas características:
    # - Superficie (50-300 m²)
    # - Num_Habitaciones (1-5)
    # - Antigüedad (0-50 años)
    # - Distancia_Centro (1-20 km)
    # - Num_Baños (1-3)
    # - Precio (10000-500000)
    data = pd.DataFrame({
        'Superficie': None,  # Usa np.random.randint()
        'Num_Habitaciones': None,
        'Antigüedad': None,
        'Distancia_Centro': None,  # Usa np.random.uniform()
        'Num_Baños': None,
        'Precio': None
    })
    
    return data


def entrenar_modelo(data):
    """
    TODO: Entrena un modelo de regresión lineal
    Hint: Usa LinearRegression() de sklearn
    """
    # TODO: Separa features (X) y target (y)
    # Hint: X son todas las columnas excepto 'Precio'
    X = None
    y = None
    
    # TODO: Divide los datos en train y test
    # Hint: Usa train_test_split con test_size=0.2
    
    # TODO: Crea y entrena el modelo
    # Hint: Usa fit() con los datos de entrenamiento
    modelo = None
    
    # TODO: Evalúa el modelo
    # Hint: Calcula MSE y R² con los datos de prueba
    
    return modelo


def predecir_precio(modelo, caracteristicas):
    """
    TODO: Realiza una predicción para una nueva vivienda
    Hint: Usa predict() del modelo entrenado
    """
    # TODO: Asegúrate que las características tienen el formato correcto
    # TODO: Realiza la predicción
    return precio_estimado


# Código de prueba
if __name__ == "__main__":
    # Generar datos
    print("Generando datos de viviendas...")
    datos = generar_datos(200)
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    modelo = entrenar_modelo(datos)
    
    # Probar con nueva vivienda
    nueva_vivienda = [
        120,  # Superficie (m²)
        3,    # Habitaciones
        10,   # Antigüedad (años)
        5,    # Distancia al centro (km)
        2     # Baños
    ]
    
    # Predecir precio
    precio = predecir_precio(modelo, nueva_vivienda)
    print("\nPredicción para nueva vivienda:")
    print(f"- Superficie: {nueva_vivienda[0]} m²")
    print(f"- Habitaciones: {nueva_vivienda[1]}")
    print(f"- Antigüedad: {nueva_vivienda[2]} años")
    print(f"- Distancia: {nueva_vivienda[3]} km")
    print(f"- Baños: {nueva_vivienda[4]}")
    print(f"\nPrecio estimado: {precio:.2f}")