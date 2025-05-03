import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Generación de datos ficticios
def generar_datos():
    np.random.seed(42)
    num_viviendas = 200
    data = pd.DataFrame({
        'Superficie': np.random.randint(50, 300, size=num_viviendas),  # Área en metros cuadrados
        'Num_Habitaciones': np.random.randint(1, 6, size=num_viviendas),  # Cantidad de habitaciones
        'Antigüedad': np.random.randint(0, 50, size=num_viviendas),  # Años desde construcción
        'Distancia_Centro': np.random.uniform(1, 20, size=num_viviendas),  # Distancia en kilómetros
        'Num_Baños': np.random.randint(1, 3, size=num_viviendas),  # Cantidad de baños
        'Precio': np.random.randint(10000, 500000, size=num_viviendas)  # Precio en la moneda local
    })
    return data


def entrenar_modelo(data):
    # Separación de características y variable objetivo
    X = data.drop(columns=['Precio'])
    y = data['Precio']

    # División de los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creación del modelo de regresión lineal
    modelo = LinearRegression()

    # Entrenamiento del modelo
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Evaluación del modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Coeficiente de determinación (R²): {r2:.2f}")

    return modelo


# Generar datos de ejemplo
data = generar_datos()

# Entrenar el modelo y evaluar
modelo_entrenado = entrenar_modelo(data)

# Ejemplo de predicción con nuevas viviendas
nueva_vivienda = np.array([[120, 3, 10, 5, 2]])  # Características de una nueva vivienda
precio_estimado = modelo_entrenado.predict(nueva_vivienda)

print(f"El precio estimado de la nueva vivienda es: {precio_estimado[0]:.2f} unidades monetarias.")

