import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class SimuladorViviendas:
    """Simula un conjunto de viviendas con características y precios."""

    def __init__(self, n=200, seed=42):
        self.n = n
        self.seed = seed

    def generar_datos(self):
        """Genera datos sintéticos de viviendas con características relevantes."""
        np.random.seed(self.seed)
        data = pd.DataFrame({
            'Superficie': np.random.uniform(50, 150, self.n),
            'Habitaciones': np.random.randint(1, 6, self.n),
            'Antigüedad': np.random.randint(0, 50, self.n),
            'Distancia_centro': np.random.uniform(1, 20, self.n),
            'Baños': np.random.randint(1, 4, self.n),
            'Precio': np.random.uniform(100000, 500000, self.n)
        })
        return data


class ModeloPrecioVivienda:
    """Modelo de regresión para predecir el precio de una vivienda."""

    def __init__(self):
        self.modelo = LinearRegression()

    def entrenar(self, data):
        """Entrena el modelo de regresión lineal con los datos proporcionados."""
        self.X = data.drop(columns=['Precio'])
        self.y = data['Precio']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.modelo.fit(self.X_train, self.y_train)
        print("Modelo entrenado correctamente.")

    def evaluar(self):
        """Evalúa el modelo y muestra métricas de rendimiento."""
        y_pred = self.modelo.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"\nError Cuadrático Medio (MSE): {mse:,.2f}")
        print(f"Coeficiente de determinación R²: {r2:.2f}")

    def predecir(self, nueva_vivienda: pd.DataFrame) -> float:
        """Predice el precio de una nueva vivienda."""
        return self.modelo.predict(nueva_vivienda)[0]


class TestModeloPrecio:
    """Clase de prueba para verificar el funcionamiento del modelo."""

    def ejecutar(self):
        """Ejecuta todo el flujo de trabajo: simula datos, entrena, evalúa y predice."""
        # Simular datos
        simulador = SimuladorViviendas()
        datos = simulador.generar_datos()
        print("Primeras filas de datos simulados:")
        print(datos.head())

        # Entrenar y evaluar modelo
        modelo = ModeloPrecioVivienda()
        modelo.entrenar(datos)
        modelo.evaluar()

        # Nueva vivienda para predecir
        nueva_vivienda = pd.DataFrame({
            'Superficie': [120],
            'Habitaciones': [3],
            'Antigüedad': [10],
            'Distancia_centro': [5],
            'Baños': [2]
        })
        prediccion = modelo.predecir(nueva_vivienda)
        print(f"\nEl precio estimado de la vivienda es: ${prediccion:,.2f}")


# Ejecución del test
if __name__ == "__main__":
    test = TestModeloPrecio()
    test.ejecutar()