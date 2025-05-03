import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class WeatherRecord:
    """Clase para almacenar registros meteorológicos"""
    def __init__(self, humidity, pressure, will_rain):
        # TODO: Guarda los atributos del registro meteorológico
        # Hint: Almacena humidity, pressure y will_rain
        pass

    def to_vector(self):
        # TODO: Retorna una lista con humedad y presión
        # Hint: return [self.humidity, self.pressure]
        pass


# Esta clase ya está implementada para ti
class WeatherDataGenerator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def generate(self):
        humidity = np.random.uniform(30, 100, self.num_samples)
        pressure = np.random.uniform(980, 1050, self.num_samples)
        
        # Cálculo de probabilidad de lluvia
        rain_prob = (humidity - 50) * 0.03 - (pressure - 1010) * 0.02
        rain_prob = 1 / (1 + np.exp(-rain_prob))
        rain = (rain_prob > 0.5).astype(int)
        
        return [WeatherRecord(h, p, r) for h, p, r in zip(humidity, pressure, rain)]


class WeatherRainClassifier:
    """Clasificador de lluvia basado en condiciones meteorológicas"""
    def __init__(self):
        # Ya inicializado para ti
        self.model = LogisticRegression()

    def fit(self, records):
        # TODO: Entrena el modelo con los registros
        # Hint: Usa self.model.fit(X, y)
        # X debe ser matriz de [humedad, presión]
        # y debe ser vector de predicciones (0/1)
        pass

    def predict(self, humidity, pressure):
        # TODO: Predice si lloverá dada humedad y presión
        # Hint: Usa self.model.predict([[humidity, pressure]])
        pass


def visualizar_predicciones(registros, predicciones):
    """
    TODO: Crea una visualización de los datos
    Hint: Usa plt.scatter para crear un gráfico de dispersión
    """
    # TODO: Crea un DataFrame con los datos
    # TODO: Crea el gráfico de dispersión
    # TODO: Añade título y etiquetas
    pass


# Código de ejemplo
if __name__ == "__main__":
    # Generar datos
    print("Generando datos meteorológicos...")
    generator = WeatherDataGenerator(num_samples=200)
    registros = generator.generate()
    
    # Dividir datos
    train_data, test_data = train_test_split(registros, test_size=0.3, random_state=42)
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    clasificador = WeatherRainClassifier()
    clasificador.fit(train_data)
    
    # Probar predicción
    humedad_prueba = 75
    presion_prueba = 1000
    
    prediccion = clasificador.predict(humedad_prueba, presion_prueba)
    print(f"\nPredicción para:")
    print(f"- Humedad: {humedad_prueba}%")
    print(f"- Presión: {presion_prueba} hPa")
    print(f"¿Lloverá? {'Sí' if prediccion == 1 else 'No'}")