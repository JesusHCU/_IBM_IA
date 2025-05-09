import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class EnergyRecord:
    """Clase para almacenar registros de energía"""
    def __init__(self, temperature, consumption):
        # TODO: Guarda los valores de temperatura y consumo
        # Hint: Guarda temperature en self.temperature y consumption en self.consumption
        pass

    def to_vector(self):
        # TODO: Retorna la temperatura como una lista
        # Hint: Retorna [self.temperature]
        pass


# Esta clase ya está implementada para ti - Genera datos de ejemplo
class EnergyDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        temperatures = np.random.uniform(-5, 35, self.num_samples)
        consumption = 100 + (np.abs(temperatures - 20) * 3) + np.random.normal(0, 5, self.num_samples)
        return [EnergyRecord(t, c) for t, c in zip(temperatures, consumption)]


class EnergyRegressor:
    """Clase para predecir consumo energético"""
    def __init__(self):
        # El modelo ya está inicializado para ti
        self.model = LinearRegression()

    def fit(self, records):
        # TODO: Entrena el modelo con los datos
        # Hint: Usa self.model.fit(X, y) donde:
        # - X es array de temperaturas (usar to_vector())
        # - y es array de consumos
        pass

    def predict(self, temperature):
        # TODO: Predice el consumo para una temperatura
        # Hint: Usa self.model.predict([[temperature]])
        pass


# Esta clase ya está implementada para ti - Ejecuta el ejemplo completo
class EnergyPredictionExample:
    def run(self):
        # Generar datos
        generator = EnergyDataGenerator(100)
        data = generator.generate()

        # Entrenar modelo
        regressor = EnergyRegressor()
        regressor.fit(data)

        # Predecir y mostrar resultados
        test_temp = 25
        prediction = regressor.predict(test_temp)
        print(f"Para {test_temp}°C, consumo predicho: {prediction:.2f} kWh")

        # Visualizar resultados
        self.plot_results(data, regressor)

    def plot_results(self, data, regressor):
        temps = [r.temperature for r in data]
        consumo = [r.consumption for r in data]

        plt.figure(figsize=(10, 6))
        plt.scatter(temps, consumo, color='blue', label='Datos', alpha=0.5)
        
        x = np.linspace(-5, 35, 100).reshape(-1, 1)
        y = regressor.predict(x)
        
        plt.plot(x, y, color='red', label='Predicción')
        plt.xlabel('Temperatura (°C)')
        plt.ylabel('Consumo (kWh)')
        plt.title('Predicción de Consumo Energético')
        plt.legend()
        plt.grid(True)
        plt.show()


# Código de prueba
if __name__ == "__main__":
    example = EnergyPredictionExample()
    example.run()