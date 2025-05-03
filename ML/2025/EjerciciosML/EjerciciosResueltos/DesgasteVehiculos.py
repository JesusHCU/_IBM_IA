import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class VehicleRecord:
    def __init__(self, hours_used, wear_level):
        self.hours_used = hours_used
        self.wear_level = wear_level

    def to_vector(self):
        return [self.hours_used]


class VehicleDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        hours = np.random.uniform(50, 500, self.num_samples)
        wear = 10 + 0.18 * hours + np.random.normal(0, 5, self.num_samples)
        wear = np.clip(wear, 0, 100)
        data = [VehicleRecord(h, w) for h, w in zip(hours, wear)]
        return data


class VehicleWearRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, records):
        X = np.array([r.to_vector() for r in records])
        y = np.array([r.wear_level for r in records])
        self.model.fit(X, y)

    def predict(self, hours):
        return self.model.predict([[hours]])[0]

    def get_model(self):
        return self.model


class VehicleWearPredictionExample:
    def run(self):
        generator = VehicleDataGenerator(100)
        records = generator.generate()
        regressor = VehicleWearRegressor()
        regressor.fit(records)
        test_hours = 250
        prediction = regressor.predict(test_hours)
        print(f"⏱ Horas de uso estimadas: {test_hours}")
        print(f"⚙️ Nivel de desgaste estimado: {prediction:.2f}%")
        df = pd.DataFrame({
            "Horas de uso": [r.hours_used for r in records],
            "Nivel de desgaste": [r.wear_level for r in records]
        })
        plt.figure(figsize=(8, 5))
        plt.scatter(df["Horas de uso"], df["Nivel de desgaste"])
        x_line = np.linspace(40, 520, 100).reshape(-1, 1)
        y_line = regressor.get_model().predict(x_line)
        plt.plot(x_line, y_line, color='red')
        plt.axvline(test_hours, color='green', linestyle='--')
        plt.title('Predicción del Desgaste de Vehículos Militares')
        plt.xlabel('Horas de Uso')
        plt.ylabel('Nivel de Desgaste (%)')
        plt.legend(['Línea de Regresión', 'Predicción', 'Datos Reales'])
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    example = VehicleWearPredictionExample()
    example.run()