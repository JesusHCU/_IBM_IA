import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class WeatherRecord:
    def __init__(self, humidity, pressure, will_rain):
        self.humidity = humidity
        self.pressure = pressure
        self.will_rain = will_rain  # 0 = no lloverá, 1 = lloverá

    def to_vector(self):
        return [self.humidity, self.pressure]


class WeatherDataGenerator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def generate(self):
        humidity = np.random.uniform(30, 100, self.num_samples)
        pressure = np.random.uniform(980, 1050, self.num_samples)

        # Modelo probabilístico con función sigmoide
        rain_prob = (humidity - 50) * 0.03 - (pressure - 1010) * 0.02
        rain_prob = 1 / (1 + np.exp(-rain_prob))

        rain = (rain_prob > 0.5).astype(int)

        data = [WeatherRecord(h, p, r) for h, p, r in zip(humidity, pressure, rain)]
        return data


class WeatherRainClassifier:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, records):
        X = np.array([r.to_vector() for r in records])
        y = np.array([r.will_rain for r in records])
        self.model.fit(X, y)

    def predict(self, humidity, pressure):
        return self.model.predict([[humidity, pressure]])[0]

    def evaluate(self, records):
        X = np.array([r.to_vector() for r in records])
        y = np.array([r.will_rain for r in records])
        y_pred = self.model.predict(X)
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))


class WeatherRainPredictionExample:
    def run(self):
        generator = WeatherDataGenerator()
        records = generator.generate()

        train_records, test_records = train_test_split(records, test_size=0.3, random_state=42)

        classifier = WeatherRainClassifier()
        classifier.fit(train_records)
        classifier.evaluate(test_records)

        # PREDICCIÓN NUEVA
        humidity_test = 80
        pressure_test = 995
        prediction = classifier.predict(humidity_test, pressure_test)

        print("🔍 Predicción para condiciones nuevas:")
        print(f"   Humedad: {humidity_test}%")
        print(f"   Presión: {pressure_test} hPa")
        print(f"   ¿Lloverá?: {'Sí ☔' if prediction == 1 else 'No ☀️'}")

        # Visualización de datos
        df = pd.DataFrame({
            "Humedad": [r.humidity for r in test_records],
            "Presión": [r.pressure for r in test_records],
            "Lluvia": [r.will_rain for r in test_records]
        })

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df["Humedad"], df["Presión"], c=df["Lluvia"], cmap="bwr", alpha=0.6)
        plt.xlabel("Humedad relativa (%)")
        plt.ylabel("Presión atmosférica (hPa)")
        plt.title("🌧️ Predicción de lluvia según condiciones meteorológicas")
        plt.grid(True)
        plt.colorbar(scatter, label="0 = No llueve, 1 = Llueve")
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    example = WeatherRainPredictionExample()
    example.run()