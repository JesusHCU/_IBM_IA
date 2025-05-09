import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

class BasketballPlayer:
    def __init__(self, height, weight, avg_points, performance):
        self.height = height          # Altura en centímetros
        self.weight = weight          # Peso en kilogramos
        self.avg_points = avg_points  # Promedio de puntos por partido
        self.performance = performance  # Rendimiento: Bajo, Medio o Alto

    def to_vector(self):
        return [self.height, self.weight, self.avg_points]

class BasketballDataGenerator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def generate(self):
        heights = np.random.normal(190, 10, self.num_samples)    # Altura media de 190 cm
        weights = np.random.normal(85, 10, self.num_samples)     # Peso medio de 85 kg
        points = np.random.normal(10, 5, self.num_samples)       # Puntos por partido, media 10

        data = []
        for h, w, p in zip(heights, weights, points):
            if p < 8:
                perf = "Bajo"
            elif p < 15:
                perf = "Medio"
            else:
                perf = "Alto"
            data.append(BasketballPlayer(h, w, max(0, p), perf))  # Evita puntos negativos

        return data

class BasketballPerformanceClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, players):
        X = [p.to_vector() for p in players]
        y = [p.performance for p in players]
        self.model.fit(X, y)

    def predict(self, height, weight, avg_points):
        return self.model.predict([[height, weight, avg_points]])[0]

    def evaluate(self, players):
        X = [p.to_vector() for p in players]
        y = [p.performance for p in players]
        y_pred = self.model.predict(X)

        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

class BasketballPredictionExample:
    def run(self):
        generator = BasketballDataGenerator()
        data = generator.generate()

        train_data, test_data = train_test_split(data, test_size=0.3, random_state=1)

        classifier = BasketballPerformanceClassifier()
        classifier.fit(train_data)
        classifier.evaluate(test_data)

        # Ejemplo de predicción personalizada
        height, weight, points = 198, 92, 17
        prediction = classifier.predict(height, weight, points)
        print(f"\n🎯 Predicción personalizada → Altura: {height} cm, Peso: {weight} kg, Prom. puntos: {points}")
        print(f"   → Categoría predicha: {prediction}")

        # Visualización con Matplotlib
        df = pd.DataFrame({
            "Altura": [p.height for p in data],
            "Prom. Puntos": [p.avg_points for p in data],
            "Rendimiento": [p.performance for p in data]
        })

        colores = {
            "Bajo": "red",
            "Medio": "orange",
            "Alto": "green"
        }

        plt.figure(figsize=(8, 6))
        for nivel, color in colores.items():
            subset = df[df["Rendimiento"] == nivel]
            plt.scatter(subset["Altura"], subset["Prom. Puntos"], label=nivel, c=color, alpha=0.6)

        plt.xlabel("Altura (cm)")
        plt.ylabel("Promedio de puntos por partido")
        plt.title("🏀 Clasificación de jugadores de baloncesto por rendimiento")
        plt.grid(True)
        plt.legend(title="Rendimiento")
        plt.show()

# Ejecución final
example = BasketballPredictionExample()
example.run()