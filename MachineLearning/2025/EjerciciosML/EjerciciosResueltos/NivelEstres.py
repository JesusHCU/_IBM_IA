import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class Individual:
    def __init__(self, heart_rate, cortisol_level, skin_conductance, stress_level):
        self.heart_rate = heart_rate
        self.cortisol_level = cortisol_level
        self.skin_conductance = skin_conductance
        self.stress_level = stress_level

    def to_vector(self):
        return [self.heart_rate, self.cortisol_level, self.skin_conductance]

class StressDataGenerator:
    def __init__(self, n=300):
        self.n = n

    def generate(self):
        heart_rates = np.random.normal(75, 15, self.n)
        cortisol = np.random.normal(12, 4, self.n)
        conductance = np.random.normal(5, 1.5, self.n)

        data = []
        for hr, cort, sc in zip(heart_rates, cortisol, conductance):
            if hr > 90 or cort > 18 or sc > 6.5:
                level = "Alto"
            elif hr > 70 or cort > 10 or sc > 4.5:
                level = "Moderado"
            else:
                level = "Bajo"
            data.append(Individual(hr, cort, sc, level))
        return data

class StressClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=1)

    def fit(self, individuals):
        X = [i.to_vector() for i in individuals]
        y = [i.stress_level for i in individuals]
        self.model.fit(X, y)

    def predict(self, heart_rate, cortisol, conductance):
        return self.model.predict([[heart_rate, cortisol, conductance]])[0]

    def evaluate(self, test_data):
        X = [i.to_vector() for i in test_data]
        y_true = [i.stress_level for i in test_data]
        y_pred = self.model.predict(X)

        print("📊 Matriz de confusión:")
        print(confusion_matrix(y_true, y_pred))
        print("\n📝 Informe de clasificación:")
        print(classification_report(y_true, y_pred))

class StressAnalysisExample:
    def run(self):
        generator = StressDataGenerator()
        data = generator.generate()

        train, test = train_test_split(data, test_size=0.3, random_state=1)

        classifier = StressClassifier()
        classifier.fit(train)
        classifier.evaluate(test)

        # Predicción personalizada
        hr, cort, sc = 95, 20, 7
        pred = classifier.predict(hr, cort, sc)
        print(f"\n🧠 Predicción para individuo personalizado:")
        print(f"  Ritmo cardíaco: {hr}, Cortisol: {cort}, Conductancia: {sc}")
        print(f"  → Nivel estimado de estrés: {pred}")

        # Visualización
        df = pd.DataFrame({
            "Ritmo cardíaco": [i.heart_rate for i in data],
            "Cortisol": [i.cortisol_level for i in data],
            "Conductancia": [i.skin_conductance for i in data],
            "Estrés": [i.stress_level for i in data]
        })

        colores = {"Bajo": "green", "Moderado": "orange", "Alto": "red"}
        plt.figure(figsize=(8, 6))
        for nivel, color in colores.items():
            subset = df[df["Estrés"] == nivel]
            plt.scatter(subset["Cortisol"], subset["Ritmo cardíaco"], label=nivel, c=color, alpha=0.6)

        plt.xlabel("Nivel de cortisol (µg/dL)")
        plt.ylabel("Ritmo cardíaco (ppm)")
        plt.title("🧬 Clasificación de nivel de estrés fisiológico")
        plt.grid(True)
        plt.legend(title="Nivel de estrés")
        plt.show()

# Ejecutar ejemplo
example = StressAnalysisExample()
example.run()