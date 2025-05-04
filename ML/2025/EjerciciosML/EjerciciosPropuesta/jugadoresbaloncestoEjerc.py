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
        # TODO: Implementar este método para retornar una lista con los atributos numéricos del jugador
        # Pista: ¿Qué atributos son numéricos y deberían usarse para el modelo?
        return [self.height, self.weight, self.avg_points] # Placeholder - ¡Los alumnos deben implementarlo!

class BasketballDataGenerator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def generate(self):
        heights = np.random.normal(190, 10, self.num_samples)    # Altura media de 190 cm
        weights = np.random.normal(85, 10, self.num_samples)     # Peso medio de 85 kg
        points = np.random.normal(10, 5, self.num_samples)       # Puntos por partido, media 10

        data = []
        for h, w, p in zip(heights, weights, points):
            # TODO: Los alumnos deben completar la lógica para asignar la categoría de rendimiento
            # basada en el promedio de puntos (p).
            # Pista: Usar condicionales (if, elif, else) para definir "Bajo", "Medio" y "Alto".
            perf = "" # Placeholder - ¡Los alumnos deben asignar el valor correcto!
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
        # TODO: Inicializar el modelo de árbol de decisión de scikit-learn
        # Pista: ¿Qué clase de sklearn.tree necesitamos usar para la clasificación?
        self.model = DecisionTreeClassifier() # Placeholder - ¡Los alumnos deben instanciar el clasificador!

    def fit(self, players):
        # TODO: Implementar el método para entrenar el modelo
        # Pista: Necesitas separar las características (X) de las etiquetas (y) de los jugadores.
        #       Usa el método to_vector() de cada jugador para obtener las características.
        X = [p.to_vector() for p in players] # Placeholder - ¡Los alumnos podrían necesitar ajustar esto!
        y = [p.performance for p in players] # Placeholder - ¡Los alumnos podrían necesitar ajustar esto!
        self.model.fit(X, y)

    def predict(self, height, weight, avg_points):
        # TODO: Implementar la predicción para un nuevo jugador
        # Pista: El método predict del modelo espera una estructura de datos específica.
        return self.model.predict([[height, weight, avg_points]])[0] # Placeholder - ¡Los alumnos deben asegurarse del formato!

    def evaluate(self, players):
        # TODO: Implementar la evaluación del modelo
        # Pista: Similar al fit, necesitas separar características y etiquetas para el conjunto de prueba.
        #        Luego, usa el modelo para hacer predicciones y compara con las etiquetas reales.
        #        Finalmente, imprime la matriz de confusión y el informe de clasificación.
        X = [p.to_vector() for p in players] # Placeholder
        y = [p.performance for p in players] # Placeholder
        y_pred = self.model.predict(X) # Placeholder
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred)) # Placeholder
        print("\nClassification Report:")
        print(classification_report(y, y_pred)) # Placeholder

class BasketballPredictionExample:
    def run(self):
        # 1. Generar datos
        generator = BasketballDataGenerator()
        data = generator.generate()

        # 2. Dividir los datos en conjuntos de entrenamiento y prueba
        # TODO: Los alumnos deben realizar la división de datos usando train_test_split
        # Pista: ¿Qué proporción de los datos debería usarse para la prueba? ¿Es importante el random_state?
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=1) # Placeholder - ¡Los alumnos deben hacerlo!

        # 3. Entrenar el clasificador
        classifier = BasketballPerformanceClassifier()
        # TODO: Los alumnos deben llamar al método para entrenar el clasificador con los datos de entrenamiento
        classifier.fit(train_data) # Placeholder - ¡Los alumnos deben llamar a la función correcta!

        # 4. Evaluar el clasificador
        # TODO: Los alumnos deben llamar al método para evaluar el clasificador con los datos de prueba
        classifier.evaluate(test_data) # Placeholder - ¡Los alumnos deben llamar a la función correcta!

        # 5. Ejemplo de predicción personalizada
        height, weight, points = 198, 92, 17
        prediction = classifier.predict(height, weight, points)
        print(f"\n🎯 Predicción personalizada → Altura: {height} cm, Peso: {weight} kg, Prom. puntos: {points}")
        print(f"   → Categoría predicha: {prediction}")

        # 6. Visualización con Matplotlib
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