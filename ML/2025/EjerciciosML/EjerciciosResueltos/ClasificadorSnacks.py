import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class Snack:
    """
    Clase que representa un snack con sus características nutricionales.
    """

    def __init__(self, calories, sugar, protein, fat, fiber, is_healthy=None):
        self.calories = calories
        self.sugar = sugar
        self.protein = protein
        self.fat = fat
        self.fiber = fiber
        self.is_healthy = is_healthy

    def to_vector(self):
        """
        Convierte las características del snack en un vector.
        """
        return [self.calories, self.sugar, self.protein, self.fat, self.fiber]

    def __str__(self):
        """
        Representación en string del snack.
        """
        return f"Calories: {self.calories}, Sugar: {self.sugar}g, Protein: {self.protein}g, Fat: {self.fat}g, Fiber: {self.fiber}g"


class SnackGenerator:
    """
    Clase para generar datos sintéticos de snacks.
    """

    def __init__(self, num_snacks=None):
        if num_snacks is None:
            # Generar entre 50 y 200 snacks si no se especifica
            self.num_snacks = np.random.randint(50, 201)
        else:
            self.num_snacks = num_snacks

    def generate(self):
        """
        Genera un conjunto de snacks con valores aleatorios.
        """
        snacks = []
        for _ in range(self.num_snacks):
            # Generar valores aleatorios para cada característica
            calories = np.random.randint(50, 400)
            sugar = np.random.randint(0, 30)
            protein = np.random.randint(0, 20)
            fat = np.random.randint(0, 25)
            fiber = np.random.randint(0, 10)

            # Determinar si es saludable según reglas aproximadas
            is_healthy = 1 if (calories < 300 and sugar < 20 and fat < 10 and
                               (protein >= 10 or fiber >= 10)) else 0

            # Crear y añadir el snack a la lista
            snack = Snack(calories, sugar, protein, fat, fiber, is_healthy)
            snacks.append(snack)

        return snacks


class SnackClassifier:
    """
    Clasificador de snacks utilizando árbol de decisión.
    """

    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.is_trained = False

    def fit(self, snacks):
        """
        Entrena el modelo con un conjunto de snacks.
        """
        # Extraer características y etiquetas
        X = [snack.to_vector() for snack in snacks]
        y = [snack.is_healthy for snack in snacks]

        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Entrenar el modelo
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluar el modelo (opcional, para información)
        score = self.model.score(X_test, y_test)
        print(f"📊 Precisión del modelo: {score:.2f}")

    def predict(self, snack):
        """
        Predice si un snack es saludable o no.
        """
        if not self.is_trained:
            raise Exception("El modelo debe ser entrenado antes de hacer predicciones")

        # Obtener el vector de características del snack
        features = [snack.to_vector()]

        # Hacer la predicción
        prediction = self.model.predict(features)[0]

        return prediction


class SnackRecommendationExample:
    """
    Ejemplo de uso del sistema de recomendación de snacks.
    """

    def __init__(self):
        # Inicializar el generador y el clasificador
        self.generator = SnackGenerator()
        self.classifier = SnackClassifier()

    def run(self):
        # Generar snacks
        snacks = self.generator.generate()
        print(f"🍎 Generados {len(snacks)} snacks para entrenamiento")

        # Entrenar el clasificador
        self.classifier.fit(snacks)

        # Crear un snack de prueba
        test_snack = Snack(150, 10, 6, 5, 3)

        # Predecir si es saludable
        prediction = self.classifier.predict(test_snack)

        # Mostrar resultados
        print("\n🔍 Snack Info:")
        print(test_snack)

        if prediction == 1:
            print("✅ Predicción: Este snack es saludable.")
        else:
            print("✅ Predicción: Este snack no es saludable.")


# Ejecutar ejemplo
if __name__ == "__main__":
    example = SnackRecommendationExample()
    example.run()