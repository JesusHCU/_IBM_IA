import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class Snack:
    """
    Clase que representa un snack con sus características nutricionales.
    TODO: Completa la clase con los atributos necesarios
    """
    def __init__(self, calories, sugar, protein, fat, fiber, is_healthy=None):
        # TODO: Inicializa los atributos del snack
        # Hint: Guarda cada parámetro como un atributo de la clase
        pass

    def to_vector(self):
        """
        TODO: Retorna una lista con las características del snack
        Hint: Debe retornar [calories, sugar, protein, fat, fiber]
        """
        pass


class SnackGenerator:
    """
    Clase para generar datos de entrenamiento.
    Ya está implementada para ti, ¡estúdiala para entender cómo funciona!
    """
    def __init__(self, num_snacks=100):
        self.num_snacks = num_snacks

    def generate(self):
        """
        Genera snacks de ejemplo con valores aleatorios.
        Reglas para snack saludable:
        - Calorías < 300
        - Azúcar < 20g
        - Grasa < 10g
        - Proteína >= 10g O Fibra >= 10g
        """
        snacks = []
        # TODO: Implementa la generación de snacks
        # Hint: Usa np.random.randint() para generar valores aleatorios
        # Hint: Rangos sugeridos:
        #   - Calorías: 50-400
        #   - Azúcar: 0-30g
        #   - Proteína: 0-20g
        #   - Grasa: 0-25g
        #   - Fibra: 0-10g
        return snacks


class SnackClassifier:
    """
    TODO: Implementa el clasificador usando DecisionTreeClassifier
    """
    def __init__(self):
        # TODO: Inicializa el modelo
        # Hint: Usa DecisionTreeClassifier(random_state=42)
        pass

    def fit(self, snacks):
        """
        TODO: Entrena el modelo con los snacks proporcionados
        Hint: Necesitas extraer características (X) y etiquetas (y)
        """
        pass

    def predict(self, snack):
        """
        TODO: Implementa la predicción para un nuevo snack
        Hint: Usa el método predict del modelo
        """
        pass


def main():
    """
    Función principal para probar tu implementación
    """
    # Crear y generar datos
    generator = SnackGenerator(num_snacks=150)
    snacks = generator.generate()
    print(f"Generados {len(snacks)} snacks para entrenamiento")

    # TODO: Crear y entrenar el clasificador
    
    # TODO: Crear un snack de prueba
    # Ejemplo: calories=150, sugar=10, protein=6, fat=5, fiber=3
    
    # TODO: Realizar y mostrar la predicción

if __name__ == "__main__":
    main()