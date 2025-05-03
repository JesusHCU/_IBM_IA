import numpy as np
from sklearn.ensemble import RandomForestClassifier


def generate_dataset(n_samples=100):
    """
    Genera un conjunto de datos artificial para el problema de recomendación
    de lenguajes de programación.

    Características:
    - velocidad_requerida: Float entre 0-1 (0: baja, 1: alta)
    - facilidad_mantenimiento: Float entre 0-1 (0: difícil, 1: fácil)
    - disponibilidad_bibliotecas: Float entre 0-1 (0: pocas, 1: muchas)
    - tipo_app: Int (0: web, 1: móvil, 2: sistemas)
    - rendimiento_deseado: Float entre 0-1 (0: bajo, 1: alto)

    Etiquetas:
    - 0: Python
    - 1: JavaScript
    - 2: Java
    - 3: C++

    Returns:
        X: np.ndarray - Matriz de características
        y: np.ndarray - Vector de etiquetas
    """
    # Generar características aleatorias
    velocidad_requerida = np.random.rand(n_samples)
    facilidad_mantenimiento = np.random.rand(n_samples)
    disponibilidad_bibliotecas = np.random.rand(n_samples)
    tipo_app = np.random.randint(0, 3, size=n_samples)
    rendimiento_deseado = np.random.rand(n_samples)

    # Combinar en una matriz de características
    X = np.column_stack((
        velocidad_requerida,
        facilidad_mantenimiento,
        disponibilidad_bibliotecas,
        tipo_app,
        rendimiento_deseado
    ))

    # Inicializar vector de etiquetas
    y = np.zeros(n_samples, dtype=int)

    # Asignar etiquetas según reglas heurísticas
    for i in range(n_samples):
        # Python: bueno para mantenimiento y bibliotecas, no tan bueno en rendimiento
        if (facilidad_mantenimiento[i] > 0.6 and
                disponibilidad_bibliotecas[i] > 0.7 and
                rendimiento_deseado[i] < 0.7):
            y[i] = 0

        # JavaScript: ideal para web, mantenimiento moderado a alto
        elif (tipo_app[i] == 0 and
              facilidad_mantenimiento[i] > 0.5):
            y[i] = 1

        # Java: equilibrado, bueno para móvil
        elif (tipo_app[i] == 1 and
              velocidad_requerida[i] > 0.4 and
              rendimiento_deseado[i] > 0.5):
            y[i] = 2

        # C++: alto rendimiento, sistemas
        elif (rendimiento_deseado[i] > 0.8 or
              (tipo_app[i] == 2 and velocidad_requerida[i] > 0.7)):
            y[i] = 3

        # Casos sin regla clara, asignar aleatoriamente
        else:
            y[i] = np.random.randint(0, 4)

    return X, y


class LanguagePredictor:
    def __init__(self):
        """Inicializa el predictor de lenguajes con un modelo Random Forest"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.language_map = {
            0: "Python",
            1: "JavaScript",
            2: "Java",
            3: "C++"
        }

    def train(self, X, y):
        """
        Entrena el modelo con los datos proporcionados

        Args:
            X: np.ndarray - Matriz de características
            y: np.ndarray - Vector de etiquetas
        """
        self.model.fit(X, y)

    def predict(self, features: np.ndarray) -> str:
        """
        Predice el lenguaje más adecuado para un nuevo proyecto

        Args:
            features: np.ndarray - Array con las 5 características del proyecto
                [velocidad_requerida, facilidad_mantenimiento, disponibilidad_bibliotecas,
                 tipo_app, rendimiento_deseado]

        Returns:
            str - Nombre del lenguaje recomendado
        """
        # Asegurar que features es un array 2D para sklearn
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Obtener la predicción numérica
        pred_idx = self.model.predict(features)[0]

        # Traducir a nombre de lenguaje
        return self.language_map[pred_idx]


# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos y entrenar
    X, y = generate_dataset(n_samples=100)

    predictor = LanguagePredictor()
    predictor.train(X, y)

    # Crear un proyecto nuevo
    # [velocidad_requerida, facilidad_mantenimiento, disponibilidad_bibliotecas, tipo_app, rendimiento_deseado]
    new_project = np.array([0.7, 0.9, 0.5, 1, 0.6])

    # Predecir lenguaje ideal
    pred = predictor.predict(new_project)
    print(f"Lenguaje recomendado para el nuevo proyecto: {pred}")