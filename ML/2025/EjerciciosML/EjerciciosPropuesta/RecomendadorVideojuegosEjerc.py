import numpy as np
from sklearn.ensemble import RandomForestClassifier


class VideoGame:
    """Clase que representa un videojuego"""
    def __init__(self, action, strategy, graphics, difficulty, liked=None):
        # TODO: Guarda los atributos del videojuego
        # Hint: Usa self.attribute = attribute para cada característica
        pass

    def to_vector(self):
        # TODO: Retorna una lista con las características numéricas
        # Hint: [action, strategy, graphics, difficulty]
        return []


# Esta clase ya está implementada para ti
class VideoGameGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        games = []
        for _ in range(self.num_samples):
            # Genera características aleatorias
            action = np.round(np.random.rand(), 2)
            strategy = np.round(np.random.rand(), 2)
            graphics = np.round(np.random.rand(), 2)
            difficulty = np.round(np.random.rand(), 2)
            
            # Determina si gustará basado en reglas
            liked = int((action > 0.7 or graphics > 0.7) and difficulty < 0.7)
            
            games.append(VideoGame(action, strategy, graphics, difficulty, liked))
        return games


class VideoGameClassifier:
    """Sistema de recomendación de videojuegos"""
    def __init__(self):
        # Ya inicializado para ti
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, games):
        """
        TODO: Entrena el modelo con la lista de juegos
        Hint: Usa self.model.fit(X, y)
        """
        # TODO: Obtén características (X) y etiquetas (y)
        # TODO: Entrena el modelo
        pass

    def predict(self, game):
        """
        TODO: Predice si a un jugador le gustará el juego
        Hint: Usa self.model.predict()
        """
        # TODO: Obtén el vector de características
        # TODO: Realiza la predicción
        return 0  # Reemplaza con tu implementación


# Código de ejemplo
if __name__ == "__main__":
    # Generar datos de entrenamiento
    print("Generando juegos de ejemplo...")
    generator = VideoGameGenerator(num_samples=200)
    juegos = generator.generate()
    
    # Crear y entrenar el clasificador
    print("\nEntrenando modelo...")
    clasificador = VideoGameClassifier()
    clasificador.fit(juegos)
    
    # Probar con nuevos juegos
    juegos_prueba = [
        VideoGame(action=0.9, strategy=0.5, graphics=0.8, difficulty=0.3),  # Debería gustar
        VideoGame(action=0.3, strategy=0.4, graphics=0.5, difficulty=0.9),  # No debería gustar
        VideoGame(action=0.8, strategy=0.7, graphics=0.9, difficulty=0.4)   # Debería gustar
    ]
    
    # Realizar predicciones
    print("\nProbando predicciones:")
    for i, juego in enumerate(juegos_prueba, 1):
        prediccion = clasificador.predict(juego)
        print(f"\nJuego {i}:")
        print(f"- Acción: {juego.action:.2f}")
        print(f"- Estrategia: {juego.strategy:.2f}")
        print(f"- Gráficos: {juego.graphics:.2f}")
        print(f"- Dificultad: {juego.difficulty:.2f}")
        print(f"¿Le gustará al jugador? {'¡Sí!' if prediccion == 1 else 'No'}")