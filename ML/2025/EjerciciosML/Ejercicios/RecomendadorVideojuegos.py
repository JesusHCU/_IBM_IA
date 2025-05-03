import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Clase que representa un videojuego con características numéricas
class VideoGame:
    def __init__(self, action, strategy, graphics, difficulty, liked=None):
        self.action = action
        self.strategy = strategy
        self.graphics = graphics
        self.difficulty = difficulty
        self.liked = liked

    def to_vector(self):
        return [self.action, self.strategy, self.graphics, self.difficulty]


# Clase para generar datos sintéticos de videojuegos
class VideoGameGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        games = []
        for _ in range(self.num_samples):
            action = np.round(np.random.rand(), 2)
            strategy = np.round(np.random.rand(), 2)
            graphics = np.round(np.random.rand(), 2)
            difficulty = np.round(np.random.rand(), 2)

            # Regla para decidir si gustó
            liked = int((action > 0.7 or graphics > 0.7) and difficulty < 0.7)

            games.append(VideoGame(action, strategy, graphics, difficulty, liked))
        return games


# Clase para entrenar un clasificador usando Random Forest
class VideoGameClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, games):
        X = [game.to_vector() for game in games]
        y = [game.liked for game in games]
        self.model.fit(X, y)

    def predict(self, game):
        return self.model.predict([game.to_vector()])[0]


# Clase de ejemplo para ejecutar el sistema de recomendación
class VideoGameRecommendationExample:
    def run(self):
        # Generar 200 videojuegos aleatorios
        generator = VideoGameGenerator(200)
        games = generator.generate()

        # Entrenar el modelo
        classifier = VideoGameClassifier()
        classifier.fit(games)

        # Crear un nuevo videojuego para predecir
        new_game = VideoGame(action=0.9, strategy=0.5, graphics=0.85, difficulty=0.4)

        # Predecir si al jugador le gustará el nuevo juego
        prediction = classifier.predict(new_game)

        # Mostrar los resultados
        print("🎮 Nuevo juego:")
        print(
            f"Action: {new_game.action}, Strategy: {new_game.strategy}, Graphics: {new_game.graphics}, Difficulty: {new_game.difficulty}")
        print("✅ Le gustará al jugador el juego?", "Si!" if prediction == 1 else "No.")


# Ejecutar el ejemplo
if __name__ == "__main__":
    example = VideoGameRecommendationExample()
    example.run()