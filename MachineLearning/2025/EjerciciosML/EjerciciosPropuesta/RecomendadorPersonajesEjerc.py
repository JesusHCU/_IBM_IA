from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


class Player:
    """Clase que representa un jugador con sus características"""
    def __init__(self, name, level, aggressiveness, cooperation, exploration, preferred_class=None):
        # TODO: Guarda los atributos del jugador
        # Hint: Usa self.attribute = attribute para cada atributo
        pass

    def to_features(self):
        # TODO: Retorna una lista con las características numéricas
        # Hint: [level, aggressiveness, cooperation, exploration]
        return []

    def __str__(self):
        # Ya implementado para ti
        return f"{self.name} (Level {self.level})"


# Esta clase ya está implementada para ti
class PlayerDataset:
    def __init__(self, players):
        self.players = players

    def get_X(self):
        return [player.to_features() for player in self.players]

    def get_y(self):
        return [player.preferred_class for player in self.players]


class ClassRecommender:
    """Sistema de recomendación de clases usando KNN"""
    def __init__(self, n_neighbors=3):
        # Ya inicializado para ti
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.trained = False

    def train(self, dataset):
        """
        TODO: Entrena el modelo con el dataset proporcionado
        Hint: Usa self.model.fit(X, y)
        """
        # TODO: Obtén X e y del dataset
        # TODO: Entrena el modelo
        self.trained = True

    def predict(self, player):
        """
        TODO: Predice la mejor clase para un nuevo jugador
        Hint: Usa self.model.predict()
        """
        if not self.trained:
            return "Error: Modelo no entrenado"
        
        # TODO: Obtén las características del jugador
        # TODO: Realiza la predicción
        return "Guerrero"  # Reemplaza con tu implementación


# Código de ejemplo
if __name__ == "__main__":
    # Datos de entrenamiento
    jugadores = [
        Player("Ana", 20, 0.8, 0.2, 0.1, "Guerrero"),
        Player("Beto", 45, 0.4, 0.8, 0.2, "Sanador"),
        Player("Carlos", 33, 0.6, 0.4, 0.6, "Arquero"),
        Player("Diana", 60, 0.3, 0.9, 0.3, "Sanador"),
        Player("Eva", 50, 0.7, 0.2, 0.9, "Mago")
    ]
    
    # Crear dataset y recomendador
    dataset = PlayerDataset(jugadores)
    recomendador = ClassRecommender(n_neighbors=3)
    
    # Entrenar modelo
    print("Entrenando modelo...")
    recomendador.train(dataset)
    
    # Probar con nuevo jugador
    nuevo_jugador = Player(
        "Jugador Nuevo", 
        level=40,              # Nivel 1-100
        aggressiveness=0.6,    # Agresividad 0-1
        cooperation=0.3,       # Cooperación 0-1
        exploration=0.8        # Exploración 0-1
    )
    
    # Obtener recomendación
    clase_recomendada = recomendador.predict(nuevo_jugador)
    print(f"\nPara el jugador {nuevo_jugador.name}:")
    print(f"Clase recomendada: {clase_recomendada}")