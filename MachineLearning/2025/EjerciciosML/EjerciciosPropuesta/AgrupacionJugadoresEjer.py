from typing import List
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict


# TODO: Implementa la clase Player
class Player:
    def __init__(self, name: str, avg_session_time: float, missions_completed: int,
                 accuracy: float, aggressiveness: float):
        # ### YOUR CODE HERE ###
        # Inicializa los atributos del jugador
        pass

    def to_feature_vector(self):
        # ### YOUR CODE HERE ###
        # Retorna una lista con las características del jugador
        pass


# TODO: Implementa la clase PlayerClusterer
class PlayerClusterer:
    def __init__(self):
        # ### YOUR CODE HERE ###
        # Inicializa el modelo KMeans y la lista de jugadores
        pass

    def fit(self, players: List[Player], n_clusters: int):
        # ### YOUR CODE HERE ###
        # Entrena el modelo KMeans con los datos de los jugadores
        pass

    def predict(self, player: Player) -> int:
        # ### YOUR CODE HERE ###
        # Predice el cluster al que pertenece un jugador
        pass

    def print_cluster_summary(self, players: List[Player]):
        # ### YOUR CODE HERE ###
        # Imprime un resumen de los clusters y sus jugadores
        pass


# Clase principal que ejecuta el análisis
class GameAnalytics:
    def run(self):
        # Datos de ejemplo
        data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            # ### YOUR CODE HERE ###
            # Añade al menos 3 jugadores más
        ]

        # ### YOUR CODE HERE ###
        # 1. Crear la lista de jugadores
        # 2. Crear y entrenar el clusterer
        # 3. Mostrar el resumen de clusters
        # 4. Crear un nuevo jugador y predecir su cluster


if __name__ == "__main__":
    analytics = GameAnalytics()
    analytics.run()