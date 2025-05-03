from typing import List
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict


# Clase para representar a un jugador
class Player:
    def __init__(self, name: str, avg_session_time: float, missions_completed: int,
                 accuracy: float, aggressiveness: float):
        self.name = name
        self.avg_session_time = avg_session_time
        self.missions_completed = missions_completed
        self.accuracy = accuracy
        self.aggressiveness = aggressiveness

    def to_feature_vector(self):
        return [self.avg_session_time, self.missions_completed, self.accuracy, self.aggressiveness]


# Clase que agrupa a los jugadores usando KMeans
class PlayerClusterer:
    def __init__(self):
        self.kmeans = None
        self.players = []

    def fit(self, players: List[Player], n_clusters: int):
        self.players = players
        X = [player.to_feature_vector() for player in players]
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(X)

    def predict(self, player: Player) -> int:
        return int(self.kmeans.predict([player.to_feature_vector()])[0])

    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_

    def print_cluster_summary(self, players: List[Player]):
        clusters = defaultdict(list)
        for player in players:
            cluster_id = self.predict(player)
            clusters[cluster_id].append(player.name)

        for cluster_id, names in sorted(clusters.items()):
            print(f"Cluster {cluster_id}:")
            for name in names:
                print(f"  - {name}")


# Clase principal que ejecuta el análisis
class GameAnalytics:
    def run(self):
        data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            ("Diana", 0.8, 15, 0.55, 0.9),
            ("Eve", 2.7, 120, 0.88, 0.25),
            ("Frank", 1.1, 30, 0.62, 0.65),
            ("Grace", 0.9, 18, 0.58, 0.85),
            ("Hank", 3.2, 160, 0.91, 0.15)
        ]

        players = [Player(*d) for d in data]
        clusterer = PlayerClusterer()
        clusterer.fit(players, n_clusters=3)

        print("Resumen de Clusters:")
        clusterer.print_cluster_summary(players)

        new_player = Player("Zoe", 1.5, 45, 0.65, 0.5)
        predicted_cluster = clusterer.predict(new_player)
        print(f"\nJugador {new_player.name} pertenece al cluster: {predicted_cluster}")


# Ejecutar si se corre como script principal
if __name__ == "__main__":
    analytics = GameAnalytics()
    analytics.run()
