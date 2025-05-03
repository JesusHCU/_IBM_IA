from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np


class Player:
    def __init__(self, player_name, character_type, avg_session_time, matches_played,
                 aggressive_actions, defensive_actions, items_bought, victories, style=None):
        self.player_name = player_name
        self.character_type = character_type
        self.avg_session_time = avg_session_time
        self.matches_played = matches_played
        self.aggressive_actions = aggressive_actions
        self.defensive_actions = defensive_actions
        self.items_bought = items_bought
        self.victories = victories
        self.style = style


class GameModel:
    def __init__(self, players):
        self.players = players
        self.classification_model = None
        self.regression_model = None
        self.cluster_model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def _prepare_features(self, for_clustering=False):
        """Prepara los datos para el entrenamiento o predicción"""
        data = []
        for player in self.players:
            data.append([
                player.character_type,
                player.avg_session_time,
                player.matches_played,
                player.aggressive_actions,
                player.defensive_actions,
                player.items_bought,
                player.victories
            ])

        df = pd.DataFrame(data, columns=[
            'character_type', 'avg_session_time', 'matches_played',
            'aggressive_actions', 'defensive_actions', 'items_bought', 'victories'
        ])

        # Codificar character_type
        df['character_type'] = self.encoder.fit_transform(df['character_type'])

        if for_clustering:
            # Para clustering, no necesitamos las victorias (podría ser una meta a predecir)
            X = df.drop('victories', axis=1)
            return self.scaler.fit_transform(X)
        else:
            return df

    def _prepare_target_classification(self):
        """Prepara los datos de estilo para clasificación"""
        return [player.style for player in self.players]

    def _prepare_target_regression(self):
        """Prepara los datos de victorias para regresión"""
        return [player.victories for player in self.players]

    def train_classification_model(self):
        """Entrena el modelo de clasificación para predecir el estilo de juego"""
        X = self._prepare_features()
        y = self._prepare_target_classification()

        self.classification_model = RandomForestClassifier(random_state=42)
        self.classification_model.fit(X, y)

    def train_regression_model(self):
        """Entrena el modelo de regresión para predecir las victorias"""
        X = self._prepare_features().drop('victories', axis=1, errors='ignore')
        y = self._prepare_target_regression()

        self.regression_model = RandomForestRegressor(random_state=42)
        self.regression_model.fit(X, y)

    def train_clustering_model(self, n_clusters=2):
        """Entrena el modelo de clustering para agrupar jugadores"""
        X = self._prepare_features(for_clustering=True)

        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_model.fit(X)

    def _prepare_player_features(self, player, for_clustering=False):
        """Prepara los datos de un jugador individual para predicción"""
        data = [[
            player.character_type,
            player.avg_session_time,
            player.matches_played,
            player.aggressive_actions,
            player.defensive_actions,
            player.items_bought,
            player.victories if hasattr(player, 'victories') else 0
        ]]

        df = pd.DataFrame(data, columns=[
            'character_type', 'avg_session_time', 'matches_played',
            'aggressive_actions', 'defensive_actions', 'items_bought', 'victories'
        ])

        # Codificar character_type
        df['character_type'] = self.encoder.transform(df['character_type'])

        if for_clustering:
            return self.scaler.transform(df.drop('victories', axis=1))
        else:
            return df

    def predict_style(self, player):
        """Predice el estilo de juego de un jugador"""
        if not self.classification_model:
            raise ValueError("Classification model not trained. Call train_classification_model() first.")

        X = self._prepare_player_features(player)
        return self.classification_model.predict(X)[0]

    def predict_victories(self, player):
        """Predice el número de victorias de un jugador"""
        if not self.regression_model:
            raise ValueError("Regression model not trained. Call train_regression_model() first.")

        X = self._prepare_player_features(player).drop('victories', axis=1, errors='ignore')
        return self.regression_model.predict(X)[0]

    def assign_cluster(self, player):
        """Asigna un cluster a un jugador"""
        if not self.cluster_model:
            raise ValueError("Cluster model not trained. Call train_clustering_model() first.")

        X = self._prepare_player_features(player, for_clustering=True)
        return self.cluster_model.predict(X)[0]

    def show_players_by_cluster(self):
        """Muestra los jugadores agrupados por cluster (tarea opcional)"""
        if not self.cluster_model:
            raise ValueError("Cluster model not trained. Call train_clustering_model() first.")

        # Obtener las asignaciones de cluster para cada jugador
        labels = self.cluster_model.labels_

        # Agrupar jugadores por cluster
        clusters = {}
        for i, player in enumerate(self.players):
            cluster = labels[i]
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(player)

        # Mostrar los resultados
        for cluster, players in clusters.items():
            print(f"Cluster {cluster}:")
            for player in players:
                print(f"{player.player_name} - {player.character_type.capitalize()} - {player.style.capitalize()}")
            print()


# Prueba del sistema con los datos proporcionados
if __name__ == "__main__":
    # Crear datos de prueba para varios jugadores
    players_data = [
        Player("P1", "mage", 40, 30, 90, 50, 20, 18, "aggressive"),
        Player("P2", "tank", 60, 45, 50, 120, 25, 24, "strategic"),
        Player("P3", "archer", 50, 35, 95, 60, 22, 20, "aggressive"),
        Player("P4", "tank", 55, 40, 60, 100, 28, 22, "strategic"),
    ]

    # Instanciar el modelo con los datos de los jugadores
    model = GameModel(players_data)

    # Entrenar los modelos
    model.train_classification_model()
    model.train_regression_model()
    model.train_clustering_model()

    # Crear un nuevo jugador para realizar predicciones
    new_player = Player("TestPlayer", "mage", 42, 33, 88, 45, 21, 0)

    # Realizar predicciones
    predicted_style = model.predict_style(new_player)
    predicted_victories = model.predict_victories(new_player)
    predicted_cluster = model.assign_cluster(new_player)

    # Imprimir los resultados de las predicciones
    print(f"Estilo de juego predicho para {new_player.player_name}: {predicted_style}")
    print(f"Victorias predichas para {new_player.player_name}: {predicted_victories:.2f}")
    print(f"Cluster asignado a {new_player.player_name}: {predicted_cluster}")

    # Mostrar jugadores por cluster (tarea opcional)
    print("\nJugadores agrupados por cluster:")
    model.show_players_by_cluster()