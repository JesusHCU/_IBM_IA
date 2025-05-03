from sklearn.linear_model import LinearRegression
import numpy as np


class Player:
    """
    Clase que representa a un jugador con sus estadísticas de juego.
    """

    def __init__(self, name, avg_session_time, avg_actions_per_min, avg_kills_per_session, victories=None):
        """
        Inicializa un nuevo jugador.

        Args:
            name (str): Nombre del jugador
            avg_session_time (float): Duración promedio de las sesiones en minutos
            avg_actions_per_min (float): Acciones por minuto realizadas
            avg_kills_per_session (float): Eliminaciones promedio por sesión
            victories (int, optional): Número de victorias. Por defecto es None.
        """
        self.name = name
        self.avg_session_time = avg_session_time
        self.avg_actions_per_min = avg_actions_per_min
        self.avg_kills_per_session = avg_kills_per_session
        self.victories = victories

    def to_features(self):
        """
        Devuelve las características del jugador como una lista para usar en el modelo.

        Returns:
            list: Lista con los valores de las características
        """
        return [self.avg_session_time, self.avg_actions_per_min, self.avg_kills_per_session]


class PlayerDataset:
    """
    Clase que representa una colección de jugadores para entrenar el modelo.
    """

    def __init__(self, players):
        """
        Inicializa el dataset con una lista de jugadores.

        Args:
            players (list): Lista de objetos Player
        """
        self.players = players

    def get_feature_matrix(self):
        """
        Obtiene la matriz de características X para el modelo.

        Returns:
            list: Lista de listas con las características de cada jugador
        """
        return [player.to_features() for player in self.players]

    def get_target_vector(self):
        """
        Obtiene el vector objetivo y (victorias) para el modelo.

        Returns:
            list: Lista con el número de victorias de cada jugador
        """
        return [player.victories for player in self.players if player.victories is not None]


class VictoryPredictor:
    """
    Clase encargada de entrenar y utilizar el modelo de regresión lineal.
    """

    def __init__(self):
        """
        Inicializa el predictor con un modelo de regresión lineal.
        """
        self.model = LinearRegression()

    def train(self, dataset):
        """
        Entrena el modelo con los datos del dataset.

        Args:
            dataset (PlayerDataset): El dataset con los jugadores para entrenar
        """
        X = np.array(dataset.get_feature_matrix())
        y = np.array(dataset.get_target_vector())
        self.model.fit(X, y)

    def predict(self, player):
        """
        Predice el número de victorias para un jugador.

        Args:
            player (Player): El jugador para el que se quiere hacer la predicción

        Returns:
            float: Número de victorias predichas
        """
        features = np.array([player.to_features()])
        return float(self.model.predict(features)[0])


# Ejemplo de uso
if __name__ == "__main__":
    # Crear jugadores de ejemplo (con victorias conocidas para entrenar)
    players = [
        Player("Alice", 40, 50, 6, 20),
        Player("Bob", 30, 35, 4, 10),
        Player("Charlie", 50, 60, 7, 25),
        Player("Diana", 20, 25, 2, 5),
        Player("Eve", 60, 70, 8, 30)
    ]

    # Crear el dataset
    dataset = PlayerDataset(players)

    # Crear y entrenar el predictor
    predictor = VictoryPredictor()
    predictor.train(dataset)

    # Crear un jugador nuevo para probar la predicción
    test_player = Player("TestPlayer", 45, 55, 5)

    # Hacer la predicción
    predicted = predictor.predict(test_player)
    print(f"Victorias predichas para {test_player.name}: {predicted:.2f}")