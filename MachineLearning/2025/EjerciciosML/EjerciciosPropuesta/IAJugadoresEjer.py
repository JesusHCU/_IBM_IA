from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np


class Player:
    """
    TODO: Implementa la clase Player para almacenar la información de los jugadores
    Atributos necesarios:
    - player_name: nombre del jugador
    - character_type: tipo de personaje (mage, tank, archer)
    - avg_session_time: tiempo promedio de sesión
    - matches_played: partidas jugadas
    - aggressive_actions: acciones agresivas
    - defensive_actions: acciones defensivas
    - items_bought: objetos comprados
    - victories: victorias
    - style: estilo de juego (opcional)
    """
    def __init__(self, player_name, character_type, avg_session_time, matches_played,
                 aggressive_actions, defensive_actions, items_bought, victories, style=None):
        # TODO: Inicializa los atributos del jugador
        pass


class GameModel:
    def __init__(self, players):
        self.players = players
        self.classification_model = None
        self.regression_model = None
        self.cluster_model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def _prepare_features(self, for_clustering=False):
        """
        TODO: Implementa la preparación de características
        Hint: Crea un DataFrame con los datos de los jugadores
        """
        # TODO: Crea una lista con los datos de cada jugador
        data = []
        
        # TODO: Convierte la lista en DataFrame
        # Hint: Usa pd.DataFrame con las columnas correctas
        
        # TODO: Codifica character_type usando self.encoder
        
        # TODO: Si es para clustering, elimina la columna 'victories'
        pass

    def train_classification_model(self):
        """
        TODO: Implementa el entrenamiento del modelo de clasificación
        Hint: Usa RandomForestClassifier
        """
        # TODO: Prepara los datos (X e y)
        # TODO: Crea y entrena el modelo
        pass

    def train_clustering_model(self, n_clusters=2):
        """
        TODO: Implementa el entrenamiento del modelo de clustering
        Hint: Usa KMeans
        """
        # TODO: Prepara los datos
        # TODO: Crea y entrena el modelo
        pass

    def predict_style(self, player):
        """
        TODO: Implementa la predicción del estilo de juego
        """
        # TODO: Verifica que el modelo está entrenado
        # TODO: Prepara los datos del jugador
        # TODO: Realiza la predicción
        pass


# Código de prueba
if __name__ == "__main__":
    # TODO: Crea algunos jugadores de prueba
    players_data = [
        # Hint: Crea al menos 4 jugadores con diferentes características
    ]

    # TODO: Crea y entrena el modelo
    model = GameModel(players_data)
    
    # TODO: Crea un nuevo jugador para pruebas
    new_player = None  # Completa con datos de prueba
    
    # TODO: Realiza predicciones
    # Hint: Usa los métodos implementados