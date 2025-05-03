from sklearn.linear_model import LinearRegression
import numpy as np


class Player:
    """Clase que representa a un jugador con sus estadísticas"""
    def __init__(self, name, avg_session_time, avg_actions_per_min, 
                 avg_kills_per_session, victories=None):
        # TODO: Guarda los atributos del jugador
        # Hint: Usa self.attribute = attribute para cada estadística
        pass

    def to_features(self):
        # TODO: Retorna una lista con las características numéricas
        # Hint: [avg_session_time, avg_actions_per_min, avg_kills_per_session]
        return []


# Esta clase ya está implementada para ti
class PlayerDataset:
    def __init__(self, players):
        self.players = players

    def get_feature_matrix(self):
        return [player.to_features() for player in self.players]

    def get_target_vector(self):
        return [player.victories for player in self.players if player.victories is not None]


class VictoryPredictor:
    """Predictor de victorias usando regresión lineal"""
    def __init__(self):
        # Ya inicializado para ti
        self.model = LinearRegression()

    def train(self, dataset):
        """
        TODO: Entrena el modelo con el dataset proporcionado
        Hint: Usa self.model.fit(X, y)
        """
        # TODO: Obtén X (características) e y (victorias)
        # TODO: Entrena el modelo
        pass

    def predict(self, player):
        """
        TODO: Predice el número de victorias para un jugador
        Hint: Usa self.model.predict()
        """
        # TODO: Obtén las características del jugador
        # TODO: Realiza la predicción
        return 0  # Reemplaza con tu implementación


# Código de ejemplo
if __name__ == "__main__":
    # Datos de entrenamiento
    jugadores_ejemplo = [
        Player("Ana", 45, 55, 6, 22),    # 45min/sesión, 55 acciones/min, 6 kills/sesión
        Player("Beto", 30, 40, 4, 12),   # Menos experiencia
        Player("Carlos", 60, 65, 8, 30),  # Jugador experto
        Player("Diana", 25, 30, 3, 8)     # Principiante
    ]
    
    # Crear dataset y predictor
    dataset = PlayerDataset(jugadores_ejemplo)
    predictor = VictoryPredictor()
    
    # Entrenar modelo
    print("Entrenando modelo...")
    predictor.train(dataset)
    
    # Probar con nuevo jugador
    nuevo_jugador = Player(
        "Jugador Nuevo",
        avg_session_time=40,      # minutos por sesión
        avg_actions_per_min=50,   # acciones por minuto
        avg_kills_per_session=5   # eliminaciones por sesión
    )
    
    # Predecir victorias
    victorias_predichas = predictor.predict(nuevo_jugador)
    print(f"\nPredicción para {nuevo_jugador.name}:")
    print(f"- Tiempo promedio de sesión: {nuevo_jugador.avg_session_time} min")
    print(f"- Acciones por minuto: {nuevo_jugador.avg_actions_per_min}")
    print(f"- Eliminaciones por sesión: {nuevo_jugador.avg_kills_per_session}")
    print(f"\nVictorias predichas: {victorias_predichas:.1f}")