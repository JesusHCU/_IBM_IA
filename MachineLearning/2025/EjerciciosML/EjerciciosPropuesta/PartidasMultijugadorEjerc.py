import numpy as np
from sklearn.linear_model import LogisticRegression


class PlayerMatchData:
    """
    TODO: Implementa la clase para almacenar datos de partidas
    Hint: Guarda las estadísticas de juego de un jugador
    """
    def __init__(self, kills, deaths, assists, damage_dealt, 
                 damage_received, healing_done, objective_time, won):
        # TODO: Inicializa los atributos del jugador
        # Hint: Guarda todos los parámetros como atributos
        pass

    def to_dict(self):
        # TODO: Retorna un diccionario con los datos del jugador
        # Hint: Incluye todas las estadísticas excepto 'won'
        return {}


def generate_synthetic_data(n=100):
    """
    TODO: Implementa la generación de datos sintéticos
    Hint: Crea n partidas aleatorias
    """
    data = []
    for _ in range(n):
        # TODO: Genera estadísticas aleatorias
        # Hint: Usa estas distribuciones:
        # - kills: np.random.poisson(5)
        # - deaths: np.random.poisson(3)
        # - assists: np.random.poisson(2)
        # - damage_dealt: kills * 300 + ruido aleatorio
        # - damage_received: deaths * 400 + ruido aleatorio
        # - healing_done: entre 0 y 300
        # - objective_time: entre 0 y 120
        pass
    return data


class VictoryPredictor:
    """
    TODO: Implementa el predictor de victorias
    Hint: Usa LogisticRegression para clasificación
    """
    def __init__(self):
        # TODO: Inicializa el modelo
        # Hint: Usa LogisticRegression()
        pass

    def train(self, data):
        # TODO: Entrena el modelo con los datos
        # Hint: 
        # 1. Extrae características (X) y etiquetas (y)
        # 2. Convierte a arrays de numpy
        # 3. Entrena el modelo
        pass

    def predict(self, player):
        # TODO: Predice si el jugador ganará
        # Hint:
        # 1. Extrae características del jugador
        # 2. Realiza la predicción
        pass


# Código de prueba
if __name__ == "__main__":
    # Generar datos de entrenamiento
    print("Generando datos de entrenamiento...")
    training_data = generate_synthetic_data(150)
    
    # Crear y entrenar el predictor
    print("\nEntrenando modelo...")
    predictor = VictoryPredictor()
    predictor.train(training_data)
    
    # Probar con un jugador de ejemplo
    print("\nProbando predicción...")
    test_player = PlayerMatchData(
        kills=8,           # Eliminaciones
        deaths=2,          # Muertes
        assists=3,         # Asistencias
        damage_dealt=2400, # Daño realizado
        damage_received=800, # Daño recibido
        healing_done=120,   # Curación realizada
        objective_time=90,  # Tiempo en objetivo
        won=None           # Resultado desconocido
    )
    
    # Realizar predicción
    prediction = predictor.predict(test_player)
    print(f"\nPredicción para el jugador de prueba:")
    print(f"¿Ganará la partida? {'Sí' if prediction == 1 else 'No'}")