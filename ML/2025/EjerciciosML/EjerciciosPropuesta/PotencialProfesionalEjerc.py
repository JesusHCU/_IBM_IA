import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GameSimulator:
    """
    TODO: Implementa el simulador de estadísticas de jugadores
    """
    def __init__(self, n_players=200):
        # TODO: Guarda el número de jugadores a simular
        # Hint: Usa self.n_players = n_players
        pass

    def run(self):
        """
        TODO: Genera datos simulados de jugadores
        Hint: Usa np.random.rand() para generar números entre 0 y 1
        """
        # Para mantener los mismos resultados siempre
        np.random.seed(42)
        
        # TODO: Genera las estadísticas para cada jugador
        # Cada estadística debe ser un array de tamaño n_players
        partidas_ganadas = None  # Ratio de victorias
        horas_jugadas = None     # Tiempo de juego
        precision = None         # Precisión en el juego
        reaccion = None         # Tiempo de reacción
        estrategia = None       # Capacidad estratégica
        
        # TODO: Determina quién es jugador profesional
        # Un jugador es pro si cumple TODAS estas condiciones:
        # - Partidas ganadas > 70%
        # - Horas jugadas > 60%
        # - Precisión > 70%
        # - Reacción > 60%
        # - Estrategia > 60%
        etiquetas = None
        
        # TODO: Combina las características en matriz X
        X = None
        
        return X, etiquetas


class ProPlayerClassifier:
    """
    TODO: Implementa el clasificador de jugadores profesionales
    """
    def __init__(self):
        # TODO: Inicializa el modelo SVM
        # Hint: Usa SVC con kernel='rbf'
        pass

    def train(self, X, y):
        # TODO: Entrena el modelo con los datos
        # Hint: Usa el método fit
        pass

    def predict(self, player_stats):
        # TODO: Predice si el jugador es profesional
        # Hint: Retorna 1 (pro) o 0 (amateur)
        pass

    def evaluate(self, X_test, y_test):
        # TODO: Calcula la precisión del modelo
        # Hint: Usa accuracy_score
        pass


# Código de prueba
if __name__ == "__main__":
    # Generar datos de ejemplo
    print("Generando datos de jugadores...")
    simulator = GameSimulator(n_players=200)
    X, y = simulator.run()
    
    # Dividir en entrenamiento y prueba
    print("\nPreparando datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar clasificador
    print("\nEntrenando modelo...")
    classifier = ProPlayerClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluar modelo
    accuracy = classifier.evaluate(X_test, y_test)
    print(f"\nPrecisión del modelo: {accuracy:.2f}")
    
    # Probar con un nuevo jugador
    print("\nProbando con jugador nuevo...")
    jugador_nuevo = [0.75, 0.8, 0.85, 0.7, 0.65]
    prediccion = classifier.predict(jugador_nuevo)
    print("Estadísticas del jugador:")
    print("- Partidas ganadas: 75%")
    print("- Horas jugadas: 80%")
    print("- Precisión: 85%")
    print("- Reacción: 70%")
    print("- Estrategia: 65%")
    print(f"Predicción: {'Profesional' if prediccion == 1 else 'Amateur'}")