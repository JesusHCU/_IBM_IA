from typing import List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # No estaba en los imports originales del template, pero es necesaria
import numpy as np
from collections import defaultdict

# TODO: Implementa la clase Player
# Representa a un jugador con sus estadísticas.
class Player:
    # Constructor: inicializa los atributos del jugador.
    def __init__(self, name: str, avg_session_time: float, missions_completed: int,
                 accuracy: float, aggressiveness: float):
        # ### YOUR CODE HERE ###
        # Asigna los valores de los argumentos a los atributos de la instancia (usando 'self.').
        pass # Elimina este 'pass' cuando añadas tu código


    # Método para obtener el vector de características numéricas del jugador.
    # Este vector se usará como entrada para el algoritmo de agrupamiento.
    # ¡No incluyas el nombre del jugador aquí!
    def to_feature_vector(self):
        # ### YOUR CODE HERE ###
        # Retorna una lista o un array de NumPy que contenga las 4 estadísticas numéricas del jugador.
        pass # Elimina este 'pass' cuando añadas tu código


    # Puedes añadir un método __repr__ para una mejor representación del objeto (opcional)
    # def __repr__(self):
    #     return f"Player(name='{self.name}', ...)"


# TODO: Implementa la clase PlayerClusterer
# Se encarga de aplicar el algoritmo K-Means para agrupar a los jugadores.
class PlayerClusterer:
    # Constructor: inicializa el modelo KMeans y el escalador StandardScaler.
    def __init__(self):
        # ### YOUR CODE HERE ###
        # Inicializa una instancia de StandardScaler para escalar los datos.
        # Inicializa una instancia de KMeans. El número de clusters se definirá en el método 'fit'.
        # Puedes establecer un random_state en KMeans para resultados reproducibles (ej: random_state=42).
        pass # Elimina este 'pass' cuando añadas tu código


    # Método para entrenar el modelo K-Means con una lista de jugadores.
    # Recibe la lista de objetos Player y el número de clusters deseado.
    def fit(self, players: List[Player], n_clusters: int):
        # ### YOUR CODE HERE ###
        # 1. Convierte la lista de objetos Player en una matriz de características numéricas (NumPy array).
        #    Usa el método 'to_feature_vector()' de cada jugador. Cada fila de la matriz será un vector de jugador.

        # 2. Escala los datos numéricos usando el StandardScaler inicializado en el constructor.
        #    Usa el método fit_transform() del escalador para entrenarlo y escalar los datos.
        #    Esto es crucial porque K-Means es sensible a la escala de las características.

        # 3. Inicializa el modelo KMeans con el número de clusters especificado y tu random_state.
        #    Usa el método fit() del modelo KMeans con los datos *escalados*.

        # Opcional: Puedes almacenar las etiquetas asignadas a los jugadores (self.kmeans.labels_).
        pass # Elimina este 'pass' cuando añadas tu código


    # Método para predecir el cluster al que pertenece un jugador nuevo.
    # Recibe un solo objeto Player.
    # Retorna el índice del cluster (un entero).
    def predict(self, player: Player) -> int:
        # ### YOUR CODE HERE ###
        # 1. Obtén el vector de características del jugador a predecir usando to_feature_vector().
        # 2. Convierte este vector a un array de NumPy y asegúrate de que tiene la forma correcta (2D array con 1 fila)
        #    para ser compatible con los métodos del escalador y KMeans. (Pista: usa .reshape(1, -1)).
        # 3. Escala el vector usando el *mismo* escalador que se entrenó en el método 'fit'.
        #    Usa el método transform() del escalador (¡no fit_transform!).
        # 4. Usa el modelo KMeans entrenado (self.kmeans) para predecir el cluster del vector escalado.
        #    Usa el método predict().
        # 5. Retorna el resultado de la predicción (el predict devuelve un array, extrae el primer elemento).
        pass # Elimina este 'pass' cuando añadas tu código


    # Método para imprimir un resumen de los clusters.
    # Muestra qué jugadores pertenecen a cada cluster.
    # Asume que el método 'fit' ya ha sido llamado.
    def print_cluster_summary(self, players: List[Player]):
        # ### YOUR CODE HERE ###
        # 1. Verifica si el modelo ha sido entrenado (es decir, si self.kmeans y self.kmeans.labels_ existen).
        # 2. Crea un diccionario (o usa collections.defaultdict) para agrupar jugadores por su etiqueta de cluster.
        # 3. Itera sobre la lista original de 'players' y sus etiquetas asignadas (self.kmeans.labels_).
        #    Añade el nombre de cada jugador al grupo de su cluster en el diccionario.
        #    Asegúrate de que el orden de las etiquetas se corresponde con el orden de los jugadores que se usaron en 'fit'.
        # 4. Imprime la información de cada cluster, listando los nombres de los jugadores que contiene.
        pass # Elimina este 'pass' cuando añadas tu código


# Clase principal que ejecuta el análisis
class GameAnalytics:
    def run(self):
        # Datos de ejemplo de jugadores: (nombre, tiempo_sesion_promedio, misiones_completadas, precision, agresividad)
        data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            # ### YOUR CODE HERE ###
            # Añade aquí al menos 3 jugadores más con estadísticas variadas.
            # Piensa en diferentes "tipos" de jugadores.
        ]

        # ### YOUR CODE HERE ###
        # 1. Crea una lista de objetos 'Player' a partir de los datos proporcionados en la lista 'data'.

        # 2. Crea una instancia de 'PlayerClusterer'.
        #    Define cuántos clusters deseas encontrar (ej: n_clusters = 3).
        #    Llama al método 'fit' del clusterer, pasándole la lista de jugadores y el número de clusters.

        # 3. Llama al método 'print_cluster_summary' del clusterer, pasándole la lista original de jugadores
        #    para ver cómo se han agrupado.

        # 4. (Opcional pero recomendado) Crea un nuevo jugador ficticio que no esté en la lista original.
        #    Usa el método 'predict' del clusterer para determinar a qué cluster pertenecería este nuevo jugador.
        #    Imprime el resultado de la predicción.
        pass # Elimina este 'pass' cuando añadas tu código


# Punto de entrada del programa.
# Crea una instancia de GameAnalytics y ejecuta el análisis.
if __name__ == "__main__":
    analytics = GameAnalytics()
    analytics.run()