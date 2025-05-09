from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import unittest


def entrenar_y_evaluar_kmeans(X, y, k):
    """
    TODO: Implementa el entrenamiento y evaluación de K-Means
    
    Parámetros:
    X: Características de las flores
    y: Etiquetas verdaderas
    k: Número de clusters
    
    Retorna:
    dict: Diccionario con resultados del modelo
    """
    # TODO: Crea y entrena el modelo K-Means
    # Hint: Usa KMeans(n_clusters=k, random_state=42)
    
    # TODO: Obtén las asignaciones de clusters
    # Hint: Usa modelo.labels_
    
    # TODO: Calcula las métricas de evaluación
    # Hint: Usa inertia_, silhouette_score y adjusted_rand_score
    
    # TODO: Retorna el diccionario con los resultados
    return {
        "clusters": None,  # Asignaciones de cluster
        "inertia": None,  # Inercia del modelo
        "silhouette_score": None,  # Puntuación de silueta
        "adjusted_rand_score": None  # Puntuación Rand ajustada
    }


# Código de prueba
if __name__ == "__main__":
    # TODO: Carga el dataset Iris
    # Hint: Usa load_iris()
    
    # TODO: Entrena el modelo y obtén resultados
    # Hint: Usa k=3 clusters
    
    # TODO: Muestra los resultados
    # Hint: Imprime las métricas y algunas asignaciones


# Pruebas unitarias
class TestKMeans(unittest.TestCase):
    def setUp(self):
        """
        TODO: Configura los datos de prueba
        Hint: Carga el dataset Iris
        """
        pass

    def test_entrenar_y_evaluar_kmeans(self):
        """
        TODO: Implementa las pruebas del modelo
        Hint: Verifica tipos de datos y rangos de valores
        """
        pass