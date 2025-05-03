# exercise.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import unittest


class IoTKNNClassifier:
    def __init__(self, n_neighbors=3, n_samples=50):
        """
        Inicializa el clasificador IoTKNNClassifier.
        :param n_neighbors: Número de vecinos para el algoritmo k-NN.
        :param n_samples: Número de dispositivos IoT en los datos sintéticos.
        """
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples

        # Generar datos sintéticos
        self._generate_data()

        # Dividir los datos en entrenamiento y prueba
        self.X = self.df.drop(columns=["seguro"])  # Características
        self.y = self.df["seguro"]  # Etiquetas

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Inicializar el modelo k-NN
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def _generate_data(self):
        """
        Genera datos sintéticos para simular dispositivos IoT.
        """
        np.random.seed(42)  # Para reproducibilidad

        paquetes_por_segundo = np.random.randint(10, 1000, self.n_samples)
        bytes_por_paquete = np.random.randint(50, 1500, self.n_samples)
        protocolo = np.random.randint(1, 4, self.n_samples)  # 1=TCP, 2=UDP, 3=HTTP
        seguro = np.random.randint(0, 2, self.n_samples)  # 0=Peligroso, 1=Seguro

        # Crear un DataFrame con los datos generados
        self.df = pd.DataFrame({
            "paquetes_por_segundo": paquetes_por_segundo,
            "bytes_por_paquete": bytes_por_paquete,
            "protocolo": protocolo,
            "seguro": seguro
        })

    def train(self):
        """
        Entrena el modelo k-NN con los datos de entrenamiento.
        """
        self.knn.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evalúa el modelo con los datos de prueba y retorna la precisión.
        :return: Precisión del modelo.
        """
        y_pred = self.knn.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def predict(self, nuevo_dispositivo):
        """
        Predice si un nuevo dispositivo IoT es seguro o peligroso.
        :param nuevo_dispositivo: Lista con [paquetes_por_segundo, bytes_por_paquete, protocolo].
        :return: 1 (seguro) o 0 (peligroso).
        """
        prediccion = self.knn.predict([nuevo_dispositivo])
        return prediccion[0]


# Pruebas unitarias
class TestIoTKNNClassifier(unittest.TestCase):
    def test_training_and_evaluation(self):
        """
        Prueba el entrenamiento y evaluación del modelo.
        """
        classifier = IoTKNNClassifier(n_neighbors=3, n_samples=100)

        # Entrenar el modelo
        classifier.train()

        # Evaluar el modelo
        accuracy = classifier.evaluate()

        # Verificar que la precisión sea razonable (entre 0 y 1)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_prediction(self):
        """
        Prueba la predicción para un nuevo dispositivo IoT.
        """
        classifier = IoTKNNClassifier(n_neighbors=3, n_samples=100)

        # Entrenar el modelo
        classifier.train()

        # Predecir para un nuevo dispositivo
        nuevo_dispositivo = [500, 1000, 2]  # Ejemplo de características
        prediccion = classifier.predict(nuevo_dispositivo)

        # Verificar que la predicción sea válida (0 o 1)
        self.assertIn(prediccion, [0, 1])


# Código principal para ejecución independiente
if __name__ == "__main__":
    # Crear una instancia del clasificador
    classifier = IoTKNNClassifier(n_neighbors=3, n_samples=100)

    # Entrenar el modelo
    classifier.train()

    # Evaluar el modelo
    precision = classifier.evaluate()
    print(f"Precisión del modelo: {precision:.2f}")

    # Predecir la seguridad de un nuevo dispositivo IoT
    nuevo_dispositivo = [500, 1000, 2]  # Ejemplo: paquetes/segundo=500, bytes/paquete=1000, protocolo=UDP (2)
    prediccion = classifier.predict(nuevo_dispositivo)
    print(f"Predicción para el nuevo dispositivo IoT: {'Seguro' if prediccion == 1 else 'Peligroso'}")

    # Ejecutar pruebas unitarias
    unittest.main(argv=[''], exit=False)
