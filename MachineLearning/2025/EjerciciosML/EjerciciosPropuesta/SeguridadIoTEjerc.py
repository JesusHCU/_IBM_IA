import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class IoTKNNClassifier:
    """Clasificador de seguridad para dispositivos IoT"""
    def __init__(self, n_neighbors=3, n_samples=50):
        # Ya inicializado para ti
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # Generar y dividir datos
        self._generate_data()
        self._split_data()

    def _generate_data(self):
        """
        TODO: Genera datos simulados de dispositivos IoT
        Hint: Usa np.random para generar datos aleatorios
        """
        # TODO: Genera datos para cada característica
        # Hint: Usa np.random.randint() con los rangos apropiados
        
        # TODO: Crea el DataFrame con los datos
        # Hint: Usa pd.DataFrame con las columnas correctas
        pass

    def _split_data(self):
        """
        TODO: Divide los datos en entrenamiento y prueba
        Hint: Usa train_test_split
        """
        # TODO: Separa características (X) y etiquetas (y)
        # TODO: Divide los datos en train y test
        pass

    def train(self):
        """
        TODO: Entrena el modelo k-NN
        Hint: Usa self.knn.fit()
        """
        pass

    def evaluate(self):
        """
        TODO: Evalúa el modelo con datos de prueba
        Hint: Usa accuracy_score
        """
        # TODO: Realiza predicciones en datos de prueba
        # TODO: Calcula y retorna la precisión
        return 0.0

    def predict(self, nuevo_dispositivo):
        """
        TODO: Predice si un nuevo dispositivo es seguro
        Hint: Usa self.knn.predict()
        """
        # TODO: Realiza la predicción
        # TODO: Retorna 0 (peligroso) o 1 (seguro)
        return 0


# Código de ejemplo
if __name__ == "__main__":
    # Crear clasificador
    print("Inicializando clasificador...")
    clasificador = IoTKNNClassifier(n_neighbors=3, n_samples=100)
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    clasificador.train()
    
    # Evaluar modelo
    precision = clasificador.evaluate()
    print(f"\nPrecisión del modelo: {precision:.2f}")
    
    # Probar con nuevos dispositivos
    dispositivos_prueba = [
        [500, 1000, 2],  # Dispositivo 1: Tráfico medio, UDP
        [100, 500, 1],   # Dispositivo 2: Tráfico bajo, TCP
        [900, 1400, 3]   # Dispositivo 3: Tráfico alto, HTTP
    ]
    
    print("\nProbando nuevos dispositivos:")
    for i, dispositivo in enumerate(dispositivos_prueba, 1):
        prediccion = clasificador.predict(dispositivo)
        print(f"\nDispositivo {i}:")
        print(f"- Paquetes/s: {dispositivo[0]}")
        print(f"- Bytes/paquete: {dispositivo[1]}")
        print(f"- Protocolo: {['TCP', 'UDP', 'HTTP'][dispositivo[2]-1]}")
        print(f"→ Estado: {'Seguro' if prediccion == 1 else 'Peligroso'}")