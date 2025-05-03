import numpy as np
from sklearn.svm import SVC


# TODO: Completa la clase AirSample
class AirSample:
    def __init__(self, pm25, pm10, o3, no2, quality=None):
        # Completa el constructor con los parámetros dados
        # Hint: Guarda los valores en self.pm25, self.pm10, etc.
        pass

    def to_vector(self):
        # Retorna una lista con las características [pm25, pm10, o3, no2]
        # Hint: Usa los atributos guardados en el constructor
        pass


# TODO: Completa la clase AirDataGenerator
class AirDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        samples = []
        # Genera num_samples muestras de aire
        # Hint: Usa np.random.uniform() para generar valores aleatorios
        # Hint: Usa AirSample para crear cada muestra
        # Hint: La calidad es 1 si pm25 > 35 o pm10 > 50 o no2 > 40, sino 0
        return samples


# La clase AirQualityClassifier ya está implementada
class AirQualityClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear', random_state=42)

    def fit(self, samples):
        X = [sample.to_vector() for sample in samples]
        y = [sample.quality for sample in samples]
        self.model.fit(X, y)

    def predict(self, sample):
        return self.model.predict([sample.to_vector()])[0]


# TODO: Completa la clase AirQualityExample
class AirQualityExample:
    def run(self):
        # 1. Crea un generador con 200 muestras
        # Hint: Usa AirDataGenerator
        
        # 2. Genera las muestras
        # Hint: Usa el método generate()
        
        # 3. Crea y entrena el clasificador
        # Hint: Usa AirQualityClassifier
        
        # 4. Crea una nueva muestra para predecir
        # Hint: Usa AirSample con valores de tu elección
        
        # 5. Realiza la predicción
        # Hint: Usa el método predict del clasificador
        
        # 6. Muestra los resultados
        # Hint: Imprime los valores de la muestra y si es saludable o no
        pass


# Ejecuta el ejemplo
if __name__ == "__main__":
    example = AirQualityExample()
    example.run()