from sklearn.linear_model import LinearRegression
import numpy as np

class App:
    """Clase que representa una aplicación móvil"""
    def __init__(self, name, downloads, rating, size_mb, reviews, revenue=None):
        # TODO: Guarda los atributos de la app
        # Hint: Usa self.attribute_name = attribute_name para cada atributo
        pass


class RevenuePredictor:
    """Predictor de ingresos usando regresión lineal"""
    def __init__(self):
        # Ya inicializado para ti
        self.model = LinearRegression()

    def extract_features(self, apps):
        """
        TODO: Extrae las características de las apps
        Hint: downloads, rating, size_mb, reviews
        """
        # TODO: Crea una lista de características para cada app
        # Hint: Usa una lista de listas con los valores numéricos
        features = []
        return np.array(features)

    def fit(self, training_apps):
        """
        TODO: Entrena el modelo con los datos proporcionados
        Hint: Usa self.model.fit(X, y)
        """
        # TODO: Obtén X (características) e y (ingresos)
        # TODO: Entrena el modelo
        pass

    def predict(self, new_app):
        """
        TODO: Predice los ingresos para una nueva app
        Hint: Usa self.model.predict()
        """
        # TODO: Prepara las características de la nueva app
        # TODO: Realiza la predicción
        pass


# Código de ejemplo - Ya implementado para ti
if __name__ == "__main__":
    # Datos de entrenamiento
    training_apps = [
        App("App1", 200, 4.2, 45.0, 1800, 120.0),
        App("App2", 150, 4.5, 60.0, 2100, 135.0),
        App("App3", 300, 4.1, 55.0, 2500, 160.0),
    ]

    # Crear y entrenar el predictor
    predictor = RevenuePredictor()
    predictor.fit(training_apps)

    # Probar con una nueva app
    nueva_app = App(
        name="MiApp",
        downloads=250,    # miles de descargas
        rating=4.5,      # calificación (1-5)
        size_mb=50.0,    # tamaño en MB
        reviews=2000     # número de reseñas
    )

    # Predecir ingresos
    ingresos_predichos = predictor.predict(nueva_app)
    print(f"\nPredicción para {nueva_app.name}:")
    print(f"Ingresos estimados: ${ingresos_predichos:.2f}K")