from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


class App:
    """Clase que representa una aplicación móvil"""
    def __init__(self, app_name, monthly_users, avg_session_length, 
                 retention_rate, social_shares, success=None):
        # TODO: Guarda los atributos de la app
        # Hint: Usa self.attribute_name = attribute_name
        pass

    def to_features(self):
        # TODO: Retorna una lista con las características numéricas
        # Hint: [monthly_users, avg_session_length, retention_rate, social_shares]
        return []


# Esta clase ya está implementada para ti
class AppDataset:
    def __init__(self, apps):
        self.apps = apps

    def get_feature_matrix(self):
        return [app.to_features() for app in self.apps]

    def get_target_vector(self):
        return [app.success for app in self.apps if app.success is not None]


class SuccessPredictor:
    """Clase para predecir el éxito de una app"""
    def __init__(self):
        # Ya inicializado para ti
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, dataset):
        # TODO: Entrena el modelo con el dataset
        # Hint: 
        # 1. Obtén X e y del dataset
        # 2. Escala X usando self.scaler
        # 3. Entrena el modelo con los datos escalados
        pass

    def predict(self, app):
        # TODO: Predice si la app será exitosa
        # Hint:
        # 1. Verifica que el modelo está entrenado
        # 2. Obtén y escala las características
        # 3. Realiza la predicción
        pass


# Código de ejemplo
if __name__ == "__main__":
    # Datos de ejemplo
    apps_training = [
        App("App1", 10000, 15.0, 0.7, 1500, 1),
        App("App2", 500, 5.0, 0.2, 50, 0),
        App("App3", 15000, 20.0, 0.8, 2000, 1),
        App("App4", 800, 8.0, 0.3, 100, 0)
    ]

    # Crear y entrenar el modelo
    dataset = AppDataset(apps_training)
    predictor = SuccessPredictor()
    predictor.train(dataset)

    # Probar con una nueva app
    nueva_app = App(
        "MiApp", 
        monthly_users=12000,      # usuarios mensuales
        avg_session_length=18.0,  # minutos por sesión
        retention_rate=0.6,       # tasa de retención (0-1)
        social_shares=1800        # veces compartida
    )

    # Realizar predicción
    resultado = predictor.predict(nueva_app)
    print(f"\nPredicción para {nueva_app.app_name}:")
    print("¿Será exitosa?:", "Sí" if resultado == 1 else "No")