from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


class App:
    """
    Representa una aplicación móvil con sus métricas relevantes.
    """

    def __init__(self, app_name, monthly_users, avg_session_length, retention_rate, social_shares, success=None):
        """
        Inicializa una app con sus características.

        Args:
            app_name (str): Nombre de la aplicación
            monthly_users (int): Número de usuarios mensuales
            avg_session_length (float): Duración media de las sesiones en minutos
            retention_rate (float): Tasa de retención entre 0 y 1
            social_shares (int): Número de veces compartida en redes sociales
            success (int, optional): 1 si fue exitosa, 0 si fracasó. Default: None
        """
        self.app_name = app_name
        self.monthly_users = monthly_users
        self.avg_session_length = avg_session_length
        self.retention_rate = retention_rate
        self.social_shares = social_shares
        self.success = success

    def to_features(self):
        """
        Devuelve una lista con las características numéricas de la app.

        Returns:
            list: Lista de características numéricas
        """
        return [
            self.monthly_users,
            self.avg_session_length,
            self.retention_rate,
            self.social_shares
        ]


class AppDataset:
    """
    Representa un conjunto de datos de aplicaciones móviles.
    """

    def __init__(self, apps):
        """
        Inicializa el dataset con una lista de apps.

        Args:
            apps (list): Lista de objetos App
        """
        self.apps = apps

    def get_feature_matrix(self):
        """
        Devuelve una matriz de características para todas las apps.

        Returns:
            list: Matriz de características (lista de listas)
        """
        return [app.to_features() for app in self.apps]

    def get_target_vector(self):
        """
        Devuelve un vector con los valores de éxito (1) o fracaso (0).

        Returns:
            list: Vector de etiquetas
        """
        return [app.success for app in self.apps if app.success is not None]


class SuccessPredictor:
    """
    Predice si una app será exitosa utilizando regresión logística.
    """

    def __init__(self):
        """
        Inicializa el predictor con un modelo de regresión logística.
        """
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, dataset):
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            dataset (AppDataset): Conjunto de datos de entrenamiento
        """
        X = np.array(dataset.get_feature_matrix())
        y = np.array(dataset.get_target_vector())

        # Escalar las características para mejorar el rendimiento
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar el modelo
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, app):
        """
        Predice si una app será exitosa (1) o no (0).

        Args:
            app (App): La app a evaluar

        Returns:
            int: 1 si se predice éxito, 0 si se predice fracaso
        """
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado. Llama a train() primero.")

        features = np.array([app.to_features()])
        features_scaled = self.scaler.transform(features)

        return int(self.model.predict(features_scaled)[0])

    def predict_proba(self, app):
        """
        Calcula la probabilidad de éxito de una app.

        Args:
            app (App): La app a evaluar

        Returns:
            float: Probabilidad de éxito entre 0 y 1
        """
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado. Llama a train() primero.")

        features = np.array([app.to_features()])
        features_scaled = self.scaler.transform(features)

        # Obtener la probabilidad de la clase 1 (éxito)
        return float(self.model.predict_proba(features_scaled)[0][1])


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento
    apps = [
        App("FastChat", 10000, 12.5, 0.65, 1500, 1),
        App("FitTrack", 500, 5.0, 0.2, 50, 0),
        App("GameHub", 15000, 25.0, 0.75, 3000, 1),
        App("BudgetBuddy", 800, 6.5, 0.3, 80, 0),
        App("EduFlash", 12000, 18.0, 0.7, 2200, 1),
        App("NoteKeeper", 600, 4.0, 0.15, 30, 0)
    ]

    dataset = AppDataset(apps)
    predictor = SuccessPredictor()
    predictor.train(dataset)

    # Nueva app a evaluar
    new_app = App("StudyBoost", 20000, 15.0, 0.5, 700)
    predicted_success = predictor.predict(new_app)
    prob = predictor.predict_proba(new_app)

    print(f"¿Será exitosa la app {new_app.app_name}? {'Sí' if predicted_success else 'No'}")
    print(f"Probabilidad estimada de éxito: {prob:.2f}")