from sklearn.linear_model import LinearRegression
import numpy as np

class App:
    """
    Representa una aplicación móvil con sus atributos.
    """
    def __init__(self, name, downloads, rating, size_mb, reviews, revenue=None):
        self.name = name
        self.downloads = downloads
        self.rating = rating
        self.size_mb = size_mb
        self.reviews = reviews
        self.revenue = revenue

class RevenuePredictor:
    """
    Clase para predecir los ingresos de una aplicación utilizando regresión lineal.
    """
    def __init__(self):
        self.model = LinearRegression()

    def extract_features(self, apps):
        """
        Extrae las características relevantes de una lista de objetos App.
        """
        features = []
        for app in apps:
            features.append([app.downloads, app.rating, app.size_mb, app.reviews])
        return np.array(features)

    def extract_target(self, apps):
        """
        Extrae la variable objetivo (ingresos) de una lista de objetos App.
        """
        target = []
        for app in apps:
            if app.revenue is not None:
                target.append(app.revenue)
        return np.array(target)

    def fit(self, training_apps):
        """
        Entrena el modelo de regresión lineal con los datos proporcionados.
        """
        X_train = self.extract_features(training_apps)
        y_train = self.extract_target(training_apps)
        self.model.fit(X_train, y_train)

    def predict(self, new_app):
        """
        Predice los ingresos de una nueva app utilizando el modelo entrenado.
        """
        new_features = np.array([[new_app.downloads, new_app.rating, new_app.size_mb, new_app.reviews]])
        predicted_revenue = self.model.predict(new_features)
        return predicted_revenue[0]

# Datos simulados de entrenamiento
training_apps = [
    App("TaskPro", 200, 4.2, 45.0, 1800, 120.0),
    App("MindSpark", 150, 4.5, 60.0, 2100, 135.0),
    App("WorkFlow", 300, 4.1, 55.0, 2500, 160.0),
    App("ZenTime", 120, 4.8, 40.0, 1700, 140.0),
    App("FocusApp", 180, 4.3, 52.0, 1900, 130.0),
    App("BoostApp", 220, 4.0, 48.0, 2300, 145.0),
]

# Creamos y entrenamos el predictor
predictor = RevenuePredictor()
predictor.fit(training_apps)

# Nueva app para predecir
new_app = App("FocusMaster", 250, 4.5, 50.0, 3000)
predicted_revenue = predictor.predict(new_app)

print(f"Ingresos estimados para {new_app.name}: ${predicted_revenue:.2f}K")