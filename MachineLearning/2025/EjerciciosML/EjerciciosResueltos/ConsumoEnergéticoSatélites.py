import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class Satellite:
    def __init__(self, duracion_mision, paneles_sol, carga_util, consumo_diario):
        """
        Representa un satélite con sus características principales.

        Args:
            duracion_mision (int): Duración de la misión en días
            paneles_sol (float): Superficie total de paneles solares en m²
            carga_util (float): Peso de instrumentos y sensores en kg
            consumo_diario (float): Energía consumida por día en kWh
        """
        self.duracion_mision = duracion_mision
        self.paneles_sol = paneles_sol
        self.carga_util = carga_util
        self.consumo_diario = consumo_diario

    def to_dict(self):
        """Convierte el objeto Satellite a un diccionario"""
        return {
            "duracion_mision_dias": self.duracion_mision,
            "paneles_sol": self.paneles_sol,
            "carga_util": self.carga_util,
            "consumo_diario": self.consumo_diario
        }


class SatelliteDatasetGenerator:
    def __init__(self, n=300):
        """
        Genera datos sintéticos de satélites.

        Args:
            n (int): Número de satélites a generar (default: 300)
        """
        self.n = n

    def generate(self):
        """Genera una lista de objetos Satellite con datos sintéticos"""
        np.random.seed(42)  # Para reproducibilidad
        duraciones = np.random.randint(100, 1000, self.n)
        paneles = np.random.uniform(10, 100, self.n)
        cargas = np.random.uniform(200, 2000, self.n)
        consumo = 5 + 0.01 * duraciones + 0.002 * cargas + np.random.normal(0, 1, self.n)

        return [Satellite(d, p, c, e) for d, p, c, e in zip(duraciones, paneles, cargas, consumo)]


class SatelliteDataProcessor:
    def __init__(self, satellites):
        """
        Procesa los datos de satélites y los convierte a DataFrame.

        Args:
            satellites (list): Lista de objetos Satellite
        """
        self.df = pd.DataFrame([sat.to_dict() for sat in satellites])
        self.df["eficiencia_energia"] = self.df["consumo_diario"] / self.df["paneles_sol"]

    def get_dataframe(self):
        """Devuelve el DataFrame procesado"""
        return self.df


class EnergyConsumptionRegressor:
    def __init__(self):
        """Modelo para predecir el consumo energético"""
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Entrena el modelo de regresión lineal

        Args:
            X (DataFrame): Variables independientes
            y (Series): Variable dependiente

        Returns:
            array: Predicciones del modelo
        """
        self.model.fit(X, y)
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Evalúa el modelo con R²

        Args:
            y_true (array): Valores reales
            y_pred (array): Valores predichos

        Returns:
            float: Coeficiente de determinación R²
        """
        return r2_score(y_true, y_pred)

    def get_coefficients(self):
        """
        Obtiene los coeficientes del modelo

        Returns:
            tuple: (coeficiente, intercepto)
        """
        return self.model.coef_, self.model.intercept_


class SatellitePlotter:
    def __init__(self, df, y_pred):
        """
        Visualiza los resultados del análisis

        Args:
            df (DataFrame): Datos de los satélites
            y_pred (array): Predicciones del modelo
        """
        self.df = df
        self.y_pred = y_pred

    def plot(self):
        """Genera el gráfico de dispersión con la línea de regresión"""
        plt.figure(figsize=(10, 6))

        # Gráfico de dispersión con color por carga útil
        scatter = plt.scatter(
            self.df["duracion_mision_dias"],
            self.df["consumo_diario"],
            c=self.df["carga_util"],
            cmap="viridis",
            alpha=0.7,
            label="Datos reales"
        )

        # Línea de regresión
        plt.plot(
            self.df["duracion_mision_dias"],
            self.y_pred,
            color="red",
            linewidth=2,
            label="Regresión lineal"
        )

        # Configuración del gráfico
        plt.title("🛰️ Consumo energético vs. duración de misión", fontsize=14)
        plt.xlabel("Duración de la misión (días)", fontsize=12)
        plt.ylabel("Consumo diario (kWh)", fontsize=12)

        # Barra de color para la carga útil
        cbar = plt.colorbar(scatter)
        cbar.set_label("Carga útil (kg)", fontsize=12)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


class SatelliteAnalysisExample:
    def run(self):
        """Ejecuta todo el flujo de análisis"""
        print("=== Generando datos de satélites ===")
        generator = SatelliteDatasetGenerator(n=500)
        satellites = generator.generate()

        print("\n=== Procesando datos ===")
        processor = SatelliteDataProcessor(satellites)
        df = processor.get_dataframe()
        print("\nMuestra de los datos generados:")
        print(df.head())

        print("\n=== Entrenando modelo de regresión ===")
        X = df[["duracion_mision_dias"]]
        y = df["consumo_diario"]

        regressor = EnergyConsumptionRegressor()
        y_pred = regressor.fit(X, y)

        print("\n=== Evaluando modelo ===")
        r2 = regressor.evaluate(y, y_pred)
        coef, intercept = regressor.get_coefficients()
        print(f"\n📈 Modelo lineal: consumo_diario = {coef[0]:.4f} * duracion_mision + {intercept:.2f}")
        print(f"🔍 Coeficiente de determinación (R²): {r2:.4f}")

        print("\n=== Visualizando resultados ===")
        plotter = SatellitePlotter(df, y_pred)
        plotter.plot()


# Ejecutar el análisis completo
if __name__ == "__main__":
    analysis = SatelliteAnalysisExample()
    analysis.run()