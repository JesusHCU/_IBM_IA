import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


class CustomerSegmentationModel:
    def __init__(self, df):
        """
        Constructor del modelo.

        Parámetros:
            df (pd.DataFrame): DataFrame con la información de los clientes.

        Se crea una copia del DataFrame para no modificar el original.
        Inicializa variables internas para almacenar el modelo, la precisión y la matriz de confusión.
        """
        self.df = df.copy()
        self.data = self.df.copy()  # Esto lo necesitan los tests para validar actualizaciones
        self.model = None
        self.accuracy = None
        self.conf_matrix = None

    def segment_customers(self, n_clusters=3):
        """
        Segmenta a los clientes utilizando el algoritmo K-Means.

        Parámetros:
            n_clusters (int): Número de clústeres (por defecto 3).

        Procedimiento:
        - Se seleccionan las columnas 'total_spent', 'total_purchases' y 'purchase_frequency'.
        - Se normalizan estos datos usando StandardScaler para que tengan media 0 y varianza 1.
        - Se aplica K-Means para agrupar los datos en el número de clústeres indicado.
        - Se añade la columna 'customer_segment' al DataFrame con las etiquetas obtenidas.
        """
        scaler = StandardScaler()
        # Seleccionar las columnas relevantes
        X = self.df[["total_spent", "total_purchases", "purchase_frequency"]]
        # Normalizar los datos
        X_scaled = scaler.fit_transform(X)

        # Aplicar el algoritmo K-Means con un número fijo de inicializaciones para estabilidad
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        segments = kmeans.fit_predict(X_scaled)

        # Agregar la columna con el segmento asignado a cada cliente
        self.df["customer_segment"] = segments
        self.data = self.df.copy()  # Actualizamos self.data para que los tests lo puedan validar

    def train_model(self):
        """
        Entrena un modelo de Regresión Logística para predecir la variable 'will_buy_next_month'.

        Procedimiento:
        - Verifica si ya se ha realizado la segmentación de clientes; de lo contrario, la ejecuta.
        - Define las variables de entrada (X) incluyendo la segmentación y la variable objetivo (y).
        - Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%) de forma aleatoria.
        - Entrena el modelo de Regresión Logística y calcula la precisión y la matriz de confusión.
        """
        # Si la segmentación aún no se ha agregado, ejecutamos el método correspondiente.
        if "customer_segment" not in self.df.columns:
            self.segment_customers()

        # Definir variables de entrada (X) y salida (y)
        X = self.df[["total_spent", "total_purchases", "purchase_frequency", "customer_segment"]]
        y = self.df["will_buy_next_month"]

        # Dividir el dataset en entrenamiento y prueba (80%/20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instanciar y entrenar el modelo de Regresión Logística
        # Se ha indicado max_iter=500 para asegurar que el algoritmo converja en poco tiempo.
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X_train, y_train)

        # Realizar predicciones y calcular métricas de evaluación
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.conf_matrix = confusion_matrix(y_test, y_pred)

    def get_accuracy(self):
        """
        Devuelve la precisión (accuracy) del modelo entrenado.

        Retorno:
            float: Precisión del modelo.
        """
        return self.accuracy

    def get_confusion_matrix(self):
        """
        Devuelve la matriz de confusión del modelo entrenado.

        Retorno:
            np.array: Matriz de confusión resultante de comparar las predicciones con los valores reales.
        """
        return self.conf_matrix


# ----------------------------------------------------------------------------
# Ejemplo de uso del modelo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Generar un conjunto de datos sintético para 500 clientes.
    np.random.seed(42)  # Fijar la semilla para reproducibilidad
    n_customers = 500

    data = pd.DataFrame({
        'total_spent': np.random.uniform(100, 5000, n_customers),  # Gasto total entre $100 y $5000
        'total_purchases': np.random.randint(1, 100, n_customers),  # Número de compras entre 1 y 100
        'purchase_frequency': np.random.uniform(1, 30, n_customers),  # Frecuencia de compra entre 1 y 30 días
        'will_buy_next_month': np.random.choice([0, 1], n_customers, p=[0.7, 0.3])  # Probabilidad 30% de comprar
    })

    # Crear una instancia del modelo pasando el DataFrame
    segmentation_model = CustomerSegmentationModel(data)

    # Aplicar segmentación de clientes
    segmentation_model.segment_customers(n_clusters=3)
    print("Segmentación completada. Ejemplo de datos con 'customer_segment':")
    print(segmentation_model.data.head())

    # Entrenar el modelo de predicción (Regresión Logística)
    segmentation_model.train_model()
    print("\nEntrenamiento completado.")

    # Mostrar la precisión del modelo y la matriz de confusión
    print(f"Precisión del modelo: {segmentation_model.get_accuracy():.2f}")
    print("Matriz de Confusión:")
    print(segmentation_model.get_confusion_matrix())
