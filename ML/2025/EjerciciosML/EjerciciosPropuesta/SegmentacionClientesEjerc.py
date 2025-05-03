import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


class CustomerSegmentationModel:
    """Modelo para segmentar clientes y predecir compras futuras"""
    
    def __init__(self, df):
        # Ya inicializado para ti
        self.df = df.copy()
        self.data = self.df.copy()
        self.model = None
        self.accuracy = None
        self.conf_matrix = None

    def segment_customers(self, n_clusters=3):
        """
        TODO: Implementa la segmentación de clientes usando K-Means
        Hint: Usa StandardScaler y KMeans
        """
        # TODO: Selecciona las columnas relevantes
        # Hint: ["total_spent", "total_purchases", "purchase_frequency"]
        
        # TODO: Normaliza los datos
        # Hint: Usa StandardScaler()
        
        # TODO: Aplica K-Means
        # Hint: KMeans(n_clusters=n_clusters)
        
        # TODO: Guarda los segmentos en el DataFrame
        pass

    def train_model(self):
        """
        TODO: Entrena un modelo de regresión logística
        Hint: Usa LogisticRegression y train_test_split
        """
        # TODO: Verifica si existe la segmentación
        if "customer_segment" not in self.df.columns:
            self.segment_customers()
        
        # TODO: Prepara X (características) e y (objetivo)
        # Hint: X debe incluir total_spent, total_purchases, 
        # purchase_frequency y customer_segment
        
        # TODO: Divide los datos en train y test
        # TODO: Entrena el modelo y calcula métricas
        pass

    def get_accuracy(self):
        """Ya implementado para ti"""
        return self.accuracy

    def get_confusion_matrix(self):
        """Ya implementado para ti"""
        return self.conf_matrix


# Código de ejemplo
if __name__ == "__main__":
    # Generar datos de ejemplo
    print("Generando datos de clientes...")
    np.random.seed(42)
    n_customers = 500
    
    data = pd.DataFrame({
        'total_spent': np.random.uniform(100, 5000, n_customers),
        'total_purchases': np.random.randint(1, 100, n_customers),
        'purchase_frequency': np.random.uniform(1, 30, n_customers),
        'will_buy_next_month': np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
    })
    
    # Crear y entrenar modelo
    print("\nCreando modelo...")
    modelo = CustomerSegmentationModel(data)
    
    # Segmentar clientes
    print("\nSegmentando clientes...")
    modelo.segment_customers(n_clusters=3)
    
    # Ver resultados de segmentación
    print("\nEjemplo de segmentación:")
    print(modelo.data[['total_spent', 'total_purchases', 
                      'purchase_frequency', 'customer_segment']].head())
    
    # Entrenar modelo predictivo
    print("\nEntrenando modelo predictivo...")
    modelo.train_model()
    
    # Mostrar resultados
    print(f"\nPrecisión del modelo: {modelo.get_accuracy():.2f}")
    print("\nMatriz de Confusión:")
    print(modelo.get_confusion_matrix())