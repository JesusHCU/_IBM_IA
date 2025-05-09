from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import unittest


def entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test):
    """
    TODO: Implementa el entrenamiento y evaluación del modelo SVM
    
    Parámetros:
    X_train, y_train: Datos de entrenamiento
    X_test, y_test: Datos de prueba
    
    Retorna:
    dict: Diccionario con métricas y predicciones
    """
    # TODO: Crea y entrena el modelo SVM
    # Hint: Usa SVC con estos parámetros:
    # - kernel='rbf' (función de base radial)
    # - C=10.0 (parámetro de regularización)
    # - gamma='scale' (coeficiente del kernel)
    # - random_state=42 (reproducibilidad)
    modelo = None  # Reemplaza con tu implementación
    
    # TODO: Realiza predicciones
    # Hint: Usa modelo.predict(X_test)
    predicciones = None  # Reemplaza con tu implementación
    
    # TODO: Calcula las métricas de evaluación
    # Hint: Usa las funciones importadas de sklearn.metrics
    accuracy = None
    matriz_confusion = None
    reporte = None
    
    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte
    }


# Código de ejemplo
if __name__ == "__main__":
    # Cargar y preparar datos
    print("Cargando dataset de dígitos...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Forma del dataset: {X.shape}")
    print(f"Número de clases: {len(set(y))}")
    
    # División de datos
    print("\nDividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenamiento y evaluación
    print("\nEntrenando modelo SVM...")
    resultados = entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test)
    
    # Mostrar resultados
    print("\nResultados del modelo:")
    print(f"Precisión: {resultados['accuracy']:.4f}")
    print("\nMatriz de Confusión:")
    print(resultados["matriz_confusion"])
    print("\nReporte de Clasificación:")
    print(resultados["reporte"])


# Pruebas unitarias básicas
class TestSVM(unittest.TestCase):
    """
    TODO: Implementa las pruebas para verificar tu modelo
    """
    def setUp(self):
        # TODO: Prepara los datos de prueba
        # Hint: Carga un conjunto pequeño de datos
        pass

    def test_entrenar_y_evaluar_svm(self):
        # TODO: Verifica que:
        # 1. Los resultados son un diccionario
        # 2. Contiene todas las claves necesarias
        # 3. La precisión es > 0.90
        pass

modelo = SVC(
    kernel='rbf',    # Función de kernel
    C=10.0,         # Control de regularización
    gamma='scale',   # Coeficiente de kernel
    random_state=42  # Semilla aleatoria
)
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
matriz_confusion = confusion_matrix(y_test, predicciones)
reporte = classification_report(y_test, predicciones)
