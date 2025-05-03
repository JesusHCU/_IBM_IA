# svm_digits.py

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import unittest


def entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo SVM y evalúa su rendimiento.

    Parámetros:
    X_train (array-like): Características de entrenamiento.
    y_train (array-like): Etiquetas de entrenamiento.
    X_test (array-like): Características de prueba.
    y_test (array-like): Etiquetas de prueba.

    Retorna:
    dict: Diccionario con los resultados del modelo.
    """
    # Entrenar el modelo SVM
    modelo = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    modelo.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)

    # Retornar resultados en un diccionario
    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte
    }


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el dataset de dígitos escritos a mano
    digits = load_digits()
    X = digits.data  # Características (matriz de píxeles)
    y = digits.target  # Etiquetas (números del 0 al 9)

    # Dividir en conjunto de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Llamar a la función y obtener las métricas
    resultados = entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test)

    # Mostrar los resultados
    print("Precisión del modelo:", resultados["accuracy"])
    print("Matriz de Confusión:\n", resultados["matriz_confusion"])
    print("Reporte de Clasificación:\n", resultados["reporte"])


# Pruebas unitarias
class TestSVM(unittest.TestCase):
    def setUp(self):
        # Cargar el dataset de dígitos
        self.digits = load_digits()
        self.X = self.digits.data
        self.y = self.digits.target

        # Dividir en conjunto de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_entrenar_y_evaluar_svm(self):
        # Llamar a la función
        resultados = entrenar_y_evaluar_svm(self.X_train, self.y_train, self.X_test, self.y_test)

        # Verificar que los resultados son del tipo correcto
        self.assertIsInstance(resultados, dict)
        self.assertIn("predicciones", resultados)
        self.assertIn("accuracy", resultados)
        self.assertIn("matriz_confusion", resultados)
        self.assertIn("reporte", resultados)

        # Verificar que la precisión es mayor o igual al 90%
        self.assertGreaterEqual(resultados["accuracy"], 0.90)

        # Verificar que la matriz de confusión tiene la forma correcta
        self.assertEqual(resultados["matriz_confusion"].shape, (10, 10))


# Ejecutar las pruebas unitarias
if __name__ == "__main__":
    unittest.main()