import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def generar_datos_emails(num_muestras):
    """
    TODO: Implementa la generación de datos de entrenamiento
    Hint: Usa np.random para generar características aleatorias
    """
    np.random.seed(42)  # Mantenemos la semilla para reproducibilidad
    
    # TODO: Inicializa las listas para características (X) y etiquetas (y)
    X = []  # [longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces]
    y = []  # 1 (Spam) o 0 (No Spam)

    # TODO: Genera num_muestras ejemplos
    # Hint: Usa estos rangos:
    # - longitud_mensaje: entre 50 y 500 caracteres
    # - frecuencia_palabra_clave: entre 0 y 1
    # - cantidad_enlaces: entre 0 y 10
    
    return np.array(X), np.array(y)


def entrenar_modelo_svm(X, y):
    """
    TODO: Implementa el entrenamiento del modelo SVM
    Hint: Usa train_test_split y SVC
    """
    # TODO: Divide los datos en conjuntos de entrenamiento y prueba
    # Hint: Usa test_size=0.3 y random_state=42
    
    # TODO: Crea y entrena el modelo SVM
    # Hint: Usa kernel='linear'
    
    # TODO: Evalúa y muestra la precisión del modelo
    
    return modelo


def predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces):
    """
    TODO: Implementa la función de predicción
    Hint: Usa el método predict del modelo
    """
    # TODO: Realiza la predicción con el modelo
    # TODO: Retorna un mensaje indicando si es spam o no
    pass


# Código de prueba
if __name__ == "__main__":
    # TODO: Genera 1000 muestras de entrenamiento
    num_muestras = 1000
    
    # TODO: Entrena el modelo
    
    # TODO: Prueba el modelo con un nuevo email
    # Ejemplo sugerido:
    # longitud_mensaje = 300
    # frecuencia_palabra_clave = 0.7
    # cantidad_enlaces = 5