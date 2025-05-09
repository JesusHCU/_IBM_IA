import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def generar_datos_frutas(num_muestras):
    # TODO: Implementa la generación de datos
    # Hint: Crea listas vacías para características y etiquetas
    caracteristicas = []
    etiquetas = []
    
    # Hint: Define las frutas disponibles
    frutas = ["Manzana", "Plátano", "Naranja"]
    
    # TODO: Genera datos aleatorios para cada fruta
    # Hint: Para cada fruta, genera:
    # - Manzana: peso entre 120-200g, tamaño entre 7-9cm
    # - Plátano: peso entre 100-150g, tamaño entre 12-20cm
    # - Naranja: peso entre 150-250g, tamaño entre 8-12cm
    
    return np.array(caracteristicas), np.array(etiquetas)

def entrenar_modelo(data):
    # TODO: Implementa el entrenamiento del modelo
    # Hint: Usa el diccionario para convertir nombres a números
    label_map = {"Manzana": 0, "Plátano": 1, "Naranja": 2}
    
    # TODO: Divide los datos en entrenamiento y prueba
    # Hint: Usa train_test_split con test_size=0.2
    
    # TODO: Crea y entrena el modelo KNN
    # Hint: Usa KNeighborsClassifier con n_neighbors=3
    
    return modelo

def predecir_fruta(modelo, peso, tamano):
    # TODO: Implementa la predicción para una nueva fruta
    # Hint: Convierte el peso y tamaño en array numpy
    # Hint: Usa el diccionario inverso para convertir números a nombres
    pass

# Código de prueba
if __name__ == "__main__":
    # TODO: Genera 100 muestras de frutas
    # TODO: Entrena el modelo
    # TODO: Prueba con una nueva fruta
    # Hint: Usa peso=150g y tamaño=10cm como ejemplo
    pass