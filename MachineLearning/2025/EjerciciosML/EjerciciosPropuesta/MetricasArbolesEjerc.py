import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def graficar_datos(x, y, titulo="Gráfico de Datos"):
    """
    TODO: Implementa la función para crear un gráfico simple
    Hint: Usa plt para crear una visualización básica
    """
    # TODO: Crea una figura con tamaño 10x5
    # Hint: plt.figure(figsize=(10, 5))
    
    # TODO: Dibuja los datos con línea y marcadores
    # Hint: Usa plt.plot() con marker='o'
    
    # TODO: Añade título y etiquetas
    # Hint: Usa plt.title(), plt.xlabel(), plt.ylabel()
    
    # TODO: Muestra el gráfico
    pass


def calcular_metricas(valores_reales, predicciones):
    """
    TODO: Implementa el cálculo de métricas básicas
    Hint: Usa las funciones de sklearn.metrics
    """
    # TODO: Calcula accuracy, matriz de confusión y reporte
    # Hint: Usa accuracy_score, confusion_matrix y classification_report
    resultados = {
        "accuracy": None,
        "matriz_confusion": None,
        "reporte": None
    }
    return resultados


def entrenar_arbol_decision(X_train, y_train):
    """
    TODO: Implementa el entrenamiento del árbol de decisión
    Hint: Usa DecisionTreeClassifier
    """
    # TODO: Crea y entrena el modelo
    # Hint: Usa random_state=42 para reproducibilidad
    modelo = None
    return modelo


# Código de prueba
if __name__ == "__main__":
    # Datos de ejemplo para graficar
    dias = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie']
    valores = [22, 24, 23, 25, 26]
    
    # TODO: Prueba la función de gráfico
    print("Creando gráfico de prueba...")
    graficar_datos(dias, valores, "Datos de Prueba")
    
    # Cargar y preparar datos de Iris
    print("\nCargando dataset Iris...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # TODO: Entrena el modelo y calcula métricas
    print("\nEntrenando modelo...")
    modelo = entrenar_arbol_decision(X_train, y_train)
    predicciones = modelo.predict(X_test)
    
    # TODO: Muestra los resultados
    metricas = calcular_metricas(y_test, predicciones)
    print("\nResultados:")
    print(f"Precisión: {metricas['accuracy']}")
    print("\nMatriz de Confusión:")
    print(metricas['matriz_confusion'])

plt.figure()  # Crear figura
plt.plot()    # Dibujar línea
plt.title()   # Añadir título
plt.show()    # Mostrar gráfico

accuracy = accuracy_score(reales, predicciones)
matriz = confusion_matrix(reales, predicciones)
reporte = classification_report(reales, predicciones)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

