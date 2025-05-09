import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# TODO: Completa la función para graficar las temperaturas
def graficar_temperaturas(dias, temperaturas):
    # ### YOUR CODE HERE ###
    # 1. Crear una figura con plt.figure()
    # 2. Realizar el plot con los datos
    # 3. Añadir título y etiquetas
    # 4. Configurar la leyenda
    # 5. Mostrar la gráfica
    pass

# TODO: Implementa el análisis financiero
def analizar_finanzas(ingresos, gastos):
    # ### YOUR CODE HERE ###
    # 1. Convertir listas a arrays de numpy
    # 2. Calcular balance mensual
    # 3. Calcular totales
    # 4. Retornar resultados
    pass

# TODO: Implementa el entrenamiento y evaluación del árbol de decisión
def entrenar_y_evaluar_arbol(X_train, X_test, y_train, y_test):
    # ### YOUR CODE HERE ###
    # 1. Crear y entrenar el modelo
    # 2. Realizar predicciones
    # 3. Calcular y retornar la precisión
    pass

# Datos de prueba
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [22, 24, 23, 25, 26, 28, 27]
ingresos = [1500, 1600, 1700, 1650, 1800]
gastos = [1000, 1100, 1200, 1150, 1300]

# Test del código
if __name__ == "__main__":
    # 1. Probar visualización
    graficar_temperaturas(dias, temperaturas)
    
    # 2. Probar análisis financiero
    resultado_financiero = analizar_finanzas(ingresos, gastos)
    print("Resultado financiero:", resultado_financiero)
    
    # 3. Probar modelo de clasificación
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    precision = entrenar_y_evaluar_arbol(X_train, X_test, y_train, y_test)
    print(f"Precisión del modelo: {precision:.2f}")