import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def graficar_datos(x, y, titulo="Gráfico de Datos"):
    """
    TODO: Implementa la función para visualizar datos
    Hint: Usa plt para crear un gráfico de líneas
    """
    # TODO: Configura el tamaño del gráfico
    plt.figure(figsize=(10, 5))
    
    # TODO: Crea el gráfico de líneas
    # Hint: Usa plt.plot() con marker='o'
    
    # TODO: Añade título y etiquetas
    # Hint: Usa plt.title(), plt.xlabel(), plt.ylabel()
    
    plt.grid(True)
    plt.show()


def analizar_datos_financieros(ingresos, gastos):
    """
    Función ya implementada para ti
    Analiza ingresos y gastos mensuales
    """
    ingresos = np.array(ingresos)
    gastos = np.array(gastos)
    
    balance = ingresos - gastos
    total_ingresos = np.sum(ingresos)
    total_gastos = np.sum(gastos)
    saldo = total_ingresos - total_gastos
    
    return balance.tolist(), total_ingresos, total_gastos, saldo


def entrenar_modelo_basico(X_train, y_train):
    """
    TODO: Implementa el entrenamiento del modelo básico
    Hint: Usa DecisionTreeClassifier
    """
    # TODO: Crea y entrena el modelo
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_random_forest(X_train, y_train):
    """
    TODO: Implementa el entrenamiento del Random Forest
    Hint: Usa RandomForestClassifier con 100 árboles
    """
    # TODO: Crea y entrena el modelo
    modelo = None
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    """
    TODO: Implementa la evaluación del modelo
    Hint: Calcula accuracy y matriz de confusión
    """
    # TODO: Realiza predicciones
    predicciones = modelo.predict(X_test)
    
    # TODO: Calcula métricas
    accuracy = accuracy_score(y_test, predicciones)
    matriz = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)
    
    return {
        "accuracy": accuracy,
        "matriz_confusion": matriz,
        "reporte": reporte
    }


# Código de prueba
if __name__ == "__main__":
    # 1. Datos de ejemplo para gráfico
    dias = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie']
    temperaturas = [22, 24, 23, 25, 26]
    
    print("Creando gráfico...")
    graficar_datos(dias, temperaturas, "Temperaturas Semanales")
    
    # 2. Cargar dataset Iris
    print("\nCargando datos de Iris...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # 3. Entrenar y evaluar modelos
    print("\nEntrenando modelos...")
    
    # Árbol de decisión básico
    modelo_basico = entrenar_modelo_basico(X_train, y_train)
    resultados_basico = evaluar_modelo(modelo_basico, X_test, y_test)
    
    # Random Forest
    modelo_rf = entrenar_random_forest(X_train, y_train)
    resultados_rf = evaluar_modelo(modelo_rf, X_test, y_test)
    
    # Mostrar resultados
    print("\nResultados Árbol de Decisión:")
    print(f"Precisión: {resultados_basico['accuracy']:.2f}")
    
    print("\nResultados Random Forest:")
    print(f"Precisión: {resultados_rf['accuracy']:.2f}")