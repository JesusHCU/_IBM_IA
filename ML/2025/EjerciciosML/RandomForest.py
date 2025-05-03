import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def graficar_temperaturas(dias, temperaturas):
    plt.figure(figsize=(10, 5))  # Tamaño del gráfico
    plt.plot(dias, temperaturas, marker='o', linestyle='--', color='b', markersize=8, linewidth=2, label='Temperatura')

    # Etiquetas y título
    plt.xlabel("Días")
    plt.ylabel("Temperatura (°C)")
    plt.title("Temperaturas Semanales")

    # Añadir leyenda
    plt.legend()

    # Mostrar la cuadrícula para mejor visualización
    plt.grid(True, linestyle='--', alpha=0.6)

    # Mostrar la gráfica
    plt.show()


def analizar_finanzas(ingresos, gastos):
    ingresos = np.array(ingresos)
    gastos = np.array(gastos)

    balance_mensual = ingresos - gastos
    total_ingresos = np.sum(ingresos)
    total_gastos = np.sum(gastos)
    saldo_final = total_ingresos - total_gastos

    return [balance_mensual.tolist(), total_ingresos, total_gastos, saldo_final]


def entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test):
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)

    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte
    }


def entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)

    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte
    }


# Datos proporcionados
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [22, 24, 23, 25, 26, 28, 27]

graficar_temperaturas(dias, temperaturas)

# Datos de finanzas
ingresos = [1500, 1600, 1700, 1650, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
gastos = [1000, 1100, 1200, 1150, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

resultado = analizar_finanzas(ingresos, gastos)
print(resultado)

# Cargar el dataset de Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Clases de las flores

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar y evaluar Árbol de Decisión
resultados = entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test)
print("Precisión del modelo Árbol de Decisión:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])

# Cargar el dataset de Vinos
wine = load_wine()
X_wine = wine.data  # Características
y_wine = wine.target  # Clases de vinos

# Dividir en conjunto de entrenamiento y prueba
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Entrenar y evaluar Random Forest
resultados_rf = entrenar_y_evaluar_random_forest(X_train_wine, y_train_wine, X_test_wine, y_test_wine)
print("Precisión del modelo Random Forest:", resultados_rf["accuracy"])
print("Matriz de Confusión:\n", resultados_rf["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados_rf["reporte"])
