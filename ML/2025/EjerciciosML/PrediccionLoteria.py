import numpy as np  # Importamos NumPy para generar números aleatorios y manipular arrays
import pandas as pd  # Importamos Pandas para manejar estructuras de datos tipo DataFrame
from sklearn.ensemble import RandomForestClassifier  # Importamos el clasificador Random Forest
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score  # Para calcular la precisión del modelo


def generar_series(num_series):
    """
    Genera una lista de combinaciones aleatorias de 6 números entre 1 y 49 (inclusive).

    Parámetros:
    - num_series (int): Número de combinaciones a generar.

    Retorna:
    - np.array: Un array de NumPy con las combinaciones generadas.
    """
    series = []  # Lista donde se guardarán las combinaciones
    for _ in range(num_series):  # Repetimos el proceso num_series veces
        # np.random.choice genera una muestra aleatoria sin reemplazo (replace=False) de 6 números únicos del 1 al 49
        combinacion = np.random.choice(range(1, 50), size=6, replace=False)
        # np.sort ordena los números de menor a mayor para uniformidad
        series.append(np.sort(combinacion))
    return np.array(series)  # Convertimos la lista a un array de NumPy para trabajar más eficientemente


def entrenar_modelo():
    """
    Simula datos de combinaciones de lotería y entrena un modelo de clasificación para predecir el éxito.

    Retorna:
    - modelo entrenado (RandomForestClassifier)
    """
    num_series = 1000  # Número de combinaciones a generar
    series = generar_series(num_series)  # Generamos las combinaciones
    etiquetas = []  # Lista vacía para guardar las etiquetas de éxito o fracaso

    for _ in range(num_series):  # Generamos 1000 etiquetas (una para cada combinación)
        # np.random.rand() genera un número entre 0 y 1. Si es menor que 0.1, etiquetamos como éxito (1)
        exito = 1 if np.random.rand() < 0.1 else 0
        etiquetas.append(exito)

    # Creamos un DataFrame con las combinaciones y asignamos nombres a las columnas
    df = pd.DataFrame(series, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'num6'])
    df['etiqueta'] = etiquetas  # Añadimos una columna de etiquetas (éxito o fracaso)

    # Separamos las características (X) y las etiquetas (y)
    X = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]
    y = df['etiqueta']

    # Dividimos los datos en conjuntos de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos el modelo de Random Forest (árboles de decisión múltiples combinados)
    modelo = RandomForestClassifier(random_state=42)  # random_state asegura reproducibilidad
    modelo.fit(X_train, y_train)  # Entrenamos el modelo con los datos de entrenamiento

    # Realizamos predicciones con los datos de prueba
    y_pred = modelo.predict(X_test)

    # Calculamos y mostramos la precisión de las predicciones
    print("Precisión del modelo:", accuracy_score(y_test, y_pred))

    return modelo  # Devolvemos el modelo ya entrenado


def predecir_mejor_serie(modelo, num_series):
    """
    Usa el modelo entrenado para predecir cuál combinación nueva de lotería tiene más probabilidad de éxito.

    Parámetros:
    - modelo: El modelo previamente entrenado.
    - num_series (int): Número de combinaciones nuevas a generar.

    Retorna:
    - mejor_combinacion (array): La combinación con mayor probabilidad de éxito.
    - mejor_probabilidad (float): La probabilidad de éxito correspondiente.
    """
    series = generar_series(num_series)  # Generamos nuevas combinaciones

    # Obtenemos las probabilidades predichas para cada combinación.
    # predict_proba devuelve un array con 2 columnas: prob. clase 0 (fracaso) y clase 1 (éxito)
    probabilidades = modelo.predict_proba(series)[:, 1]  # Seleccionamos la columna de probabilidad de éxito

    # np.argmax devuelve el índice del valor máximo en el array
    mejor_indice = np.argmax(probabilidades)
    mejor_combinacion = series[mejor_indice]  # Combinación con la mayor probabilidad
    mejor_probabilidad = probabilidades[mejor_indice]

    return mejor_combinacion, mejor_probabilidad  # Devolvemos ambos resultados


# Punto de entrada principal del script
if __name__ == "__main__":
    # Entrenamos el modelo
    modelo = entrenar_modelo()

    # Elegimos cuántas combinaciones nuevas queremos analizar
    num_series = 10
    mejor_combinacion, mejor_probabilidad = predecir_mejor_serie(modelo, num_series)

    print(f"\nLa mejor combinación generada es: {mejor_combinacion}")
    print(f"Probabilidad de éxito: {mejor_probabilidad:.4f}")
