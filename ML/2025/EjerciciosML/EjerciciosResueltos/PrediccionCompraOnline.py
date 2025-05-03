import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Función para generar datos de comportamiento de los usuarios
def generar_datos_compras(num_muestras):
    np.random.seed(42)

    # Inicializar listas para las características y etiquetas
    X = []  # Características: [num_paginas_vistas, tiempo_en_sitio]
    y = []  # Etiquetas: 1 (compra) o 0 (no compra)

    for _ in range(num_muestras):
        num_paginas_vistas = np.random.randint(1, 21)  # Número de páginas vistas entre 1 y 20
        tiempo_en_sitio = np.random.uniform(0, 30)  # Tiempo en sitio entre 0 y 30 minutos

        # Etiqueta 1 (compra) si el usuario vio más de 5 páginas y pasó más de 10 minutos
        if num_paginas_vistas > 5 and tiempo_en_sitio > 10:
            y.append(1)
        else:
            y.append(0)

        X.append([num_paginas_vistas, tiempo_en_sitio])  # Añadir características

    return np.array(X), np.array(y)


# Función para entrenar el modelo de clasificación
def entrenar_modelo(X, y):
    # Dividir los datos en conjunto de entrenamiento y prueba (70% entrenamiento, 30% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Usar regresión logística como modelo de clasificación
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)  # Entrenar el modelo

    # Evaluar el modelo en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision * 100:.2f}%")

    return modelo


# Función para predecir la compra de un nuevo usuario
def predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):
    # Hacer una predicción para un nuevo usuario
    prediccion = modelo.predict([[num_paginas_vistas, tiempo_en_sitio]])

    # Devolver el mensaje adecuado según la predicción
    if prediccion == 1:
        return 'El usuario comprará el producto.'
    else:
        return 'El usuario no comprará el producto.'


# Función para evaluar el modelo
def evaluar_modelo(modelo, X, y):
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hacer predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Calcular la precisión del modelo
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo en el conjunto de prueba: {precision * 100:.2f}%")

    return precision


# Generar un conjunto de datos de usuarios
num_muestras = 1000  # Puedes ajustar este número para generar más o menos datos
X, y = generar_datos_compras(num_muestras)

# Entrenar el modelo
modelo = entrenar_modelo(X, y)

# Evaluar el modelo
precision = evaluar_modelo(modelo, X, y)

# Realizar una predicción para un nuevo usuario
num_paginas_vistas = 8  # Ejemplo: el usuario vio 8 páginas
tiempo_en_sitio = 12  # Ejemplo: el usuario pasó 12 minutos en el sitio

mensaje_prediccion = predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio)

# Mostrar el mensaje de predicción
print(mensaje_prediccion)
