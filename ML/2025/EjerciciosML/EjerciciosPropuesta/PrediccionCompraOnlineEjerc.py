import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def generar_datos_compras(num_muestras):
    """
    TODO: Genera datos simulados de usuarios
    Hint: Usa np.random para generar datos aleatorios
    """
    np.random.seed(42)  # Para reproducibilidad
    
    # TODO: Inicializa las listas para guardar los datos
    X = []  # [num_paginas_vistas, tiempo_en_sitio]
    y = []  # 1 (compra) o 0 (no compra)
    
    for _ in range(num_muestras):
        # TODO: Genera datos aleatorios para cada usuario
        # Hint: páginas_vistas entre 1-20, tiempo entre 0-30 minutos
        num_paginas_vistas = None
        tiempo_en_sitio = None
        
        # TODO: Determina si el usuario compra
        # Hint: compra = 1 si páginas > 5 y tiempo > 10
        compra = None
        
        # Añade los datos a las listas
        X.append([num_paginas_vistas, tiempo_en_sitio])
        y.append(compra)
    
    return np.array(X), np.array(y)


# Esta función ya está implementada para ti
def entrenar_modelo(X, y):
    """Entrena el modelo de regresión logística"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision * 100:.2f}%")
    
    return modelo


def predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):
    """
    TODO: Implementa la predicción para un nuevo usuario
    Hint: Usa modelo.predict() y retorna un mensaje apropiado
    """
    # TODO: Realiza la predicción
    # Hint: La predicción será 0 o 1
    prediccion = None
    
    # TODO: Retorna el mensaje adecuado
    # Hint: "El usuario comprará/no comprará el producto"
    return ""


# Código de prueba
if __name__ == "__main__":
    # Generar datos
    print("Generando datos de usuarios...")
    X, y = generar_datos_compras(1000)
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    modelo = entrenar_modelo(X, y)
    
    # Probar con nuevos usuarios
    print("\nProbando predicciones...")
    usuarios_prueba = [
        (8, 15),  # Probable comprador
        (3, 5),   # Probable no comprador
        (10, 20)  # Probable comprador
    ]
    
    for paginas, tiempo in usuarios_prueba:
        print(f"\nUsuario con {paginas} páginas vistas y {tiempo} minutos:")
        prediccion = predecir_compra(modelo, paginas, tiempo)
        print(prediccion)