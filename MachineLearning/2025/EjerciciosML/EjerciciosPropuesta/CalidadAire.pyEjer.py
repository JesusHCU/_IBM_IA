import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# TODO: Implementa la función de regresión logística
def regresion_logistica(datos):
    # 1. Separa las características (X) y la variable objetivo (y)
    # Hint: X debe contener 'Edad' y 'Colesterol', y debe contener 'Enfermedad'
    
    # 2. Divide los datos en conjuntos de entrenamiento y prueba
    # Hint: Usa train_test_split con test_size=0.3
    
    # 3. Crea y entrena el modelo
    # Hint: Usa LogisticRegression()
    
    # 4. Realiza predicciones
    # Hint: Usa el método predict()
    
    # 5. Calcula y muestra la precisión
    # Hint: Usa accuracy_score()
    
    return modelo


# Código para probar tu implementación
if __name__ == "__main__":
    # Crear datos de ejemplo
    datos_ejemplo = pd.DataFrame({
        'Edad': [25, 35, 45, 55, 65, 75],
        'Colesterol': [180, 200, 220, 240, 260, 280],
        'Enfermedad': [0, 0, 1, 1, 1, 1]
    })
    
    # TODO: Llama a tu función y muestra los resultados
    # Hint: Usa la función regresion_logistica()
    # Hint: Prueba con nuevos datos