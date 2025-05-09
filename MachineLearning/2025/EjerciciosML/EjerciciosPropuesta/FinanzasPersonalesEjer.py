import matplotlib.pyplot as plt
import numpy as np


def graficar_datos_financieros(meses, datos, tipo="ingresos"):
    """
    TODO: Implementa la función para graficar datos financieros
    Parámetros:
    - meses: lista de meses (eje x)
    - datos: lista de valores (eje y)
    - tipo: "ingresos" o "gastos" para personalizar el gráfico
    """
    # TODO: Crea una figura con tamaño específico
    # Hint: Usa plt.figure(figsize=(10, 5))
    
    # TODO: Crea el gráfico de líneas
    # Hint: Usa plt.plot() con marker='o' y un color apropiado
    
    # TODO: Añade etiquetas y título
    # Hint: Usa plt.xlabel(), plt.ylabel() y plt.title()
    
    # TODO: Configura la leyenda y la cuadrícula
    # Hint: Usa plt.legend() y plt.grid()
    
    # TODO: Muestra el gráfico
    pass


def analizar_finanzas(ingresos, gastos):
    """
    TODO: Implementa el análisis financiero
    Hint: Usa numpy para los cálculos
    """
    # TODO: Convierte las listas a arrays de numpy
    # Hint: Usa np.array()
    
    # TODO: Calcula:
    # 1. Balance mensual (ingresos - gastos)
    # 2. Total de ingresos
    # 3. Total de gastos
    # 4. Saldo final
    
    # TODO: Retorna los resultados en una lista
    pass


def generar_reporte_financiero(ingresos, gastos):
    """
    TODO: Implementa un reporte detallado
    Hint: Usa f-strings para formatear el texto
    """
    # TODO: Calcula estadísticas básicas
    # Hint: Usa np.mean(), np.max(), np.min()
    
    # TODO: Genera un reporte con:
    # - Promedio mensual de ingresos y gastos
    # - Mes con mayores/menores ingresos y gastos
    # - Balance general
    pass


# Código de prueba
if __name__ == "__main__":
    # Datos de ejemplo
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']
    ingresos = [1500, 1600, 1700, 1650, 1800, 1900]
    gastos = [1000, 1100, 1200, 1150, 1300, 1400]

    # TODO: Prueba la función de gráficos
    # Hint: Grafica ingresos y gastos
    
    # TODO: Analiza las finanzas
    # Hint: Usa analizar_finanzas()
    
    # TODO: Genera y muestra el reporte
    # Hint: Usa generar_reporte_financiero()