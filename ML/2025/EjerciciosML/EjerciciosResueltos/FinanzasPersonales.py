import matplotlib.pyplot as plt
import numpy as np


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


# Datos proporcionados
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [22, 24, 23, 25, 26, 28, 27]

graficar_temperaturas(dias, temperaturas)

# Datos de finanzas
ingresos = [1500, 1600, 1700, 1650, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
gastos = [1000, 1100, 1200, 1150, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

resultado = analizar_finanzas(ingresos, gastos)
print(resultado)
