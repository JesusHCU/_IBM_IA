import matplotlib.pyplot as plt


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


# Datos proporcionados
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [22, 24, 23, 25, 26, 28, 27]

graficar_temperaturas(dias, temperaturas)