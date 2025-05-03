import matplotlib.pyplot as plt


def graficar_linea(x, y):
    plt.figure(figsize=(8, 5))  # Tamaño de la figura
    plt.plot(x, y, marker='o', linestyle='-', color='b', markersize=8, linewidth=2)

    # Etiquetas y título
    plt.xlabel("Horas de Estudio")
    plt.ylabel("Calificación")
    plt.title("Relación entre horas de estudio y calificaciones")

    # Mostrar la cuadrícula para mejor visualización
    plt.grid(True, linestyle='--', alpha=0.6)

    # Mostrar la gráfica
    plt.show()


# Datos proporcionados
horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8]
calificaciones = [55, 60, 65, 70, 75, 80, 85, 90]

graficar_linea(horas_estudio, calificaciones)