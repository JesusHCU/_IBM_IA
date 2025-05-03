def operaciones_matematicas(a, b):
    """
    Realiza operaciones matemáticas básicas entre dos números enteros.

    Args:
        a: El primer número entero.
        b: El segundo número entero.

    Returns:
        Una tupla con los resultados de la suma, resta, multiplicación,
        división y resto. Devuelve mensajes de error para la división por cero.
    """
    suma = a + b
    resta = a - b
    multiplicacion = a * b
    if b == 0:
        division = "División por cero no permitida"
        resto = "División por cero no permitida"
    else:
        division = a / b
        resto = a % b
    return (suma, resta, multiplicacion, division, resto)

# Ejemplo de uso con los valores originales del ejercicio:
a_ejemplo = 10
b_ejemplo = 3
resultado_ejemplo = operaciones_matematicas(a_ejemplo, b_ejemplo)
print("Con a =", a_ejemplo, "y b =", b_ejemplo, ":")
print("Suma:", resultado_ejemplo[0])
print("Resta:", resultado_ejemplo[1])
print("Multiplicación:", resultado_ejemplo[2])
print("División:", resultado_ejemplo[3])
print("Residuo:", resultado_ejemplo[4])