def division(numero1, numero2):
    """
    Esta función recibe dos números y devuelve el resultado de la división entre ellos.

    Args:
        numero1 (float): El primer número de la división.
        numero2 (float): El segundo número de la división.

    Returns:
        float: El resultado de la división entre numero1 y numero2.

        Si se intenta dividir entre cero, la función mostrará un mensaje de error
        en la consola y devolverá None.
    """
    try:
        resultado = numero1 / numero2
    except ZeroDivisionError:
        print("No se puede dividir entre cero")
        resultado = None
    return resultado
