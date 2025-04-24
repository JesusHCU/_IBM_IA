from funciones import division

if __name__ == "__main__":
    numero1 = None
    while not isinstance(numero1, float):
        try:
            numero1 = float(input("Ingresa un número: "))
        except ValueError:
            print("El valor ingresado no es válido, por favor intenta nuevamente")

    numero2 = None
    while not isinstance(numero2, float):
        try:
            numero2 = float(input("Ingresa otro número distinto de cero: "))
            if numero2 == 0:
                print("El número debe ser distinto de cero")
                numero2 = None
        except ValueError:
            print("El valor ingresado no es válido, por favor intenta nuevamente")

    resultado = division(numero1, numero2)
    if resultado is not None:
        print("El resultado es:", resultado)
