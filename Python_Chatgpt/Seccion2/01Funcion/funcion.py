"""
def division(numero1, numero2):
    resultado = numero1 / numero_2
    return resultado


numero1 = input("Ingresa un número: "
numero2 = input("Ingresa otro número: ")
resultado = dividir_dos_numeros(numero1, numero2)
print("El resultado es: ", resultado)

Explicación de las correcciones:
Paréntesis cerrados en input: Se añadieron los paréntesis faltantes en cada llamada a input.
Nombres de variables consistentes: numero_2 fue cambiado a numero2 en la función division.
Nombre correcto de la función: La llamada a la función dividir_dos_numeros fue corregida a division.
Conversión a float: Los valores ingresados por el usuario se convierten a float para poder realizar la división.

# Definimos la función 'division' que toma dos números como parámetros
def division(numero1, numero2):
    # Función para dividir dos números.
    # Calcula el resultado de dividir numero1 entre numero2
    resultado = numero1 / numero2
    # Devuelve el resultado de la división
    return resultado

# Bloque principal que se ejecutará solo si este archivo se ejecuta directamente
if __name__ == "__main__":
    # Solicita al usuario que ingrese el primer número
    # 'input' recibe el valor en formato de texto, así que usamos 'float' para convertirlo a un número decimal
    numero1 = float(input("Ingresa un número: "))

    # Solicita al usuario que ingrese el segundo número y lo convierte a 'float'
    numero2 = float(input("Ingresa otro número: "))

    # Llama a la función 'division' con los dos números ingresados y guarda el resultado en la variable 'resultado'
    resultado = division(numero1, numero2)

    # Muestra el resultado de la división en la consola
    print("El resultado es:", resultado)

Explicación de las excepciones utilizadas:
ZeroDivisionError: Se captura dentro de la función division para manejar el caso en que el divisor (numero2) es cero,
lo cual no está permitido en las operaciones matemáticas. En este caso, se devuelve un mensaje de error en lugar del 
resultado de la división.
ValueError: Se captura en el bloque principal (donde se solicita la entrada del usuario) para manejar el caso en que 
el usuario ingresa un valor que no se puede convertir a un número (por ejemplo, texto en lugar de un número). Esto 
asegura que el programa continúe de forma segura si hay un error de entrada.

"""
# Definimos la función 'division' que toma dos números como parámetros
def division(numero1, numero2):
    """Función para dividir dos números."""
    try:
        # Calcula el resultado de dividir numero1 entre numero2
        resultado = numero1 / numero2
    except ZeroDivisionError:
        # Captura el error en caso de que numero2 sea cero
        return "Error: No se puede dividir entre cero."
    return resultado

# Bloque principal que se ejecutará solo si este archivo se ejecuta directamente
if __name__ == "__main__":
    try:
        # Solicita al usuario que ingrese el primer número y lo convierte a 'float'
        numero1 = float(input("Ingresa un número: "))

        # Solicita al usuario que ingrese el segundo número y lo convierte a 'float'
        numero2 = float(input("Ingresa otro número: "))

        # Llama a la función 'division' con los dos números ingresados y guarda el resultado en la variable 'resultado'
        resultado = division(numero1, numero2)

        # Muestra el resultado de la división en la consola
        print("El resultado es:", resultado)

    except ValueError:
        # Captura el error si el usuario ingresa un valor no numérico
        print("Error: Debes ingresar un número válido.")
