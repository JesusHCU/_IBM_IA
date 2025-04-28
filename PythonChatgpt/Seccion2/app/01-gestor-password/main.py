# Importar las bibliotecas necesarias
import tkinter as tk  # Librería para crear interfaces gráficas
from tkinter import messagebox  # Para mostrar mensajes de advertencia o error
import random  # Para generar valores aleatorios
import string  # Para acceder a conjuntos de caracteres


# Función principal para generar la contraseña
def generar_contrasena():
    try:
        # Obtener la longitud de la contraseña desde el cuadro de entrada
        # Convertimos el valor a entero, ya que se ingresa como texto
        longitud = int(longitud_entry.get())

        # Definimos una cadena vacía para almacenar los caracteres seleccionados
        caracteres = ""8

        # Verificar qué opciones de caracteres están seleccionadas:
        # Si la opción "minusculas_var" es verdadera, agregamos letras minúsculas
        if minusculas_var.get():
            caracteres += string.ascii_lowercase  # string.ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"

        # Si la opción "mayusculas_var" es verdadera, agregamos letras mayúsculas
        if mayusculas_var.get():
            caracteres += string.ascii_uppercase  # string.ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Si la opción "numeros_var" es verdadera, agregamos dígitos
        if numeros_var.get():
            caracteres += string.digits  # string.digits = "0123456789"

        # Si la opción "especiales_var" es verdadera, agregamos caracteres especiales
        if especiales_var.get():
            caracteres += string.punctuation  # string.punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        # Verificar que al menos un tipo de carácter esté seleccionado
        # Si el usuario no selecciona ninguna categoría, mostramos advertencia y detenemos la función
        if not caracteres:
            messagebox.showwarning("Advertencia", "Selecciona al menos un tipo de carácter.")
            return  # Finaliza la función sin generar contraseña

        # Generar la contraseña aleatoria
        # Usamos una lista de comprensión para seleccionar caracteres aleatorios
        # random.choice(caracteres) elige un carácter aleatorio de la cadena "caracteres"
        contrasena = ''.join(random.choice(caracteres) for _ in range(longitud))

        # Limpiar el cuadro de texto y mostrar la contraseña generada
        resultado_entry.delete(0, tk.END)  # Borra cualquier texto anterior en el cuadro
        resultado_entry.insert(0, contrasena)  # Inserta la contraseña generada en el cuadro

    except ValueError:
        # Manejar el error si la longitud ingresada no es un número entero válido
        messagebox.showerror("Error", "Introduce un número válido para la longitud de la contraseña.")


# Configuración de la ventana principal de la aplicación
root = tk.Tk()  # Inicializa la ventana principal
root.title("Generador de Contraseñas")  # Título de la ventana
root.geometry("400x300")  # Tamaño de la ventana (ancho x alto)

# Etiqueta y cuadro de entrada para la longitud de la contraseña
longitud_label = tk.Label(root, text="Longitud de la contraseña:")
# Posiciona la etiqueta en la ventana con un pequeño margen vertical
longitud_label.pack(pady=5)

# Cuadro de entrada donde el usuario ingresará la longitud deseada de la contraseña
longitud_entry = tk.Entry(root)
longitud_entry.pack(pady=5)

# Definición de variables de tipo BooleanVar para almacenar el estado de las opciones de caracteres
# BooleanVar es una variable de tkinter que representa valores booleanos (True o False)
minusculas_var = tk.BooleanVar()  # Almacena la selección para incluir minúsculas
mayusculas_var = tk.BooleanVar()  # Almacena la selección para incluir mayúsculas
numeros_var = tk.BooleanVar()  # Almacena la selección para incluir números
especiales_var = tk.BooleanVar()  # Almacena la selección para incluir caracteres especiales

# Casillas de verificación para cada tipo de carácter en la contraseña
# El usuario puede seleccionar los tipos que desee, y la variable asociada guarda el estado de cada opción
tk.Checkbutton(root, text="Incluir minúsculas", variable=minusculas_var).pack(anchor='w')
tk.Checkbutton(root, text="Incluir mayúsculas", variable=mayusculas_var).pack(anchor='w')
tk.Checkbutton(root, text="Incluir números", variable=numeros_var).pack(anchor='w')
tk.Checkbutton(root, text="Incluir caracteres especiales", variable=especiales_var).pack(anchor='w')

# Botón para generar la contraseña
# Al hacer clic, se llama a la función generar_contrasena
generar_button = tk.Button(root, text="Generar Contraseña", command=generar_contrasena)
generar_button.pack(pady=10)  # Se coloca el botón en la ventana con margen

# Cuadro de entrada para mostrar la contraseña generada
# Aquí se mostrará la contraseña generada para que el usuario pueda copiarla
resultado_entry = tk.Entry(root, width=40)  # Ancho del cuadro para acomodar contraseñas largas
resultado_entry.pack(pady=5)

# Ejecutar el bucle principal de la aplicación
# root.mainloop() mantiene la ventana abierta hasta que el usuario la cierre
root.mainloop()

# ========================================================
# EXPLICACIÓN DETALLADA DEL CÓDIGO

# 1. Importación de bibliotecas:
# - tkinter: Es una biblioteca en Python que permite crear interfaces gráficas de usuario (GUIs).
#   Permite crear ventanas, botones, etiquetas, cuadros de texto, etc.
# - messagebox: Forma parte de tkinter y permite mostrar ventanas emergentes de mensajes,
#   como advertencias o errores.
# - random: Esta biblioteca permite generar números y valores aleatorios. En este caso, se utiliza
#   para seleccionar caracteres aleatorios al crear la contraseña.
# - string: Proporciona accesos rápidos a colecciones de caracteres comunes, como letras mayúsculas,
#   minúsculas, dígitos y caracteres de puntuación.

# 2. Función generar_contrasena:
# - Esta función se llama cuando el usuario hace clic en el botón "Generar Contraseña".
# - Primero, obtiene el valor ingresado por el usuario para la longitud de la contraseña, y lo convierte
#   a entero.
# - Luego, define una variable "caracteres" para almacenar el conjunto de caracteres posibles para la
#   contraseña.
# - Las opciones de caracteres están controladas por casillas de verificación, cada una vinculada a una
#   variable BooleanVar. Dependiendo de las opciones seleccionadas, se agregan distintos tipos de
#   caracteres a "caracteres".
# - Si el usuario no selecciona ninguna opción de carácter, la función muestra una advertencia y no genera
#   la contraseña.
# - Para la generación de la contraseña, utiliza una comprensión de lista junto con random.choice() para
#   elegir caracteres aleatorios de "caracteres" hasta alcanzar la longitud deseada.
# - Finalmente, limpia el cuadro de texto del resultado e inserta la contraseña generada.

# 3. Widgets en tkinter:
# - Label: Utilizado para mostrar texto en la ventana. Aquí se usa para mostrar la etiqueta
#   "Longitud de la contraseña".
# - Entry: Permite al usuario ingresar texto. Aquí se usa tanto para ingresar la longitud de la contraseña
#   como para mostrar la contraseña generada.
# - BooleanVar y Checkbutton: Cada Checkbutton permite seleccionar si incluir letras minúsculas,
#   mayúsculas, números o caracteres especiales. Cada uno está vinculado a una variable BooleanVar, que
#   almacena True o False dependiendo del estado de selección.
# - Button: Un botón que, al hacer clic en él, llama a la función generar_contrasena para generar la
#   contraseña.
# - root.mainloop(): Inicia el bucle principal de la ventana para que permanezca abierta y el usuario
#   pueda interactuar con los elementos.

# Este código permite al usuario generar una contraseña personalizada según su preferencia, con la
# longitud y los tipos de caracteres seleccionados. Los widgets de tkinter se usan para proporcionar
# una interfaz interactiva fácil de usar.
