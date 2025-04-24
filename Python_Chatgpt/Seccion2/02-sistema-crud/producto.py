class Producto:
    """
    Clase que representa un producto.

    Atributos:
    - nombre (str): el nombre del producto.
    - descripcion (str): la descripción del producto.
    - precio (float): el precio del producto.
    """

    def __init__(self, nombre, descripcion, precio):
        """
        Inicializa una instancia de la clase Producto.

        Args:
        - nombre (str): el nombre del producto.
        - descripcion (str): la descripción del producto.
        - precio (float): el precio del producto.
        """
        self.nombre = nombre
        self.descripcion = descripcion
        self.precio = precio


class Subproducto(Producto):
    """
    Clase que representa un subproducto, que es un tipo de producto.

    Atributos:
    - marca (str): la marca del subproducto.
    """

    def __init__(self, nombre, descripcion, precio, marca):
        """
        Inicializa una instancia de la clase Subproducto.

        Args:
        - nombre (str): el nombre del subproducto.
        - descripcion (str): la descripción del subproducto.
        - precio (float): el precio del subproducto.
        - marca (str): la marca del subproducto.
        """
        super().__init__(nombre, descripcion, precio)
        self.marca = marca
