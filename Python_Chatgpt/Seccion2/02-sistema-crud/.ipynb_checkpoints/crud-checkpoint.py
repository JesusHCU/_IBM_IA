from producto import Producto, Subproducto


class CRUD:
    """
    Clase que representa un sistema CRUD (Crear, Leer, Actualizar, Eliminar) para productos.

    Atributos:
    - productos (list): La lista de productos.

    Métodos:
    - crear_producto(nombre, descripcion, precio, marca=None):
        Crea un nuevo producto y lo agrega a la lista de productos.

    - leer_producto(nombre):
        Busca un producto por nombre en la lista de productos.

    - actualizar_producto(nombre, descripcion=None, precio=None, marca=None):
        Actualiza un producto existente en la lista de productos.

    - borrar_producto(nombre):
        Elimina un producto existente de la lista de productos.

    Raises:
    - ValueError: si el producto no existe en la lista de productos o si se proporcionan valores nulos
    para todos los atributos en el método actualizar_producto.
    """

    def __init__(self):
        """
        Inicializa una instancia de la clase CRUD.
        """
        self.productos = []

    def crear_producto(self, nombre, descripcion, precio, marca=None):
        """
        Crea un nuevo producto y lo agrega a la lista de productos.

        Args:
        - nombre (str): El nombre del producto.
        - descripcion (str): La descripción del producto.
        - precio (float): El precio del producto.
        - marca (str, opcional): La marca del subproducto, si es un subproducto.

        Returns:
        - None.
        """
        if marca:
            producto = Subproducto(nombre, descripcion, precio, marca)
        else:
            producto = Producto(nombre, descripcion, precio)
        self.productos.append(producto)

    def leer_producto(self, nombre):
        """
        Busca un producto por nombre en la lista de productos.

        Args:
        - nombre (str): El nombre del producto a buscar.

        Returns:
        - El producto encontrado.

        Raises:
        - ValueError: si el producto no existe en la lista de productos.
        """
        for producto in self.productos:
            if producto.nombre == nombre:
                return producto
        raise ValueError("Producto no encontrado")

    def actualizar_producto(self, nombre, descripcion=None, precio=None, marca=None):
        """
        Actualiza un producto existente en la lista de productos.

        Args:
        - nombre (str): El nombre del producto a actualizar.
        - descripcion (str, opcional): La nueva descripción del producto.
        - precio (float, opcional): El nuevo precio del producto.
        - marca (str, opcional): La nueva marca del subproducto, si es un subproducto.

        Returns:
        - None.

        Raises:
        - ValueError: si el producto no existe en la lista de productos o si se proporcionan valores nulos
        para todos los atributos en el método actualizar_producto.
        """
        producto = self.leer_producto(nombre)
        if descripcion is not None:
            producto.descripcion = descripcion
        if precio is not None:
            producto.precio = precio
        if marca is not None:
            producto.marca = marca
        if descripcion is None and precio is None and marca is None:
            raise ValueError("Debe proporcionar al menos un atributo de actualización")

    def borrar_producto(self, nombre):
        """
        Elimina un producto existente de la lista de productos.

        Args:
        - nombre (str): El nombre del producto a eliminar.

        Returns:
        - None.

        Raises:
        - ValueError: si el producto no existe en la lista de productos.
        """
        producto = self.leer_producto(nombre)
        self.productos.remove(producto)
