def mostrar_productos(productos):
    for producto in productos:
        print(f"{producto.nombre}: {producto.descripcion}, precio: {producto.precio}")
