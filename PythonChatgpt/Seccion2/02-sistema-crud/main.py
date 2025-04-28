from producto import Producto, Subproducto
from crud import CRUD
from utilidades import mostrar_productos

if __name__ == "__main__":
    crud = CRUD()

    # Crear algunos productos de prueba
    crud.crear_producto('Camisa', 'Camisa de algodón', 20.0)
    crud.crear_producto('Pantalón', 'Pantalón de mezclilla', 30.0)
    crud.crear_producto('Zapatillas', 'Zapatillas deportivas', 50.0, 'Nike')

    # Mostrar todos los productos disponibles
    mostrar_productos(crud.productos)

    # Actualizar un producto existente
    crud.actualizar_producto('Camisa', descripcion='Camisa de algodón de alta calidad', precio=25.0)

    # Mostrar todos los productos disponibles nuevamente
    mostrar_productos(crud.productos)

    # Eliminar un producto existente
    crud.borrar_producto('Pantalón')

    # Mostrar todos los productos disponibles nuevamente
    mostrar_productos(crud.productos)
