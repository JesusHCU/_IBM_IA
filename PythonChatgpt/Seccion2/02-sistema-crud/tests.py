import io
import unittest
from contextlib import redirect_stdout

from producto import Producto, Subproducto
from crud import CRUD
from utilidades import mostrar_productos


class TestCRUD(unittest.TestCase):
    def setUp(self):
        self.crud = CRUD()
        self.crud.crear_producto('Camisa', 'Camisa de algodón', 20.0)
        self.crud.crear_producto('Pantalón', 'Pantalón de mezclilla', 30.0)
        self.crud.crear_producto('Zapatillas', 'Zapatillas deportivas', 50.0, 'Nike')

    def test_leer_producto(self):
        producto = self.crud.leer_producto('Camisa')
        self.assertEqual(producto.nombre, 'Camisa')
        self.assertEqual(producto.descripcion, 'Camisa de algodón')
        self.assertEqual(producto.precio, 20.0)

    def test_leer_producto_no_existente(self):
        with self.assertRaises(ValueError):
            self.crud.leer_producto('Calcetines')

    def test_actualizar_producto(self):
        self.crud.actualizar_producto('Camisa', descripcion='Camisa de algodón de alta calidad', precio=25.0)
        producto = self.crud.leer_producto('Camisa')
        self.assertEqual(producto.descripcion, 'Camisa de algodón de alta calidad')
        self.assertEqual(producto.precio, 25.0)

    def test_actualizar_producto_no_existente(self):
        with self.assertRaises(ValueError):
            self.crud.actualizar_producto('Calcetines', descripcion='Calcetines de lana', precio=5.0)

    def test_actualizar_producto_sin_atributos(self):
        with self.assertRaises(ValueError):
            self.crud.actualizar_producto('Camisa')

    def test_borrar_producto(self):
        self.crud.borrar_producto('Pantalón')
        with self.assertRaises(ValueError):
            self.crud.leer_producto('Pantalón')

    def test_borrar_producto_no_existente(self):
        with self.assertRaises(ValueError):
            self.crud.borrar_producto('Calcetines')


class TestUtilidades(unittest.TestCase):
    def setUp(self):
        self.productos = [
            Producto('Camisa', 'Camisa de algodón', 20.0),
            Producto('Pantalón', 'Pantalón de mezclilla', 30.0),
            Subproducto('Zapatillas', 'Zapatillas deportivas', 50.0, 'Nike')
        ]

    def test_mostrar_productos(self):
        expected_output = "Camisa: Camisa de algodón, precio: 20.0\n" \
                          "Pantalón: Pantalón de mezclilla, precio: 30.0\n" \
                          "Zapatillas: Zapatillas deportivas, precio: 50.0\n"
        with io.StringIO() as buffer, redirect_stdout(buffer):
            mostrar_productos(self.productos)
            self.assertEqual(buffer.getvalue(), expected_output)


if __name__ == '__main__':
    unittest.main()
