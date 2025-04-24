import unittest
from funcion import division

class TestDivision(unittest.TestCase):

    def test_division_entre_dos_numeros_positivos(self):
        self.assertEqual(division(10, 2), 5)

    def test_division_entre_dos_numeros_negativos(self):
        self.assertEqual(division(-10, -2), 5)

    def test_division_entre_numeros_positivos_y_negativos(self):
        self.assertEqual(division(10, -2), -5)

    def test_division_entre_cero_y_un_numero(self):
        self.assertEqual(division(0, 10), 0.0)

    def test_division_entre_dos_numeros_decimales(self):
        self.assertAlmostEqual(division(5.5, 2), 2.75)


if __name__ == '__main__':
    unittest.main()
