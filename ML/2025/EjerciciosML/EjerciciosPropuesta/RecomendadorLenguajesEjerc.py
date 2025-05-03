import numpy as np
from sklearn.ensemble import RandomForestClassifier


def generate_dataset(n_samples=100):
    """
    TODO: Genera datos de entrenamiento para el recomendador
    Hint: Usa np.random para generar características aleatorias
    """
    # Características ya generadas para ti
    velocidad_requerida = np.random.rand(n_samples)
    facilidad_mantenimiento = np.random.rand(n_samples)
    disponibilidad_bibliotecas = np.random.rand(n_samples)
    tipo_app = np.random.randint(0, 3, size=n_samples)
    rendimiento_deseado = np.random.rand(n_samples)

    # TODO: Combina las características en una matriz
    # Hint: Usa np.column_stack()
    X = None
    
    # Inicializar etiquetas
    y = np.zeros(n_samples, dtype=int)
    
    # TODO: Implementa las reglas para asignar lenguajes
    for i in range(n_samples):
        # Python (0): Fácil mantenimiento, muchas bibliotecas
        if (facilidad_mantenimiento[i] > 0.6 and 
            disponibilidad_bibliotecas[i] > 0.7):
            y[i] = 0
            
        # TODO: Implementa reglas para JavaScript (1)
        # Hint: Bueno para web (tipo_app == 0)
        
        # TODO: Implementa reglas para Java (2)
        # Hint: Bueno para móvil (tipo_app == 1)
        
        # TODO: Implementa reglas para C++ (3)
        # Hint: Alto rendimiento o sistemas
        
        else:
            y[i] = np.random.randint(0, 4)
    
    return X, y


class LanguagePredictor:
    """Clase para recomendar lenguajes de programación"""
    def __init__(self):
        # Ya inicializado para ti
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.language_map = {
            0: "Python",
            1: "JavaScript",
            2: "Java",
            3: "C++"
        }

    def train(self, X, y):
        """
        TODO: Entrena el modelo con los datos proporcionados
        Hint: Usa self.model.fit()
        """
        pass

    def predict(self, features):
        """
        TODO: Predice el mejor lenguaje para un nuevo proyecto
        Hint: Usa self.model.predict() y self.language_map
        """
        # TODO: Asegura que features es 2D
        # TODO: Realiza la predicción
        # TODO: Convierte el índice a nombre de lenguaje
        return "Python"  # Reemplaza con tu implementación


# Código de prueba
if __name__ == "__main__":
    # Generar datos
    print("Generando datos de entrenamiento...")
    X, y = generate_dataset(n_samples=100)
    
    # Crear y entrenar predictor
    print("\nEntrenando modelo...")
    predictor = LanguagePredictor()
    predictor.train(X, y)
    
    # Probar con nuevos proyectos
    proyectos_prueba = [
        # [velocidad, mantenimiento, bibliotecas, tipo, rendimiento]
        [0.7, 0.9, 0.8, 0, 0.3],  # Debería sugerir Python
        [0.5, 0.6, 0.4, 0, 0.4],  # Debería sugerir JavaScript
        [0.6, 0.5, 0.6, 1, 0.7],  # Debería sugerir Java
        [0.9, 0.3, 0.4, 2, 0.9],  # Debería sugerir C++
    ]
    
    print("\nProbando predicciones:")
    for i, proyecto in enumerate(proyectos_prueba, 1):
        lenguaje = predictor.predict(np.array(proyecto))
        print(f"\nProyecto {i}:")
        print(f"- Velocidad requerida: {proyecto[0]:.2f}")
        print(f"- Facilidad mantenimiento: {proyecto[1]:.2f}")
        print(f"- Disponibilidad bibliotecas: {proyecto[2]:.2f}")
        print(f"- Tipo app: {['Web', 'Móvil', 'Sistemas'][int(proyecto[3])]}")
        print(f"- Rendimiento deseado: {proyecto[4]:.2f}")
        print(f"→ Lenguaje recomendado: {lenguaje}")