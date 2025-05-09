import numpy as np
from sklearn.cluster import KMeans

class Traveler:
    """Clase que representa un viajero con sus preferencias"""
    def __init__(self, beach, mountain, city, countryside):
        # TODO: Guarda las preferencias del viajero (0-10)
        # Hint: Usa self.attribute = attribute para cada preferencia
        pass

    def to_vector(self):
        # TODO: Retorna una lista con las preferencias
        # Hint: [beach, mountain, city, countryside]
        return []


# Esta clase ya está implementada para ti
class TravelerGenerator:
    def __init__(self, num_travelers=100):
        self.num_travelers = num_travelers

    def generate(self):
        travelers = []
        for _ in range(self.num_travelers):
            # Genera preferencias aleatorias (0-10)
            beach = np.round(np.random.uniform(0, 10), 2)
            mountain = np.round(np.random.uniform(0, 10), 2)
            city = np.round(np.random.uniform(0, 10), 2)
            countryside = np.round(np.random.uniform(0, 10), 2)
            travelers.append(Traveler(beach, mountain, city, countryside))
        return travelers


class TravelerClusterer:
    """Agrupa viajeros según sus preferencias usando K-Means"""
    def __init__(self, n_clusters=3):
        # Ya inicializado para ti
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers = None

    def fit(self, travelers):
        """
        TODO: Entrena el modelo con la lista de viajeros
        Hint: Usa self.model.fit()
        """
        # TODO: Convierte los viajeros a vectores de características
        # TODO: Entrena el modelo
        # TODO: Guarda los centros de los clusters
        pass

    def predict(self, traveler):
        """
        TODO: Predice el grupo al que pertenece un viajero
        Hint: Usa self.model.predict()
        """
        # TODO: Convierte el viajero a vector
        # TODO: Realiza la predicción
        return 0  # Reemplaza con tu implementación


# Código de ejemplo
if __name__ == "__main__":
    # Generar datos
    print("Generando viajeros...")
    generator = TravelerGenerator(num_travelers=100)
    viajeros = generator.generate()
    
    # Crear y entrenar agrupador
    print("\nAgrupando viajeros...")
    agrupador = TravelerClusterer(n_clusters=3)
    agrupador.fit(viajeros)
    
    # Probar con nuevo viajero
    nuevo_viajero = Traveler(
        beach=8.5,       # Le encanta la playa
        mountain=2.0,    # No le gustan las montañas
        city=9.0,        # Le encantan las ciudades
        countryside=1.5   # No le gusta el campo
    )
    
    # Predecir grupo
    grupo = agrupador.predict(nuevo_viajero)
    
    # Mostrar resultados
    print("\n🧳 Nuevo viajero:")
    print(f"- Playa: {nuevo_viajero.beach}/10")
    print(f"- Montaña: {nuevo_viajero.mountain}/10")
    print(f"- Ciudad: {nuevo_viajero.city}/10")
    print(f"- Campo: {nuevo_viajero.countryside}/10")
    print(f"\n📍 Grupo asignado: {grupo}")