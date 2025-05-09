from sklearn.neighbors import NearestNeighbors
import numpy as np

class Song:
    """Clase que representa una canción"""
    def __init__(self, title, artist, energy, danceability, duration, popularity):
        # TODO: Guarda los atributos de la canción
        # Hint: Usa self.attribute = attribute para cada atributo
        pass

    def to_vector(self):
        # TODO: Retorna una lista con las características numéricas
        # Hint: [energy, danceability, duration, popularity]
        return []

    def __str__(self):
        # Ya implementado para ti
        return f"{self.title} by {self.artist}"


# Esta clase ya está implementada para ti
class SongGenerator:
    def __init__(self, num_songs=30):
        self.num_songs = num_songs

    def generate(self):
        songs = []
        for i in range(1, self.num_songs + 1):
            title = f"Song{i}"
            artist = f"Artist{np.random.randint(1, 6)}"
            energy = np.random.uniform(0.4, 1.0)
            danceability = np.random.uniform(0.4, 1.0)
            duration = np.random.randint(180, 301)
            popularity = np.random.randint(50, 101)
            songs.append(Song(title, artist, energy, danceability, duration, popularity))
        return songs


class SongRecommender:
    """Sistema de recomendación basado en similitud"""
    def __init__(self, k=3):
        # Ya inicializado para ti
        self.k = k
        self.songs = []
        self.model = NearestNeighbors(n_neighbors=k+1)
        self.features = []

    def fit(self, song_list):
        # TODO: Prepara el modelo con la lista de canciones
        # Hint: 
        # 1. Guarda la lista de canciones
        # 2. Extrae las características
        # 3. Entrena el modelo
        pass

    def recommend(self, target_song):
        # TODO: Encuentra las k canciones más similares
        # Hint:
        # 1. Obtén el vector de la canción objetivo
        # 2. Usa kneighbors para encontrar vecinos
        # 3. Filtra la canción objetivo
        return []


# Código de ejemplo
if __name__ == "__main__":
    # Generar canciones de ejemplo
    print("Generando canciones...")
    generator = SongGenerator(num_songs=20)
    canciones = generator.generate()
    
    # Crear y entrenar recomendador
    print("\nPreparando sistema de recomendación...")
    recomendador = SongRecommender(k=3)
    recomendador.fit(canciones)
    
    # Canción de prueba
    cancion_objetivo = Song(
        "Mi Canción",
        "Artista Ejemplo",
        energy=0.8,        # Energía (0-1)
        danceability=0.7,  # Bailabilidad (0-1)
        duration=240,      # Duración en segundos
        popularity=85      # Popularidad (0-100)
    )
    
    # Obtener recomendaciones
    print("\nBuscando recomendaciones...")
    recomendaciones = recomendador.recommend(cancion_objetivo)
    
    # Mostrar resultados
    print(f"\nPara la canción '{cancion_objetivo.title}'")
    print("Recomendaciones:")
    for cancion in recomendaciones:
        print(f"- {cancion}")