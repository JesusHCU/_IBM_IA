class Song:
    def __init__(self, title, artist, energy, danceability, duration, popularity):
        self.title = title
        self.artist = artist
        self.energy = energy
        self.danceability = danceability
        self.duration = duration
        self.popularity = popularity

    def to_vector(self):
        return [self.energy, self.danceability, self.duration, self.popularity]

    def __str__(self):
        return f"{self.title} by {self.artist}"
from sklearn.neighbors import NearestNeighbors

class SongRecommender:
    def __init__(self, k=3):
        self.k = k
        self.songs = []
        self.model = NearestNeighbors(n_neighbors=k+1)  # +1 por si la canción objetivo se incluye
        self.features = []

    def fit(self, song_list):
        self.songs = song_list
        self.features = [song.to_vector() for song in song_list]
        self.model.fit(self.features)

    def recommend(self, target_song):
        target_vector = [target_song.to_vector()]
        distances, indices = self.model.kneighbors(target_vector)
        recommended_songs = []

        for index in indices[0]:
            candidate = self.songs[index]
            if candidate != target_song:  # Evita recomendar la misma canción
                recommended_songs.append(candidate)

        return recommended_songs[:self.k]
import numpy as np

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
class SongRecommendationExample:
    def run(self):
        generator = SongGenerator()
        song_list = generator.generate()

        # Definimos una canción objetivo personalizada
        target_song = Song("Mi Canción", "Yo", 0.8, 0.7, 240, 90)

        # Entrenamos el recomendador
        recommender = SongRecommender(k=3)
        recommender.fit(song_list + [target_song])  # Incluimos la canción objetivo en el set

        recommendations = recommender.recommend(target_song)

        print(f"🎵 Recomendaciones para '{target_song.title}':")
        for song in recommendations:
            print(f" - {song}")
if __name__ == "__main__":
    example = SongRecommendationExample()
    example.run()
