from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


class Player:
    """
    Represents a player with their characteristics and preferred class.
    """

    def __init__(self, name, level, aggressiveness, cooperation, exploration, preferred_class=None):
        self.name = name
        self.level = level  # 1-100
        self.aggressiveness = aggressiveness  # 0-1
        self.cooperation = cooperation  # 0-1
        self.exploration = exploration  # 0-1
        self.preferred_class = preferred_class

    def to_features(self):
        """
        Converts player data to numerical features for the model.
        Returns a list of features without the preferred class.
        """
        return [
            self.level,
            self.aggressiveness,
            self.cooperation,
            self.exploration
        ]

    def __str__(self):
        return f"{self.name} (Level {self.level})"


class PlayerDataset:
    """
    Handles a collection of players for training or testing.
    """

    def __init__(self, players):
        self.players = players

    def get_X(self):
        """Returns the feature matrix for all players in the dataset."""
        return [player.to_features() for player in self.players]

    def get_y(self):
        """Returns the target classes for all players in the dataset."""
        return [player.preferred_class for player in self.players]


class ClassRecommender:
    """
    Recommends character classes based on player features using KNN.
    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.trained = False

    def train(self, dataset):
        """
        Trains the model using the provided PlayerDataset.
        """
        X = dataset.get_X()
        y = dataset.get_y()

        # Check if we have enough data
        if len(X) < self.n_neighbors:
            raise ValueError(f"Not enough training data. Need at least {self.n_neighbors} players.")

        self.model.fit(X, y)
        self.trained = True

    def predict(self, player):
        """
        Predicts the best character class for a new player.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before making predictions.")

        features = [player.to_features()]
        return self.model.predict(features)[0]

    def get_nearest_neighbors(self, player):
        """
        Returns indices of the k nearest players to the given player.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before finding neighbors.")

        features = [player.to_features()]
        distances, indices = self.model.kneighbors(features)

        # Return a flat list of indices
        return indices[0]

    def evaluate(self, dataset, cv=5):
        """
        Evaluates the model using cross-validation.
        Returns the average accuracy score.
        """
        X = dataset.get_X()
        y = dataset.get_y()

        scores = cross_val_score(self.model, X, y, cv=cv)
        return np.mean(scores)


# Example usage
if __name__ == "__main__":
    # Training data
    players = [
        Player("Alice", 20, 0.8, 0.2, 0.1, "Warrior"),
        Player("Bob", 45, 0.4, 0.8, 0.2, "Healer"),
        Player("Cleo", 33, 0.6, 0.4, 0.6, "Archer"),
        Player("Dan", 60, 0.3, 0.9, 0.3, "Healer"),
        Player("Eli", 50, 0.7, 0.2, 0.9, "Mage"),
        Player("Fay", 25, 0.9, 0.1, 0.2, "Warrior"),
    ]

    # New player to recommend a class for
    new_player = Player("TestPlayer", 40, 0.6, 0.3, 0.8)

    # Set up the dataset and recommender
    dataset = PlayerDataset(players)
    recommender = ClassRecommender(n_neighbors=3)
    recommender.train(dataset)

    # Get and display recommendation
    recommended_class = recommender.predict(new_player)
    neighbors_indices = recommender.get_nearest_neighbors(new_player)

    print(f"Clase recomendada para {new_player.name}: {recommended_class}")
    print("Jugadores similares:")
    for i in neighbors_indices:
        print(f"- {players[i].name} ({players[i].preferred_class})")

    # Optional: Evaluate model accuracy with cross-validation
    if len(players) >= 5:  # Need at least 5 samples for 5-fold CV
        accuracy = recommender.evaluate(dataset)
        print(f"\nPrecisión del modelo (cross-validation): {accuracy:.2f}")