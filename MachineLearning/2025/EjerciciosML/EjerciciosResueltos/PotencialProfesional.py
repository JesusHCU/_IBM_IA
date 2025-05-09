import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GameSimulator:
    def __init__(self, n_players=200):
        self.n_players = n_players

    def run(self):
        np.random.seed(42)
        partidas_ganadas = np.random.rand(self.n_players)
        horas_jugadas = np.random.rand(self.n_players)
        precision = np.random.rand(self.n_players)
        reaccion = np.random.rand(self.n_players)
        estrategia = np.random.rand(self.n_players)

        etiquetas = (
            (partidas_ganadas > 0.7) &
            (horas_jugadas > 0.6) &
            (precision > 0.7) &
            (reaccion > 0.6) &
            (estrategia > 0.6)
        ).astype(int)

        X = np.column_stack([partidas_ganadas, horas_jugadas, precision, reaccion, estrategia])
        y = etiquetas
        return X, y


class ProPlayerClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale')

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, player_stats):
        prediction = self.model.predict([player_stats])
        return int(prediction[0])

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)
