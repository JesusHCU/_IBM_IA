import numpy as np
from sklearn.linear_model import LogisticRegression


class PlayerMatchData:
    def __init__(self, kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won):
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.damage_dealt = damage_dealt
        self.damage_received = damage_received
        self.healing_done = healing_done
        self.objective_time = objective_time
        self.won = won  # 1 si ganó, 0 si perdió

    def to_dict(self):
        return {
            'kills': self.kills,
            'deaths': self.deaths,
            'assists': self.assists,
            'damage_dealt': self.damage_dealt,
            'damage_received': self.damage_received,
            'healing_done': self.healing_done,
            'objective_time': self.objective_time
        }


def generate_synthetic_data(n=100):
    data = []
    for _ in range(n):
        kills = np.random.poisson(5)
        deaths = np.random.poisson(3)
        assists = np.random.poisson(2)
        damage_dealt = kills * 300 + np.random.normal(0, 100)
        damage_received = deaths * 400 + np.random.normal(0, 100)
        healing_done = np.random.randint(0, 301)
        objective_time = np.random.randint(0, 121)
        # Determinar si ganó
        won = 1 if (damage_dealt > damage_received) and (kills > deaths) else 0
        player = PlayerMatchData(
            kills, deaths, assists, damage_dealt,
            damage_received, healing_done, objective_time, won
        )
        data.append(player)
    return data


class VictoryPredictor:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, data):
        X = []
        y = []
        for player in data:
            features = [
                player.kills,
                player.deaths,
                player.assists,
                player.damage_dealt,
                player.damage_received,
                player.healing_done,
                player.objective_time
            ]
            X.append(features)
            y.append(player.won)
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y)

    def predict(self, player):
        features = [
            player.kills,
            player.deaths,
            player.assists,
            player.damage_dealt,
            player.damage_received,
            player.healing_done,
            player.objective_time
        ]
        X_test = np.array([features])
        return self.model.predict(X_test)[0]


# Generar datos de entrenamiento
training_data = generate_synthetic_data(150)

# Entrenar el modelo
predictor = VictoryPredictor()
predictor.train(training_data)

# Crear jugador de prueba
test_player = PlayerMatchData(8, 2, 3, 2400, 800, 120, 90, None)

# Predecir resultado
prediction = predictor.predict(test_player)
print(f"¿El jugador ganará? {'Sí' if prediction == 1 else 'No'}")