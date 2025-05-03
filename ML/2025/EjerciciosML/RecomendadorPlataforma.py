from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Clase para representar un proyecto
class Project:
    def __init__(self, project_name, team_size, budget, duration_months,
                 realtime_required, needs_offline, target_users, recommended_platform=None):
        self.project_name = project_name
        self.team_size = team_size
        self.budget = budget
        self.duration_months = duration_months
        self.realtime_required = int(realtime_required)
        self.needs_offline = int(needs_offline)
        self.target_users = target_users
        self.recommended_platform = recommended_platform


# Clase para gestionar un conjunto de proyectos y preparar los datos
class ProjectDataset:
    def __init__(self, projects):
        self.projects = projects
        self.label_encoder = LabelEncoder()
        self._fit_label_encoder()

    def _fit_label_encoder(self):
        target_users_list = [p.target_users for p in self.projects]
        self.label_encoder.fit(target_users_list)

    def get_features_and_labels(self):
        features = []
        labels = []
        for p in self.projects:
            features.append([
                p.team_size,
                p.budget,
                p.duration_months,
                p.realtime_required,
                p.needs_offline,
                self.label_encoder.transform([p.target_users])[0]
            ])
            labels.append(p.recommended_platform)
        return np.array(features), np.array(labels)

    def encode_project(self, project):
        return np.array([[
            project.team_size,
            project.budget,
            project.duration_months,
            project.realtime_required,
            project.needs_offline,
            self.label_encoder.transform([project.target_users])[0]
        ]])


# Clase que entrena y realiza predicciones
class PlatformRecommender:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.dataset = None

    def train(self, dataset):
        self.dataset = dataset
        X, y = dataset.get_features_and_labels()
        self.model.fit(X, y)

    def predict(self, project):
        if not self.dataset:
            raise ValueError("El modelo no ha sido entrenado todavía.")
        X_new = self.dataset.encode_project(project)
        return self.model.predict(X_new)[0]
