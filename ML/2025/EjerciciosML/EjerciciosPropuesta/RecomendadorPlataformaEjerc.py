from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


class Project:
    """Clase que representa un proyecto de desarrollo"""
    def __init__(self, project_name, team_size, budget, duration_months,
                 realtime_required, needs_offline, target_users, recommended_platform=None):
        # TODO: Guarda los atributos del proyecto
        # Hint: Cada atributo debe guardarse como self.attribute = attribute
        pass


# Esta clase ya está implementada para ti
class ProjectDataset:
    def __init__(self, projects):
        self.projects = projects
        self.label_encoder = LabelEncoder()
        self._fit_label_encoder()

    def _fit_label_encoder(self):
        target_users_list = [p.target_users for p in self.projects]
        self.label_encoder.fit(target_users_list)

    def get_features_and_labels(self):
        # Extrae características y etiquetas
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
        # Codifica un nuevo proyecto
        return np.array([[
            project.team_size,
            project.budget,
            project.duration_months,
            project.realtime_required,
            project.needs_offline,
            self.label_encoder.transform([project.target_users])[0]
        ]])


class PlatformRecommender:
    """Sistema de recomendación de plataformas"""
    def __init__(self):
        # Ya inicializado para ti
        self.model = DecisionTreeClassifier()
        self.dataset = None

    def train(self, dataset):
        """
        TODO: Entrena el modelo con el dataset proporcionado
        Hint: Usa self.model.fit(X, y)
        """
        # TODO: Guarda el dataset
        # TODO: Obtén características (X) y etiquetas (y)
        # TODO: Entrena el modelo
        pass

    def predict(self, project):
        """
        TODO: Predice la plataforma recomendada para un proyecto
        Hint: Usa self.model.predict()
        """
        # TODO: Verifica que el modelo está entrenado
        # TODO: Codifica el proyecto nuevo
        # TODO: Realiza la predicción
        return "Android"  # Reemplaza con tu implementación


# Código de ejemplo
if __name__ == "__main__":
    # Crear datos de ejemplo
    proyectos_ejemplo = [
        Project("App1", 5, 50000, 6, True, False, "jóvenes", "Android"),
        Project("App2", 3, 30000, 4, False, True, "profesionales", "iOS"),
        Project("App3", 8, 100000, 12, True, True, "empresas", "Web"),
    ]
    
    # Crear dataset y recomendador
    dataset = ProjectDataset(proyectos_ejemplo)
    recomendador = PlatformRecommender()
    
    # Entrenar modelo
    print("Entrenando modelo...")
    recomendador.train(dataset)
    
    # Probar con nuevo proyecto
    nuevo_proyecto = Project(
        "MiApp",
        team_size=4,           # Tamaño del equipo
        budget=40000,          # Presupuesto
        duration_months=5,      # Duración en meses
        realtime_required=True, # ¿Necesita tiempo real?
        needs_offline=False,    # ¿Necesita modo offline?
        target_users="jóvenes", # Usuarios objetivo
        recommended_platform=None
    )
    
    # Obtener recomendación
    plataforma = recomendador.predict(nuevo_proyecto)
    print(f"\nPara el proyecto {nuevo_proyecto.project_name}:")
    print(f"Plataforma recomendada: {plataforma}")