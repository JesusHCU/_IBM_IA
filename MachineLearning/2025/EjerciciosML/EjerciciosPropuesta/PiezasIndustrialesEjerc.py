import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 2. Clase Piece: representa una pieza con sus características y etiqueta
class Piece:
    def __init__(self, texture, symmetry, edges, center_offset, label):
        self.texture = texture
        self.symmetry = symmetry
        self.edges = edges
        self.center_offset = center_offset
        self.label = label  # "Correcta" o "Defectuosa"

    def to_vector(self):
        # TODO: Retornar una lista con las características numéricas de la pieza
        # Esta función es fundamental para preparar los datos para el modelo
        pass

    def __repr__(self):
        return f"Piece(texture={self.texture:.2f}, symmetry={self.symmetry:.2f}, edges={self.edges}, center_offset={self.center_offset:.2f}, label='{self.label}')"

# 3. Clase PieceDatasetGenerator: genera un conjunto de piezas con etiquetas
class PieceDatasetGenerator:
    def __init__(self, n=400):
        self.n = n

    def generate(self):
        np.random.seed(42)
        textures = np.random.normal(0.5, 0.15, size=self.n)
        symmetries = np.random.normal(0.6, 0.2, size=self.n)
        edges = np.clip(np.random.normal(50, 15, size=self.n), 0, None).astype(int)
        offsets = np.abs(np.random.normal(0, 0.2, size=self.n))

        data = []
        for t, s, e, o in zip(textures, symmetries, edges, offsets):
            # TODO: Implementar las reglas para asignar la etiqueta "Correcta" o "Defectuosa"
            # Pista: Usa condiciones sobre s, o, t y e como en el ejemplo original
            label = None
            data.append(Piece(t, s, e, o, label))
        return data

# 4. Clase PieceClassifier: entrena un modelo SVM para clasificar piezas
class PieceClassifier:
    def __init__(self):
        # TODO: Inicializar el modelo SVC con kernel 'rbf' y el escalador StandardScaler
        pass

    def fit(self, pieces):
        # TODO: Extraer características y etiquetas de la lista pieces
        # Escalar las características y entrenar el modelo SVM
        pass

    def predict(self, texture, symmetry, edges, offset):
        # TODO: Escalar las características de la pieza y predecir su etiqueta
        pass

    def evaluate(self, test_data):
        # TODO: Evaluar el modelo con los datos de prueba e imprimir matriz de confusión e informe
        pass

# 5. Clase PieceAnalysisExample: orquesta la generación, entrenamiento, evaluación y visualización
class PieceAnalysisExample:
    def run(self):
        print("Generando dataset...")
        generator = PieceDatasetGenerator(n=400)
        pieces = generator.generate()
        print(f"Dataset generado con {len(pieces)} piezas.")

        print("Dividiendo datos en entrenamiento y prueba...")
        labels = [p.label for p in pieces]
        if len(set(labels)) < 2 or min([labels.count(l) for l in set(labels)]) < 2:
            print("Advertencia: No se puede estratificar. División no estratificada.")
            train, test = train_test_split(pieces, test_size=0.3, random_state=42)
        else:
            train, test = train_test_split(pieces, test_size=0.3, random_state=42, stratify=labels)

        print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
        print(f"Tamaño del conjunto de prueba: {len(test)}")

        print("Inicializando y entrenando clasificador...")
        classifier = PieceClassifier()
        classifier.fit(train)
        print("Modelo entrenado.")

        print("Evaluando modelo en datos de prueba...")
        classifier.evaluate(test)

        print("\nPredicción de pieza personalizada:")
        sample_attrs = (0.45, 0.5, 45, 0.15)
        prediction = classifier.predict(*sample_attrs)
        print(f"Textura: {sample_attrs[0]}, Simetría: {sample_attrs[1]}, Bordes: {sample_attrs[2]}, Offset: {sample_attrs[3]}")
        print(f"Clasificación: {prediction}")

        print("\nVisualizando resultados...")
        df = pd.DataFrame({
            "Textura": [p.texture for p in pieces],
            "Simetría": [p.symmetry for p in pieces],
            "Bordes": [p.edges for p in pieces],
            "Offset": [p.center_offset for p in pieces],
            "Etiqueta": [p.label for p in pieces]
        })

        colores = {"Correcta": "green", "Defectuosa": "red"}

        plt.figure(figsize=(8,6))
        for label, color in colores.items():
            subset = df[df["Etiqueta"] == label]
            plt.scatter(subset["Textura"], subset["Offset"], label=label, c=color, alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.xlabel("Nivel de textura (homogeneidad)")
        plt.ylabel("Desviación del centro de masa")
        plt.title("Clasificación de piezas industriales")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Estado de la pieza")
        plt.xlim(0, 1)
        plt.ylim(0, 0.5)
        plt.show()

if __name__ == "__main__":
    example = PieceAnalysisExample()
    example.run()
