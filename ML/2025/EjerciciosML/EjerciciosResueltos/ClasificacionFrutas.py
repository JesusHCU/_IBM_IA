import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def generar_datos_frutas(num_muestras):
    caracteristicas = []
    etiquetas = []
    frutas = ["Manzana", "Plátano", "Naranja"]
    for _ in range(num_muestras):
        fruta = np.random.choice(frutas)
        if fruta == "Manzana":
            peso = np.random.randint(120, 201)
            tamano = np.random.uniform(7.0, 9.0)
        elif fruta == "Plátano":
            peso = np.random.randint(100, 151)
            tamano = np.random.uniform(12.0, 20.0)
        else:  # Naranja
            peso = np.random.randint(150, 251)
            tamano = np.random.uniform(8.0, 12.0)
        caracteristicas.append([peso, tamano])
        etiquetas.append(fruta)
    return np.array(caracteristicas), np.array(etiquetas)

def entrenar_modelo(data):
    caracteristicas, etiquetas = data
    label_map = {"Manzana": 0, "Plátano": 1, "Naranja": 2}
    etiquetas_numericas = np.array([label_map[etiqueta] for etiqueta in etiquetas])
    X_train, X_test, y_train, y_test = train_test_split(
        caracteristicas, etiquetas_numericas, test_size=0.2, random_state=42
    )
    modelo = KNeighborsClassifier(n_neighbors=3)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision * 100:.2f}%")
    return modelo

def predecir_fruta(modelo, peso, tamano):
    muestra = np.array([[peso, tamano]])
    pred = modelo.predict(muestra)
    inverse_label_map = {0: "Manzana", 1: "Plátano", 2: "Naranja"}
    return inverse_label_map[pred[0]]

if __name__ == "__main__":
    datos = generar_datos_frutas(100)
    modelo = entrenar_modelo(datos)
    peso_nueva_fruta = 150
    tamano_nueva_fruta = 10
    fruta_predicha = predecir_fruta(modelo, peso_nueva_fruta, tamano_nueva_fruta)
    print(f"La fruta con peso {peso_nueva_fruta}g y tamaño {tamano_nueva_fruta}cm es un(a) {fruta_predicha}.")
