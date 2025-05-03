import numpy as np
from typing import Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def generar_datos_clientes(
        num_muestras: int,
        num_categorias: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    monto = rng.uniform(100, 10000, size=num_muestras)
    frecuencia = rng.integers(1, 101, size=num_muestras)

    compras_cat = np.zeros((num_muestras, num_categorias))
    for i in range(num_muestras):
        if frecuencia[i] > 0 and num_categorias > 0:
            compras_cat[i] = rng.multinomial(
                frecuencia[i],
                np.ones(num_categorias) / num_categorias
            )

    X = np.hstack([
        monto.reshape(-1, 1),
        frecuencia.reshape(-1, 1),
        compras_cat
    ])
    y_dummy = np.zeros(num_muestras, dtype=int)
    return X, y_dummy


def encontrar_numero_optimo_clusters(
        data: np.ndarray,
        k_max: int = 10
) -> int:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    best_k = 2
    best_score = -1.0

    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def entrenar_modelo_cluster(
        data: np.ndarray,
        n_clusters: Optional[int] = None,
        k_max: int = 10
) -> Tuple[int, KMeans]:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    if n_clusters is None:
        n_clusters = encontrar_numero_optimo_clusters(scaled_data, k_max)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(scaled_data)
    model.scaler_ = scaler

    return n_clusters, model


def predecir_cluster(*args) -> int:
    if len(args) == 2:
        model, sample = args
    elif len(args) == 3:
        _, model, sample = args

    sample = np.array(sample).reshape(1, -1)
    if hasattr(model, 'scaler_'):
        sample = model.scaler_.transform(sample)

    return int(model.predict(sample)[0])