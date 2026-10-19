"""Comparación de cuatro familias de agrupación sobre el dataset Iris.

``species`` se conserva exclusivamente como ground truth. Las técnicas reciben
solo las cuatro variables numéricas del dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    rand_score,
    silhouette_score,
)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


DATASET_PATH = Path(__file__).parent / "Datasets" / "iris.csv"
FIGURES_DIR = Path(__file__).parent / "figuras"
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
RANDOM_STATE = 42

# Estos valores son parámetros experimentales, no etiquetas entregadas al modelo.
# Se fija k=3 para comparar las cuatro familias con el mismo número de grupos.
N_CLUSTERS = 3
KMEANS_COMPARISON_KS = (N_CLUSTERS - 1, N_CLUSTERS, N_CLUSTERS + 1)
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
CLIQUE_BINS = 3
CLIQUE_MIN_UNIT_POINTS = 5


def load_data(
    path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carga el CSV y separa X de las etiquetas reservadas como ground truth."""
    data = pd.read_csv(path)
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas en {path}: {missing}")

    x = data[FEATURE_COLUMNS].copy()
    y_true = data[TARGET_COLUMN].copy()
    return data, x, y_true


def clique_cluster(
    x_scaled: np.ndarray,
    n_bins: int = CLIQUE_BINS,
    min_unit_points: int = CLIQUE_MIN_UNIT_POINTS,
) -> np.ndarray:
    """Aplica una versión sencilla de CLIQUE sobre las cuatro variables.

    CLIQUE divide cada dimensión en intervalos, identifica las unidades densas
    y une las unidades densas adyacentes. Las observaciones que caen en una
    unidad no densa se marcan como ruido (-1). En esta versión se utilizan las
    cuatro dimensiones y conectividad por caras.
    """

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="uniform",
        subsample=None,
    )
    units = discretizer.fit_transform(x_scaled).astype(int)
    grid_shape = (n_bins,) * x_scaled.shape[1]
    counts = np.zeros(grid_shape, dtype=int)
    for unit in units:
        counts[tuple(unit)] += 1

    dense_units = counts >= min_unit_points
    # En CLIQUE se conectan unidades que comparten una cara. Las diagonales no
    # se fusionan para evitar unir regiones que solo se tocan en una esquina.
    grid_labels, dense_cluster_count = ndimage.label(
        dense_units,
        structure=ndimage.generate_binary_structure(x_scaled.shape[1], 1),
    )
    if dense_cluster_count < 2:
        raise ValueError(
            "Los parámetros de CLIQUE producen menos de dos regiones densas; "
            "reduzca min_unit_points o n_bins."
        )

    labels = np.array(
        [
            grid_labels[tuple(unit)] - 1
            if grid_labels[tuple(unit)] > 0
            else -1
            for unit in units
        ],
        dtype=int,
    )
    return labels


def clustering_metrics(
    x_scaled: np.ndarray,
    y_true: pd.Series,
    labels: np.ndarray,
) -> dict[str, float | int]:
    """Calcula tres índices intrínsecos y dos extrínsecos.

    En DBSCAN los puntos con etiqueta ``-1`` son ruido. Las métricas
    intrínsecas se calculan sobre los puntos no considerados ruido, mientras
    que las extrínsecas incluyen el ruido como una agrupación adicional.
    """
    non_noise = labels != -1
    evaluated_labels = labels[non_noise]
    if len(np.unique(evaluated_labels)) < 2:
        raise ValueError("Se necesitan al menos dos grupos para evaluar la agrupación.")

    return {
        "clusters": int(len(np.unique(evaluated_labels))),
        "noise": int((~non_noise).sum()),
        "evaluated_points": int(non_noise.sum()),
        "silhouette": silhouette_score(x_scaled[non_noise], evaluated_labels),
        "davies_bouldin": davies_bouldin_score(
            x_scaled[non_noise], evaluated_labels
        ),
        "calinski_harabasz": calinski_harabasz_score(
            x_scaled[non_noise], evaluated_labels
        ),
        "rand_index": rand_score(y_true, labels),
        "fowlkes_mallows": fowlkes_mallows_score(y_true, labels),
    }


def evaluate_kmeans_k_values(
    x_scaled: np.ndarray,
    k_values: Iterable[int] = range(2, 7),
) -> pd.DataFrame:
    """Explora k sin usar las etiquetas para apoyar la elección de parámetros."""
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        labels = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE,
        ).fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(x_scaled, labels),
                "davies_bouldin": davies_bouldin_score(x_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def evaluate_kmeans_comparison(
    x_scaled: np.ndarray,
    y_true: pd.Series,
    k_values: Iterable[int] = KMEANS_COMPARISON_KS,
) -> pd.DataFrame:
    """Evalúa K-Means con k central y sus vecinos k-1 y k+1."""
    rows = []
    for k in k_values:
        labels = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE,
        ).fit_predict(x_scaled)
        rows.append({"k": k, **clustering_metrics(x_scaled, y_true, labels)})
    return pd.DataFrame(rows)


def fit_algorithm_labels(
    x_scaled: np.ndarray,
) -> list[tuple[str, str, str, np.ndarray]]:
    """Ajusta las cuatro técnicas y devuelve sus etiquetas de agrupación."""
    return [
        (
            "K-Means",
            "Partición",
            f"n_clusters={N_CLUSTERS}, n_init=20",
            KMeans(
                n_clusters=N_CLUSTERS,
                n_init=20,
                random_state=RANDOM_STATE,
            ).fit_predict(x_scaled),
        ),
        (
            "DBSCAN",
            "Densidad",
            f"eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}",
            DBSCAN(
                eps=DBSCAN_EPS,
                min_samples=DBSCAN_MIN_SAMPLES,
            ).fit_predict(x_scaled),
        ),
        (
            "Agglomerative Ward",
            "Jerárquica",
            f"n_clusters={N_CLUSTERS}, linkage=ward",
            AgglomerativeClustering(
                n_clusters=N_CLUSTERS,
                linkage="ward",
            ).fit_predict(x_scaled),
        ),
        (
            "CLIQUE",
            "Cuadrícula",
            f"n_bins={CLIQUE_BINS}, min_unit_points={CLIQUE_MIN_UNIT_POINTS}, 4D",
            clique_cluster(x_scaled),
        ),
    ]


def run_algorithms(
    x_scaled: np.ndarray,
    y_true: pd.Series,
) -> pd.DataFrame:
    """Ejecuta una técnica de cada familia y devuelve sus evaluaciones."""
    rows: list[dict[str, str | float | int]] = []
    for name, family, parameters, labels in fit_algorithm_labels(x_scaled):
        rows.append(
            {
                "algorithm": name,
                "family": family,
                "parameters": parameters,
                **clustering_metrics(x_scaled, y_true, labels),
            }
        )
    return pd.DataFrame(rows)


def save_cluster_figure(
    path: Path,
    title: str,
    projection: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Guarda una vista 2D de las etiquetas obtenidas por una técnica."""
    figure, axis = plt.subplots(figsize=(8, 6))
    color_map = plt.get_cmap("tab10")
    unique_labels = sorted(np.unique(labels).tolist())

    for color_index, label in enumerate(unique_labels):
        selected = labels == label
        is_noise = label == -1
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            s=48,
            alpha=0.85,
            color="black" if is_noise else color_map(color_index % 10),
            marker="x" if is_noise else "o",
            label="Ruido (-1)" if is_noise else f"Grupo {label + 1}",
        )

    axis.set_title(title)
    axis.set_xlabel("Componente principal 1")
    axis.set_ylabel("Componente principal 2")
    axis.legend(title="Asignación", loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_kmeans_comparison_figure(
    x_scaled: np.ndarray,
    k_values: Iterable[int],
    path: Path,
) -> None:
    """Compara en una figura las asignaciones de k-1, k y k+1."""
    projection = PCA(n_components=2).fit_transform(x_scaled)
    k_values = list(k_values)
    figure, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5.5), sharex=True, sharey=True)
    color_map = plt.get_cmap("tab10")
    for axis, k in zip(axes, k_values):
        labels = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE,
        ).fit_predict(x_scaled)
        for label in sorted(np.unique(labels)):
            selected = labels == label
            axis.scatter(
                projection[selected, 0],
                projection[selected, 1],
                s=35,
                alpha=0.8,
                color=color_map(int(label) % 10),
                label=f"Grupo {label + 1}",
            )
        axis.set_title(f"K-Means (k={k})")
        axis.set_xlabel("Componente principal 1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Componente principal 2")
    axes[-1].legend(title="Asignación", loc="best", fontsize="small")
    figure.suptitle("Comparación de K-Means para k-1, k y k+1")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_diagnostic_figures(
    x_scaled: np.ndarray,
    algorithms: list[tuple[str, str, str, np.ndarray]],
    kmeans_sweep: pd.DataFrame,
) -> None:
    """Genera las imágenes que documentan los métodos utilizados."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    projection = PCA(n_components=2).fit_transform(x_scaled)

    figure_names = {
        "K-Means": "kmeans.png",
        "DBSCAN": "dbscan.png",
        "Agglomerative Ward": "agglomerative-ward.png",
        "CLIQUE": "clique.png",
    }
    for name, _, parameters, labels in algorithms:
        save_cluster_figure(
            FIGURES_DIR / figure_names[name],
            f"{name} — {parameters}\nProyección PCA para visualización",
            projection,
            labels,
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("silhouette", "Silhouette", True),
        ("davies_bouldin", "Davies-Bouldin", False),
        ("calinski_harabasz", "Calinski-Harabasz", True),
    ]
    for axis, (column, title, higher_is_better) in zip(axes, metrics):
        axis.plot(kmeans_sweep["k"], kmeans_sweep[column], marker="o")
        best_index = (
            kmeans_sweep[column].idxmax()
            if higher_is_better
            else kmeans_sweep[column].idxmin()
        )
        best_row = kmeans_sweep.loc[best_index]
        axis.scatter(best_row["k"], best_row[column], color="crimson", zorder=3)
        axis.set_title(title)
        axis.set_xlabel("k")
        axis.grid(alpha=0.2)
    figure.suptitle("Exploración de k para K-Means")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "kmeans-exploracion-k.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage(x_scaled, method="ward"),
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=axis,
    )
    axis.set_title("Dendrograma del agrupamiento Agglomerative Ward")
    axis.set_xlabel("Observaciones o grupos contraídos")
    axis.set_ylabel("Distancia Ward")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "agglomerative-ward-dendrograma.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    save_kmeans_comparison_figure(
        x_scaled,
        KMEANS_COMPARISON_KS,
        FIGURES_DIR / "kmeans-comparacion-k.png",
    )


def main() -> None:
    """Ejecuta e imprime la comparación de las cuatro técnicas."""
    data, x, y_true = load_data()
    x_scaled = StandardScaler().fit_transform(x)
    results = run_algorithms(x_scaled, y_true)
    kmeans_sweep = evaluate_kmeans_k_values(x_scaled)
    kmeans_comparison = evaluate_kmeans_comparison(x_scaled, y_true)
    algorithms = fit_algorithm_labels(x_scaled)
    save_diagnostic_figures(x_scaled, algorithms, kmeans_sweep)

    print(f"Registros: {len(data)} | Variables de entrada: {', '.join(FEATURE_COLUMNS)}")
    print(f"Ground truth excluido del ajuste: {TARGET_COLUMN}")
    print("\nEvaluación (mayor es mejor salvo Davies-Bouldin):")
    print(
        results.to_string(
            index=False,
            formatters={
                "silhouette": "{:.4f}".format,
                "davies_bouldin": "{:.4f}".format,
                "calinski_harabasz": "{:.2f}".format,
                "rand_index": "{:.4f}".format,
                "fowlkes_mallows": "{:.4f}".format,
            },
        )
    )

    best_extrinsic = results.loc[results["rand_index"].idxmax()]
    best_silhouette = results.loc[results["silhouette"].idxmax()]
    print(
        f"\nMejor Rand Index: {best_extrinsic['algorithm']} "
        f"({best_extrinsic['rand_index']:.4f})"
    )
    print(
        f"Mejor Silhouette: {best_silhouette['algorithm']} "
        f"({best_silhouette['silhouette']:.4f})"
    )
    print("\nExploración de k para K-Means (sin usar species):")
    print(
        kmeans_sweep.to_string(
            index=False,
            formatters={
                "silhouette": "{:.4f}".format,
                "davies_bouldin": "{:.4f}".format,
                "calinski_harabasz": "{:.2f}".format,
            },
        )
    )
    print("\nComparación de K-Means con k-1, k y k+1:")
    print(
        kmeans_comparison.to_string(
            index=False,
            formatters={
                "silhouette": "{:.4f}".format,
                "davies_bouldin": "{:.4f}".format,
                "calinski_harabasz": "{:.2f}".format,
                "rand_index": "{:.4f}".format,
                "fowlkes_mallows": "{:.4f}".format,
            },
        )
    )
    print(f"\nFiguras guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
