"""Comparación de cuatro técnicas de agrupación sobre IMDb Top 1000."""

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
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    rand_score,
    silhouette_score,
)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "Datasets" / "imdb.csv"
FIGURES_DIR = BASE_DIR / "figuras"
FEATURE_COLUMNS = [
    "Runtime",
    "IMDB_Rating",
    "Meta_score",
    "No_of_Votes",
    "Gross",
]
LOG_COLUMNS = ["No_of_Votes", "Gross"]
TARGET_COLUMN = "Genre"
RANDOM_STATE = 42
KMEANS_K_VALUES = (2, 7)
ELBOW_K_VALUES = range(1,11)
DBSCAN_MIN_SAMPLES = 10
DBSCAN_EPS_VALUES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)
CLIQUE_BINS = 2
CLIQUE_MIN_UNIT_POINTS = 5


def load_data(
    path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Carga, convierte y completa las variables usadas en la agrupación."""
    data = pd.read_csv(path)
    numeric = pd.DataFrame(index=data.index)
    numeric["Runtime"] = pd.to_numeric(
        data["Runtime"].astype(str).str.replace(" min", "", regex=False),
        errors="coerce",
    )
    numeric["IMDB_Rating"] = pd.to_numeric(data["IMDB_Rating"], errors="coerce")
    numeric["Meta_score"] = pd.to_numeric(data["Meta_score"], errors="coerce")
    numeric["No_of_Votes"] = pd.to_numeric(
        data["No_of_Votes"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    numeric["Gross"] = pd.to_numeric(
        data["Gross"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    missing_counts = numeric.isna().sum().to_frame("faltantes")
    numeric = numeric.fillna(numeric.median())
    numeric[LOG_COLUMNS] = np.log1p(numeric[LOG_COLUMNS])

    y_true = data[TARGET_COLUMN].fillna("Sin género").astype(str)
    y_primary = y_true.str.split(",").str[0].str.strip()
    return data, numeric[FEATURE_COLUMNS], y_primary, missing_counts


def clique_cluster(
    x_scaled: np.ndarray,
    n_bins: int = CLIQUE_BINS,
    min_unit_points: int = CLIQUE_MIN_UNIT_POINTS,
) -> np.ndarray:
    """Versión sencilla de CLIQUE con unidades densas y vecindad por caras."""
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="uniform",
        subsample=None,
    )
    units = discretizer.fit_transform(x_scaled).astype(int)
    counts = np.zeros((n_bins,) * x_scaled.shape[1], dtype=int)
    for unit in units:
        counts[tuple(unit)] += 1

    dense_units = counts >= min_unit_points
    grid_labels, dense_cluster_count = ndimage.label(
        dense_units,
        structure=ndimage.generate_binary_structure(x_scaled.shape[1], 1),
    )
    if dense_cluster_count < 2:
        raise ValueError(
            "CLIQUE produjo menos de dos regiones densas; ajuste n_bins o "
            "min_unit_points."
        )
    return np.array(
        [
            grid_labels[tuple(unit)] - 1 if grid_labels[tuple(unit)] > 0 else -1
            for unit in units
        ],
        dtype=int,
    )


def clustering_metrics(
    x_scaled: np.ndarray,
    y_true: pd.Series,
    labels: np.ndarray,
) -> dict[str, float | int]:
    """Calcula índices internos sobre no-ruido y externos sobre todos los datos."""
    non_noise = labels != -1
    evaluated_labels = labels[non_noise]
    cluster_count = len(np.unique(evaluated_labels))
    if cluster_count < 2 or cluster_count >= int(non_noise.sum()):
        raise ValueError("La asignación no permite calcular métricas intrínsecas.")
    return {
        "clusters": int(cluster_count),
        "noise": int((~non_noise).sum()),
        "evaluated_points": int(non_noise.sum()),
        "silhouette": silhouette_score(x_scaled[non_noise], evaluated_labels),
        "davies_bouldin": davies_bouldin_score(x_scaled[non_noise], evaluated_labels),
        "calinski_harabasz": calinski_harabasz_score(x_scaled[non_noise], evaluated_labels),
        "rand_index": rand_score(y_true, labels),
        "fowlkes_mallows": fowlkes_mallows_score(y_true, labels),
    }


def evaluate_kmeans_k_values(
    x_scaled: np.ndarray,
    k_values: Iterable[int] = KMEANS_K_VALUES,
) -> pd.DataFrame:
    rows = []
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
    k_values: Iterable[int],
) -> pd.DataFrame:
    """Evalúa K-Means para cada valor de k solicitado."""
    rows = []
    for k in k_values:
        labels = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE,
        ).fit_predict(x_scaled)
        rows.append({"k": k, **clustering_metrics(x_scaled, y_true, labels)})
    return pd.DataFrame(rows)


def select_dbscan(x_scaled: np.ndarray) -> tuple[float, np.ndarray, dict[str, float | int]]:
    """Explora radios y conserva la solución válida con mejor Silhouette."""
    candidates = []
    for eps in DBSCAN_EPS_VALUES:
        labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(x_scaled)
        non_noise = labels != -1
        unique = np.unique(labels[non_noise])
        if len(unique) < 2 or len(unique) >= int(non_noise.sum()):
            continue
        metrics = {
            "clusters": int(len(unique)),
            "noise": int((~non_noise).sum()),
            "evaluated_points": int(non_noise.sum()),
            "silhouette": silhouette_score(x_scaled[non_noise], labels[non_noise]),
        }
        # Se evita seleccionar una solución que excluya más de la mitad del dataset.
        if metrics["noise"] <= len(labels) / 2:
            candidates.append((eps, labels, metrics))
    if not candidates:
        raise ValueError("No se encontró una solución DBSCAN evaluable en los radios probados.")
    return max(candidates, key=lambda item: item[2]["silhouette"])


def select_clique(
    x_scaled: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int], str]:
    # CLIQUE busca subespacios; se prueban subconjuntos descriptivos para evitar
    # que la cuadrícula completa de cinco dimensiones quede demasiado dispersa.
    subspaces = [
        ("IMDB_Rating, Meta_score", [1, 2]),
        ("Runtime, log(No_of_Votes)", [0, 3]),
        ("Runtime, IMDB_Rating, Meta_score", [0, 1, 2]),
        ("Runtime, IMDB_Rating, Meta_score, log(No_of_Votes)", [0, 1, 2, 3]),
        ("Runtime, IMDB_Rating, Meta_score, log(Gross)", [0, 1, 2, 4]),
        ("IMDB_Rating, Meta_score, log(No_of_Votes), log(Gross)", [1, 2, 3, 4]),
    ]
    candidates = []
    diagnostics = []
    for description, indices in subspaces:
        for bins in (2, 3, 4, 5, 6):
            for threshold in (5, 10, 15, 20, 30, 50):
                try:
                    labels = clique_cluster(x_scaled[:, indices], bins, threshold)
                except ValueError as error:
                    diagnostics.append(
                        f"{description}, bins={bins}, min={threshold}: {error}"
                    )
                    continue
                non_noise = labels != -1
                evaluated_labels = labels[non_noise]
                unique_count = len(np.unique(evaluated_labels))
                noise_count = int((~non_noise).sum())
                if (
                    unique_count >= 2
                    and unique_count < int(non_noise.sum())
                    and noise_count <= len(labels) / 2
                ):
                    selection_silhouette = silhouette_score(
                        x_scaled[non_noise],
                        evaluated_labels,
                        sample_size=min(500, int(non_noise.sum())),
                        random_state=RANDOM_STATE,
                    )
                    parameters = (
                        f"subespacio={description}, n_bins={bins}, "
                        f"min_unit_points={threshold}"
                    )
                    candidates.append(
                        (
                            labels,
                            {"silhouette": selection_silhouette},
                            parameters,
                        )
                    )
    if not candidates:
        detail = " | ".join(diagnostics[:8])
        raise ValueError(
            "No se encontró una solución CLIQUE evaluable con los parámetros probados. "
            + detail
        )
    return max(candidates, key=lambda item: item[1]["silhouette"])


def save_cluster_figure(
    path: Path,
    title: str,
    projection: np.ndarray,
    labels: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    color_map = plt.get_cmap("tab10")
    for color_index, label in enumerate(sorted(np.unique(labels).tolist())):
        selected = labels == label
        is_noise = label == -1
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            s=24,
            alpha=0.75,
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
    """Compara en una figura las asignaciones para los valores de k indicados."""
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
                s=20,
                alpha=0.75,
                color=color_map(int(label) % 10),
                label=f"Grupo {label + 1}",
            )
        axis.set_title(f"K-Means (k={k})")
        axis.set_xlabel("Componente principal 1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Componente principal 2")
    axes[-1].legend(title="Asignación", loc="best", fontsize="small")
    figure.suptitle("Comparación de K-Means para distintos valores de k — IMDb")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_kmeans_elbow_figure(
    x_scaled: np.ndarray,
    path: Path,
    dataset_name: str,
) -> None:
    """Guarda la curva de inercia de K-Means para inspección del codo."""
    k_values = list(ELBOW_K_VALUES)
    inertias = [
        KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE,
        ).fit(x_scaled).inertia_
        for k in k_values
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(k_values, inertias, marker="o")
    axis.set_xticks(k_values)
    axis.set_xlabel("Número de grupos (k)")
    axis.set_ylabel("Inercia")
    axis.set_title(f"Método Elbow para K-Means — {dataset_name}")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    data, features, y_true, missing_counts = load_data()
    x_scaled = StandardScaler().fit_transform(features)
    kmeans_sweep = evaluate_kmeans_k_values(x_scaled)
    # K-Means y Agglomerative usan el k que maximiza Silhouette en la exploración.
    selected_k = int(kmeans_sweep.loc[kmeans_sweep["silhouette"].idxmax(), "k"])
    kmeans_comparison_ks = KMEANS_K_VALUES
    kmeans_comparison = evaluate_kmeans_comparison(
        x_scaled,
        y_true,
        kmeans_comparison_ks,
    )

    kmeans_labels = KMeans(
        n_clusters=selected_k,
        n_init=20,
        random_state=RANDOM_STATE,
    ).fit_predict(x_scaled)
    dbscan_eps, dbscan_labels, _ = select_dbscan(x_scaled)
    ward_labels = AgglomerativeClustering(
        n_clusters=selected_k,
        linkage="ward",
    ).fit_predict(x_scaled)
    clique_labels, _, clique_parameters = select_clique(x_scaled)

    algorithms = [
        ("K-Means", "Partición", f"k={selected_k}, n_init=20", kmeans_labels),
        (
            "DBSCAN",
            "Densidad",
            f"eps={dbscan_eps:g}, min_samples={DBSCAN_MIN_SAMPLES}",
            dbscan_labels,
        ),
        (
            "Agglomerative Ward",
            "Jerárquica",
            f"n_clusters={selected_k}, linkage=ward",
            ward_labels,
        ),
        (
            "CLIQUE",
            "Cuadrícula",
            clique_parameters,
            clique_labels,
        ),
    ]
    result_rows = []
    for name, family, parameters, labels in algorithms:
        result_rows.append(
            {
                "algorithm": name,
                "family": family,
                "parameters": parameters,
                **clustering_metrics(x_scaled, y_true, labels),
            }
        )
    results = pd.DataFrame(result_rows)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    projection = PCA(n_components=2).fit_transform(x_scaled)
    figure_names = {
        "K-Means": "imdb-kmeans.png",
        "DBSCAN": "imdb-dbscan.png",
        "Agglomerative Ward": "imdb-agglomerative-ward.png",
        "CLIQUE": "imdb-clique.png",
    }
    for name, _, parameters, labels in algorithms:
        save_cluster_figure(
            FIGURES_DIR / figure_names[name],
            f"{name} — {parameters}\nProyección PCA para visualización",
            projection,
            labels,
        )
    save_kmeans_comparison_figure(
        x_scaled,
        kmeans_comparison_ks,
        FIGURES_DIR / "imdb-kmeans-comparacion-k.png",
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (column, title) in zip(
        axes,
        [
            ("silhouette", "Silhouette"),
            ("davies_bouldin", "Davies-Bouldin"),
            ("calinski_harabasz", "Calinski-Harabasz"),
        ],
    ):
        axis.plot(
            kmeans_sweep["k"],
            kmeans_sweep[column],
            marker="o",
            color="tab:blue",
        )
        axis.set_title(title)
        axis.set_xlabel("k")
        axis.set_xticks(kmeans_sweep["k"])
        axis.grid(alpha=0.2)
    figure.suptitle("Exploración de k para K-Means — IMDb")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "imdb-kmeans-exploracion-k.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    save_kmeans_elbow_figure(
        x_scaled,
        FIGURES_DIR / "imdb-kmeans-elbow.png",
        "IMDb",
    )

    figure, axis = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage(x_scaled, method="ward"),
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=axis,
    )
    axis.set_title("Dendrograma del agrupamiento Agglomerative Ward — IMDb")
    axis.set_xlabel("Observaciones o grupos contraídos")
    axis.set_ylabel("Distancia Ward")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "imdb-agglomerative-ward-dendrograma.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Registros: {len(data)} | Variables de entrada: {', '.join(FEATURE_COLUMNS)}")
    print(f"Géneros primarios para evaluación externa: {y_true.nunique()}")
    print("Valores faltantes originales por variable:")
    print(missing_counts.to_string())
    print("\nDistribución por género primario:")
    print(y_true.value_counts().to_string())
    print("\nExploración de k para K-Means:")
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
    print("\nComparación de K-Means para los valores evaluados:")
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
    print(f"\nParámetros CLIQUE seleccionados: {clique_parameters}")
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
    print(f"\nFiguras guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
