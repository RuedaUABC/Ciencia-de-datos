"""Análisis reproducible de agrupación para ``movies.csv``.

El script aplica cuatro familias de agrupación:

* K-Means (partición),
* DBSCAN (densidad),
* Agglomerative Ward (jerárquica), y
* CLIQUE (cuadrícula).

``genre`` se conserva únicamente como ground truth para las métricas
extrínsecas. El script guarda las figuras en ``U3/figuras`` y genera el
reporte ``U3/movies.md``.
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


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Datasets" / "movies.csv"
FIGURES_DIR = BASE_DIR / "figuras"
REPORT_PATH = BASE_DIR / "movies.md"

# ``year`` se deja fuera para que la época de estreno no domine la distancia.
FEATURE_COLUMNS = ["score", "votes", "gross", "runtime"]
LOG_COLUMNS = ["votes", "gross"]
TARGET_COLUMN = "genre"
RANDOM_STATE = 42
KMEANS_N_INIT = 20
K_VALUES = tuple(range(2, 9))
DBSCAN_MIN_SAMPLES = 10
DBSCAN_EPS_VALUES = (0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
CLIQUE_BINS_VALUES = tuple(range(4, 13))
# Se mantiene el mínimo de cinco observaciones por unidad usado en U3.md.
CLIQUE_THRESHOLDS = (5, 8, 10, 15, 20, 30, 50, 75, 100)
CLIQUE_SUBSPACES = (
    ("score, log(votes)", [0, 1]),
    ("score, log(gross)", [0, 2]),
    ("score, runtime", [0, 3]),
    ("score, log(votes), log(gross)", [0, 1, 2]),
    ("score, log(votes), runtime", [0, 1, 3]),
    ("score, log(gross), runtime", [0, 2, 3]),
    ("score, log(votes), log(gross), runtime", [0, 1, 2, 3]),
)
METRIC_SAMPLE_SIZE = 3000
DENDROGRAM_SAMPLE_SIZE = 1200


def load_data(
    path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """Carga el CSV, elimina filas incompletas y prepara las variables.

    Las filas con algún dato faltante se eliminan porque esa es la operación
    solicitada para ``movies.csv``. La función también conserva un conteo de
    faltantes detectados al cargar el archivo para documentarlo en el reporte.
    """

    data = pd.read_csv(path, na_values=["", "NA", "NaN", "null", "none"])
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas en {path}: {missing}")

    missing_counts = data.isna().sum()
    incomplete_rows = data.isna().any(axis=1)
    rows_removed = int(incomplete_rows.sum())
    data = data.loc[~incomplete_rows].copy().reset_index(drop=True)

    numeric = pd.DataFrame(index=data.index)
    for column in FEATURE_COLUMNS:
        numeric[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_numeric = numeric.isna().any(axis=1)
    if invalid_numeric.any():
        rows_removed += int(invalid_numeric.sum())
        data = data.loc[~invalid_numeric].copy().reset_index(drop=True)
        numeric = numeric.loc[~invalid_numeric].copy().reset_index(drop=True)

    if data.empty:
        raise ValueError("No quedaron registros válidos después de la limpieza.")

    numeric[LOG_COLUMNS] = np.log1p(numeric[LOG_COLUMNS])
    y_true = data[TARGET_COLUMN].astype(str)
    return data, numeric, y_true, missing_counts, rows_removed


def scale_features(features: pd.DataFrame) -> np.ndarray:
    """Estandariza las variables para que estén en escalas comparables."""

    return StandardScaler().fit_transform(features)


def clique_cluster(
    x_scaled: np.ndarray,
    indices: list[int],
    n_bins: int,
    min_unit_points: int,
) -> np.ndarray:
    """Aplica una versión sencilla de CLIQUE con conectividad por caras."""

    subspace = x_scaled[:, indices]
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="uniform",
        subsample=None,
    )
    units = discretizer.fit_transform(subspace).astype(int)
    counts = np.zeros((n_bins,) * subspace.shape[1], dtype=int)
    for unit in units:
        counts[tuple(unit)] += 1

    dense_units = counts >= min_unit_points
    structure = ndimage.generate_binary_structure(subspace.shape[1], 1)
    grid_labels, dense_cluster_count = ndimage.label(dense_units, structure=structure)
    if dense_cluster_count < 2:
        raise ValueError("CLIQUE produjo menos de dos regiones densas.")

    return np.array(
        [
            grid_labels[tuple(unit)] - 1
            if grid_labels[tuple(unit)] > 0
            else -1
            for unit in units
        ],
        dtype=int,
    )


def metric_sample(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce la muestra únicamente para métricas costosas de distancia."""

    if len(x) <= METRIC_SAMPLE_SIZE:
        return x, labels
    rng = np.random.default_rng(RANDOM_STATE)
    # Se conserva al menos una observación de cada grupo para que una muestra
    # aleatoria no elimine por accidente un grupo pequeño.
    selected_by_group = [
        rng.choice(np.flatnonzero(labels == label), size=1)[0]
        for label in np.unique(labels)
    ]
    remaining = np.setdiff1d(np.arange(len(x)), selected_by_group, assume_unique=False)
    extra = rng.choice(
        remaining,
        size=METRIC_SAMPLE_SIZE - len(selected_by_group),
        replace=False,
    )
    selected = np.concatenate([np.asarray(selected_by_group), extra])
    return x[selected], labels[selected]


def clustering_metrics(
    x_scaled: np.ndarray,
    y_true: pd.Series,
    labels: np.ndarray,
) -> dict[str, float | int]:
    """Calcula tres métricas intrínsecas y dos extrínsecas.

    Las métricas intrínsecas ignoran las observaciones marcadas como ruido por
    DBSCAN o CLIQUE. Rand Index y Fowlkes-Mallows incluyen todas las
    observaciones, considerando ``-1`` como una asignación adicional.
    """

    non_noise = labels != -1
    evaluated_x = x_scaled[non_noise]
    evaluated_labels = labels[non_noise]
    unique_labels = np.unique(evaluated_labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(evaluated_labels):
        raise ValueError("La asignación no permite calcular métricas intrínsecas.")

    sampled_x, sampled_labels = metric_sample(evaluated_x, evaluated_labels)
    return {
        "clusters": int(len(unique_labels)),
        "noise": int((~non_noise).sum()),
        "evaluated_points": int(non_noise.sum()),
        "silhouette": silhouette_score(sampled_x, sampled_labels),
        "davies_bouldin": davies_bouldin_score(sampled_x, sampled_labels),
        "calinski_harabasz": calinski_harabasz_score(sampled_x, sampled_labels),
        "rand_index": rand_score(y_true, labels),
        "fowlkes_mallows": fowlkes_mallows_score(y_true, labels),
    }


def evaluate_kmeans_k_values(
    x_scaled: np.ndarray,
    k_values: Iterable[int] = K_VALUES,
) -> pd.DataFrame:
    """Explora varios valores de k sin usar el ground truth."""

    rows: list[dict[str, float | int]] = []
    for k in k_values:
        labels = KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
        ).fit_predict(x_scaled)
        sampled_x, sampled_labels = metric_sample(x_scaled, labels)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(sampled_x, sampled_labels),
                "davies_bouldin": davies_bouldin_score(sampled_x, sampled_labels),
                "calinski_harabasz": calinski_harabasz_score(
                    sampled_x, sampled_labels
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_kmeans_comparison(
    x_scaled: np.ndarray,
    y_true: pd.Series,
    k_values: Iterable[int],
) -> pd.DataFrame:
    """Calcula las métricas de K-Means para los valores seleccionados."""

    rows = []
    for k in k_values:
        labels = KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
        ).fit_predict(x_scaled)
        rows.append({"k": k, **clustering_metrics(x_scaled, y_true, labels)})
    return pd.DataFrame(rows)


def select_dbscan(
    x_scaled: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, float | int]]:
    """Selecciona el DBSCAN válido con mejor Silhouette."""

    candidates = []
    for eps in DBSCAN_EPS_VALUES:
        labels = DBSCAN(
            eps=eps,
            min_samples=DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        ).fit_predict(x_scaled)
        non_noise = labels != -1
        evaluated_labels = labels[non_noise]
        if len(evaluated_labels) == 0:
            continue
        if len(np.unique(evaluated_labels)) < 2:
            continue
        if len(np.unique(evaluated_labels)) >= len(evaluated_labels):
            continue
        if int((~non_noise).sum()) > len(labels) / 2:
            continue
        sampled_x, sampled_labels = metric_sample(x_scaled[non_noise], evaluated_labels)
        candidates.append(
            (
                eps,
                labels,
                {
                    "silhouette": silhouette_score(sampled_x, sampled_labels),
                    "noise": int((~non_noise).sum()),
                    "clusters": int(len(np.unique(evaluated_labels))),
                },
            )
        )

    if not candidates:
        raise ValueError("No se encontró una solución DBSCAN evaluable.")
    return max(candidates, key=lambda item: item[2]["silhouette"])


def select_clique(
    x_scaled: np.ndarray,
) -> tuple[np.ndarray, str, dict[str, float | int]]:
    """Busca una solución CLIQUE evaluable entre varios subespacios."""

    candidates = []
    for description, indices in CLIQUE_SUBSPACES:
        for bins in CLIQUE_BINS_VALUES:
            for threshold in CLIQUE_THRESHOLDS:
                try:
                    labels = clique_cluster(x_scaled, indices, bins, threshold)
                except ValueError:
                    continue
                non_noise = labels != -1
                evaluated_labels = labels[non_noise]
                unique_count = len(np.unique(evaluated_labels))
                noise_count = int((~non_noise).sum())
                if unique_count < 2 or unique_count >= len(evaluated_labels):
                    continue
                if noise_count > len(labels) / 2:
                    continue
                sampled_x, sampled_labels = metric_sample(
                    x_scaled[non_noise], evaluated_labels
                )
                candidates.append(
                    (
                        labels,
                        (
                            f"subespacio={description}, n_bins={bins}, "
                            f"min_unit_points={threshold}"
                        ),
                        {
                            "silhouette": silhouette_score(sampled_x, sampled_labels),
                            "noise": noise_count,
                            "clusters": unique_count,
                        },
                    )
                )

    if not candidates:
        raise ValueError("No se encontró una solución CLIQUE evaluable.")
    return max(candidates, key=lambda item: item[2]["silhouette"])


def fit_algorithms(
    x_scaled: np.ndarray,
    y_true: pd.Series,
    selected_k: int,
) -> tuple[list[tuple[str, str, str, np.ndarray]], pd.DataFrame]:
    """Ajusta los cuatro algoritmos y crea la tabla final de resultados."""

    kmeans_labels = KMeans(
        n_clusters=selected_k,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    ).fit_predict(x_scaled)
    dbscan_eps, dbscan_labels, _ = select_dbscan(x_scaled)
    ward_labels = AgglomerativeClustering(
        n_clusters=selected_k,
        linkage="ward",
    ).fit_predict(x_scaled)
    clique_labels, clique_parameters, _ = select_clique(x_scaled)

    algorithms = [
        (
            "K-Means",
            "Partición",
            f"k={selected_k}, n_init={KMEANS_N_INIT}",
            kmeans_labels,
        ),
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
        ("CLIQUE", "Cuadrícula", clique_parameters, clique_labels),
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
    return algorithms, pd.DataFrame(result_rows)


def save_cluster_figure(
    path: Path,
    title: str,
    projection: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Guarda las asignaciones en una proyección PCA de dos dimensiones."""

    figure, axis = plt.subplots(figsize=(8, 6))
    color_map = plt.get_cmap("tab10")
    for color_index, label in enumerate(sorted(np.unique(labels).tolist())):
        selected = labels == label
        is_noise = label == -1
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            s=18,
            alpha=0.65,
            color="black" if is_noise else color_map(color_index % 10),
            marker="x" if is_noise else "o",
            label="Ruido (-1)" if is_noise else f"Grupo {label + 1}",
        )
    axis.set_title(title)
    axis.set_xlabel("Componente principal 1")
    axis.set_ylabel("Componente principal 2")
    axis.legend(title="Asignación", loc="best", fontsize="small")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_kmeans_exploration(sweep: pd.DataFrame, path: Path) -> None:
    """Grafica las tres métricas internas exploradas para K-Means."""

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("silhouette", "Silhouette", True),
        ("davies_bouldin", "Davies-Bouldin", False),
        ("calinski_harabasz", "Calinski-Harabasz", True),
    ]
    for axis, (column, title, maximize) in zip(axes, metrics):
        axis.plot(sweep["k"], sweep[column], marker="o")
        best_index = sweep[column].idxmax() if maximize else sweep[column].idxmin()
        best_row = sweep.loc[best_index]
        axis.scatter(best_row["k"], best_row[column], color="crimson", zorder=3)
        axis.set_title(title)
        axis.set_xlabel("k")
        axis.set_xticks(sweep["k"])
        axis.grid(alpha=0.2)
    figure.suptitle("Exploración de k para K-Means — Movies")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_kmeans_elbow(x_scaled: np.ndarray, path: Path) -> None:
    """Guarda la curva de inercia para los valores explorados de k."""

    inertias = [
        KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
        ).fit(x_scaled).inertia_
        for k in K_VALUES
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(K_VALUES, inertias, marker="o")
    axis.set_xticks(K_VALUES)
    axis.set_xlabel("Número de grupos (k)")
    axis.set_ylabel("Inercia")
    axis.set_title("Método Elbow para K-Means — Movies")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_kmeans_comparison(
    x_scaled: np.ndarray,
    k_values: Iterable[int],
    path: Path,
) -> None:
    """Compara las asignaciones de K-Means para varios valores de k."""

    projection = PCA(n_components=2).fit_transform(x_scaled)
    k_values = list(k_values)
    figure, axes = plt.subplots(
        1,
        len(k_values),
        figsize=(6 * len(k_values), 5.5),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    color_map = plt.get_cmap("tab10")
    for axis, k in zip(axes, k_values):
        labels = KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
        ).fit_predict(x_scaled)
        for label in sorted(np.unique(labels)):
            selected = labels == label
            axis.scatter(
                projection[selected, 0],
                projection[selected, 1],
                s=14,
                alpha=0.6,
                color=color_map(int(label) % 10),
                label=f"Grupo {label + 1}",
            )
        axis.set_title(f"K-Means (k={k})")
        axis.set_xlabel("Componente principal 1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Componente principal 2")
    axes[-1].legend(title="Asignación", loc="best", fontsize="small")
    figure.suptitle("Comparación de K-Means para distintos valores de k — Movies")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_dendrogram(x_scaled: np.ndarray, path: Path) -> None:
    """Guarda un dendrograma sobre una muestra para mantenerlo legible."""

    sample_size = min(DENDROGRAM_SAMPLE_SIZE, len(x_scaled))
    rng = np.random.default_rng(RANDOM_STATE)
    selected = rng.choice(len(x_scaled), size=sample_size, replace=False)
    tree = linkage(x_scaled[selected], method="ward")
    figure, axis = plt.subplots(figsize=(12, 6))
    dendrogram(
        tree,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=axis,
    )
    axis.set_title(
        f"Dendrograma Agglomerative Ward — muestra de {sample_size} películas"
    )
    axis.set_xlabel("Observaciones o grupos contraídos")
    axis.set_ylabel("Distancia Ward")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_figures(
    x_scaled: np.ndarray,
    algorithms: list[tuple[str, str, str, np.ndarray]],
    sweep: pd.DataFrame,
    comparison_k: Iterable[int],
) -> None:
    """Genera todas las figuras referenciadas por el reporte."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    projection = PCA(n_components=2).fit_transform(x_scaled)
    names = {
        "K-Means": "movies-kmeans.png",
        "DBSCAN": "movies-dbscan.png",
        "Agglomerative Ward": "movies-agglomerative-ward.png",
        "CLIQUE": "movies-clique.png",
    }
    for name, _, parameters, labels in algorithms:
        save_cluster_figure(
            FIGURES_DIR / names[name],
            f"{name} — {parameters}\nProyección PCA para visualización",
            projection,
            labels,
        )
    save_kmeans_exploration(sweep, FIGURES_DIR / "movies-kmeans-exploracion-k.png")
    save_kmeans_elbow(x_scaled, FIGURES_DIR / "movies-kmeans-elbow.png")
    save_kmeans_comparison(
        x_scaled,
        comparison_k,
        FIGURES_DIR / "movies-kmeans-comparacion-k.png",
    )
    save_dendrogram(
        x_scaled,
        FIGURES_DIR / "movies-agglomerative-ward-dendrograma.png",
    )


def fmt(value: object, decimals: int = 4) -> str:
    """Formatea números para tablas Markdown."""

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def markdown_table(
    rows: Iterable[Iterable[object]],
    headers: Iterable[str],
    decimals: int = 4,
) -> str:
    """Convierte filas sencillas en una tabla Markdown."""

    headers = list(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value, decimals) for value in row) + " |")
    return "\n".join(lines)


def frame_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str] | None = None,
) -> str:
    """Convierte un DataFrame en tabla usando columnas previamente elegidas."""

    headers = headers or columns
    return markdown_table(frame[columns].itertuples(index=False, name=None), headers)


def render_report(
    data: pd.DataFrame,
    features: pd.DataFrame,
    y_true: pd.Series,
    missing_counts: pd.Series,
    rows_removed: int,
    selected_k: int,
    sweep: pd.DataFrame,
    comparison: pd.DataFrame,
    results: pd.DataFrame,
    algorithms: list[tuple[str, str, str, np.ndarray]],
) -> str:
    """Construye el reporte Markdown con los resultados ejecutados."""

    distribution = y_true.value_counts().rename_axis("género").reset_index(name="cantidad")
    missing_rows = [(column, int(value)) for column, value in missing_counts.items()]
    distribution_rows = distribution.itertuples(index=False, name=None)
    result_columns = [
        "algorithm",
        "family",
        "clusters",
        "noise",
        "evaluated_points",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "rand_index",
        "fowlkes_mallows",
    ]
    result_headers = [
        "algoritmo",
        "familia",
        "grupos",
        "ruido",
        "puntos evaluados",
        "Silhouette",
        "Davies-Bouldin",
        "Calinski-Harabasz",
        "Rand Index",
        "Fowlkes-Mallows",
    ]
    algorithm_parameters = {
        name: parameters for name, _, parameters, _ in algorithms
    }
    selected_row = results.loc[results["algorithm"] == "K-Means"].iloc[0]
    best_silhouette = results.loc[results["silhouette"].idxmax()]
    best_db = results.loc[results["davies_bouldin"].idxmin()]
    best_ch = results.loc[results["calinski_harabasz"].idxmax()]
    best_rand = results.loc[results["rand_index"].idxmax()]
    best_fm = results.loc[results["fowlkes_mallows"].idxmax()]

    missing_table = markdown_table(
        missing_rows,
        ["variable", "faltantes al cargar"],
        decimals=0,
    )
    distribution_table = markdown_table(
        distribution_rows,
        ["género", "cantidad"],
        decimals=0,
    )
    sweep_table = frame_table(
        sweep,
        ["k", "silhouette", "davies_bouldin", "calinski_harabasz"],
        ["k", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
    )
    comparison_table = frame_table(
        comparison,
        [
            "k",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "rand_index",
            "fowlkes_mallows",
        ],
        [
            "k",
            "Silhouette",
            "Davies-Bouldin",
            "Calinski-Harabasz",
            "Rand Index",
            "Fowlkes-Mallows",
        ],
    )
    results_table = frame_table(results, result_columns, result_headers)

    return f"""# Análisis de agrupación del dataset Movies

## Objetivo

Aplicar cuatro técnicas de agrupación al dataset `movies.csv`, comparar la
calidad de sus grupos con métricas intrínsecas y contrastar las asignaciones
contra `genre` como `ground truth` mediante métricas extrínsecas.

El reporte fue generado automáticamente por [`movies.py`](movies.py). No se
utiliza `genre` durante el entrenamiento de ningún algoritmo.

## Dataset

El archivo contiene **{len(data):,} películas** y 14 columnas. La versión de
entrada ya se encuentra depurada; durante esta ejecución se eliminaron
{rows_removed} filas adicionales con datos faltantes o valores numéricos
inválidos.

Para evitar que el año de estreno domine las distancias, `year` no se utiliza
como variable de entrada. Las variables empleadas son `score`, `votes`,
`gross` y `runtime`. `votes` y `gross` se transforman con `log1p` porque tienen
una distribución muy asimétrica; después todas las variables se estandarizan.

### Faltantes

{missing_table}

No se imputaron datos: las filas incompletas se eliminaron antes de formar la
matriz de entrada.

### Distribución del ground truth

`genre` tiene {y_true.nunique()} géneros. Se utiliza como una referencia
externa, no como variable de entrenamiento.

{distribution_table}

## Técnicas utilizadas

### K-Means

K-Means asigna cada observación al centroide más cercano y actualiza los
centroides hasta estabilizar la solución. Se exploraron los valores de
`k={min(K_VALUES)}` a `k={max(K_VALUES)}` usando `n_init={KMEANS_N_INIT}`; se
seleccionó `k={selected_k}` por obtener el mayor Silhouette en la exploración.

{sweep_table}

![Exploración de k para K-Means](figuras/movies-kmeans-exploracion-k.png)

*Figura 1. Métricas intrínsecas al variar `k`; el punto rojo marca el mejor
valor de cada métrica.*

![Método Elbow para Movies](figuras/movies-kmeans-elbow.png)

*Figura 2. Inercia de K-Means para los valores explorados.*

La comparación detallada de los valores seleccionados quedó así:

{comparison_table}

![Comparación de K-Means](figuras/movies-kmeans-comparacion-k.png)

*Figura 3. Proyección PCA común para comparar las asignaciones de K-Means.*

### DBSCAN

DBSCAN identifica regiones densas a partir de un radio `eps` y un mínimo de
vecinos. Las observaciones que no pertenecen a una región densa se marcan
como ruido (`-1`). Se probaron los radios configurados en el script y se
conservó la solución válida con mejor Silhouette, sin permitir que el ruido
superara la mitad del dataset.

Parámetros seleccionados: **{algorithm_parameters['DBSCAN']}**.

![Agrupación de Movies con DBSCAN](figuras/movies-dbscan.png)

*Figura 4. Resultado de DBSCAN; las cruces negras representan el ruido.*

### Agglomerative Ward

El método jerárquico comienza con una observación por grupo y va uniendo los
grupos que producen el menor aumento de variación interna. Se utilizó Ward
con distancia euclidiana y el mismo `k={selected_k}` seleccionado para
K-Means. El dendrograma se generó con una muestra reproducible de hasta
{DENDROGRAM_SAMPLE_SIZE} películas para mantener la figura legible.

Parámetros: **{algorithm_parameters['Agglomerative Ward']}**.

![Agrupación de Movies con Agglomerative Ward](figuras/movies-agglomerative-ward.png)

*Figura 5. Resultado de Agglomerative Ward en la proyección PCA.*

![Dendrograma de Agglomerative Ward](figuras/movies-agglomerative-ward-dendrograma.png)

*Figura 6. Dendrograma truncado sobre una muestra reproducible.*

### CLIQUE

CLIQUE divide cada variable en intervalos, cuenta las observaciones de cada
celda y conecta las celdas densas que comparten una cara. Las observaciones
que caen en celdas no densas se marcan como ruido. Se exploraron varios
subespacios, números de intervalos y umbrales, conservando la solución válida
con mayor Silhouette.

Parámetros seleccionados: **{algorithm_parameters['CLIQUE']}**.

![Agrupación de Movies con CLIQUE](figuras/movies-clique.png)

*Figura 7. Resultado de CLIQUE en el subespacio elegido; las cruces negras
representan el ruido.*

## Métricas

Las métricas intrínsecas se calcularon sobre las observaciones no marcadas
como ruido:

* **Silhouette:** valores altos indican que las observaciones están cerca de su
  propio grupo y lejos de los demás.
* **Davies-Bouldin:** valores bajos indican grupos compactos y separados.
* **Calinski-Harabasz:** valores altos indican mayor separación entre grupos
  frente a la dispersión interna.

Rand Index y Fowlkes-Mallows son métricas extrínsecas: comparan las
asignaciones contra `genre`. En estas dos métricas sí se incluyen las
observaciones marcadas como ruido, tratándolas como la etiqueta `-1`.

Para que el cálculo sea reproducible y manejable en este dataset, las métricas
de distancia usan una muestra aleatoria fija de hasta {METRIC_SAMPLE_SIZE}
observaciones cuando hay más datos; Rand Index y Fowlkes-Mallows se calculan
con todos los registros.

## Resultados

Se utilizó `random_state={RANDOM_STATE}` cuando el algoritmo lo permite.

{results_table}

## Conclusiones

* **Mejor Silhouette:** {best_silhouette['algorithm']} con
  `{best_silhouette['silhouette']:.4f}`. Sus grupos son los más compactos y
  separados bajo esta métrica, considerando únicamente los puntos sin ruido.
* **Mejor Davies-Bouldin:** {best_db['algorithm']} con
  `{best_db['davies_bouldin']:.4f}`; un valor menor es mejor.
* **Mejor Calinski-Harabasz:** {best_ch['algorithm']} con
  `{best_ch['calinski_harabasz']:.2f}`.
* **Mejor Rand Index:** {best_rand['algorithm']} con
  `{best_rand['rand_index']:.4f}`.
* **Mejor Fowlkes-Mallows:** {best_fm['algorithm']} con
  `{best_fm['fowlkes_mallows']:.4f}`.

Las métricas extrínsecas deben interpretarse con cuidado: `genre` es una
categoría editorial, tiene una distribución desbalanceada y no necesariamente
representa la geometría numérica formada por calificación, popularidad,
recaudación y duración. Por eso, la elección final del algoritmo debe
considerar simultáneamente la calidad geométrica, la cantidad de ruido y el
objetivo del análisis.
"""


def main() -> None:
    """Ejecuta el análisis, guarda figuras y escribe el reporte Markdown."""

    data, features, y_true, missing_counts, rows_removed = load_data()
    x_scaled = scale_features(features)
    sweep = evaluate_kmeans_k_values(x_scaled)
    selected_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    comparison_k = sorted({max(2, selected_k - 1), selected_k, selected_k + 1})
    comparison = evaluate_kmeans_comparison(x_scaled, y_true, comparison_k)
    algorithms, results = fit_algorithms(x_scaled, y_true, selected_k)
    save_figures(x_scaled, algorithms, sweep, comparison_k)

    report = render_report(
        data,
        features,
        y_true,
        missing_counts,
        rows_removed,
        selected_k,
        sweep,
        comparison,
        results,
        algorithms,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Registros analizados: {len(data)}")
    print(f"Variables de entrada: {', '.join(FEATURE_COLUMNS)}")
    print(f"K seleccionado: {selected_k}")
    print("\nResultados:")
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
    print(f"\nReporte guardado en: {REPORT_PATH}")
    print(f"Figuras guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
