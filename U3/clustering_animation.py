"""Animaciones educativas y reutilizables para algoritmos de clustering.

El módulo separa la simulación del algoritmo de su representación gráfica:
los generadores ``*_snapshots`` producen estados intermedios que pueden
inspeccionarse, probarse o reutilizarse en otra interfaz. Las funciones
``animate_*`` consumen esos estados con :class:`matplotlib.animation.FuncAnimation`
y los guardan como GIF usando ``PillowWriter``.

Las funciones trabajan con puntos bidimensionales. Para datos con más
variables, se recomienda proyectarlos previamente (por ejemplo, con PCA) para
que el movimiento de los grupos pueda verse en una diapositiva.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# El módulo guarda GIF y también debe funcionar cuando se importa directamente
# desde una terminal o CI sin servidor gráfico.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from scipy.cluster.hierarchy import linkage


__all__ = [
    "AgglomerativeSnapshot",
    "CLIQUEAnimationSnapshot",
    "DBSCANSnapshot",
    "KMeansSnapshot",
    "agglomerative_snapshots",
    "animate_agglomerative",
    "animate_clique",
    "animate_dbscan",
    "animate_kmeans",
    "clique_snapshots",
    "dbscan_snapshots",
    "kmeans_snapshots",
    "save_clustering_animations",
]


@dataclass(frozen=True)
class KMeansSnapshot:
    """Estado de K-Means durante una asignación o actualización."""

    labels: np.ndarray
    centers: np.ndarray
    iteration: int
    phase: str
    old_centers: np.ndarray | None = None


@dataclass(frozen=True)
class AgglomerativeSnapshot:
    """Estado de Ward y cajas delimitadoras de los grupos actuales."""

    labels: np.ndarray
    bounds: tuple[tuple[float, float, float, float], ...]
    n_clusters: int
    merged: tuple[int, int] | None
    distance: float | None


@dataclass(frozen=True)
class DBSCANSnapshot:
    """Estado de la expansión de DBSCAN."""

    labels: np.ndarray
    visited: np.ndarray
    core_mask: np.ndarray
    current_point: int | None
    neighbors: np.ndarray
    queue: np.ndarray
    cluster_id: int | None
    phase: str


@dataclass(frozen=True)
class CLIQUEAnimationSnapshot:
    """Estado de la discretización, filtrado o unión de celdas de CLIQUE."""

    labels: np.ndarray
    cell_indices: np.ndarray
    counts: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    dense_cells: frozenset[tuple[int, int]]
    active_cells: frozenset[tuple[int, int]]
    phase: str


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Valida y normaliza una matriz de puntos bidimensionales."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points debe tener forma (n_observaciones, 2).")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("points debe contener al menos un punto finito.")
    return values


def _sample_points(
    points: np.ndarray,
    max_points: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce una nube grande y devuelve también los índices conservados."""

    if max_points is None or len(points) <= max_points:
        return points.copy(), np.arange(len(points))
    if max_points < 2:
        raise ValueError("max_points debe ser al menos 2.")
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[indices].copy(), indices


def _point_colors(labels: np.ndarray, cmap_name: str = "tab20") -> np.ndarray:
    """Convierte etiquetas en colores: gris no visitado y negro para ruido."""

    labels = np.asarray(labels)
    colors = np.tile(mcolors.to_rgba("lightgray"), (len(labels), 1))
    color_map = plt.get_cmap(cmap_name)
    assigned = labels >= 0
    if np.any(assigned):
        colors[assigned] = [
            color_map(int(label) % color_map.N) for label in labels[assigned]
        ]
    colors[labels == -1] = mcolors.to_rgba("black")
    return colors


def _cluster_colors(count: int, cmap_name: str = "tab20") -> list:
    """Genera una paleta estable para los grupos de una animación."""

    color_map = plt.get_cmap(cmap_name)
    return [color_map(index % color_map.N) for index in range(count)]


def _axis_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calcula límites con un margen visual estable."""

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    margin = span * 0.12
    return (
        (float(minimum[0] - margin[0]), float(maximum[0] + margin[0])),
        (float(minimum[1] - margin[1]), float(maximum[1] + margin[1])),
    )


def _configure_axis(axis: plt.Axes, points: np.ndarray, title: str) -> None:
    """Configura un eje común para las cuatro animaciones."""

    (x_min, x_max), (y_min, y_max) = _axis_limits(points)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Componente 1")
    axis.set_ylabel("Componente 2")
    axis.set_title(title)
    axis.grid(alpha=0.2)


def _save_animation(
    figure: plt.Figure,
    update: Callable[[int], tuple],
    frame_count: int,
    output_path: Path,
    fps: int,
    dpi: int,
) -> Path:
    """Crea y guarda una animación GIF, cerrando la figura al terminar."""

    if frame_count == 0:
        plt.close(figure)
        raise ValueError("No se puede guardar una animación sin frames.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        repeat=False,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    return output_path


def kmeans_snapshots(
    points: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 20,
    random_state: int = 42,
    tolerance: float = 1e-4,
) -> Iterator[KMeansSnapshot]:
    """Genera estados de K-Means: asignación y movimiento de centroides.

    La implementación es deliberadamente explícita para que cada paso del
    algoritmo sea visible en una diapositiva. Los centros iniciales se toman
    de observaciones reales y los clusters vacíos conservan su centro.
    """

    values = _validate_points(points)
    if not 1 <= n_clusters <= len(values):
        raise ValueError("n_clusters debe estar entre 1 y el número de puntos.")
    if max_iter < 1:
        raise ValueError("max_iter debe ser positivo.")

    rng = np.random.default_rng(random_state)
    centers = values[rng.choice(len(values), size=n_clusters, replace=False)].copy()
    labels = np.full(len(values), -1, dtype=int)
    yield KMeansSnapshot(labels.copy(), centers.copy(), 0, "centros iniciales")

    for iteration in range(1, max_iter + 1):
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        yield KMeansSnapshot(
            labels.copy(),
            centers.copy(),
            iteration,
            "reasignación al centroide más cercano",
        )

        new_centers = np.array(
            [
                values[labels == cluster].mean(axis=0)
                if np.any(labels == cluster)
                else centers[cluster]
                for cluster in range(n_clusters)
            ]
        )
        displacement = np.linalg.norm(new_centers - centers, axis=1)
        yield KMeansSnapshot(
            labels.copy(),
            new_centers.copy(),
            iteration,
            f"actualización de centroides (desplazamiento máximo: "
            f"{displacement.max():.3f})",
            old_centers=centers.copy(),
        )
        if np.all(displacement <= tolerance):
            break
        centers = new_centers


def agglomerative_snapshots(
    points: np.ndarray,
    n_clusters: int,
) -> Iterator[AgglomerativeSnapshot]:
    """Genera las fusiones Ward desde ``n`` clusters hasta ``n_clusters``."""

    values = _validate_points(points)
    if not 1 <= n_clusters <= len(values):
        raise ValueError("n_clusters debe estar entre 1 y el número de puntos.")

    members: dict[int, list[int]] = {
        index: [index] for index in range(len(values))
    }

    def state(
        merged: tuple[int, int] | None,
        distance: float | None,
    ) -> AgglomerativeSnapshot:
        active = sorted(members)
        label_by_node = {node: label for label, node in enumerate(active)}
        labels = np.empty(len(values), dtype=int)
        bounds: list[tuple[float, float, float, float]] = []
        for node in active:
            indices = np.asarray(members[node])
            labels[indices] = label_by_node[node]
            cluster_points = values[indices]
            minimum = cluster_points.min(axis=0)
            maximum = cluster_points.max(axis=0)
            width, height = np.maximum(maximum - minimum, 1e-6)
            bounds.append(
                (
                    float(minimum[0]),
                    float(minimum[1]),
                    float(width),
                    float(height),
                )
            )
        return AgglomerativeSnapshot(
            labels=labels,
            bounds=tuple(bounds),
            n_clusters=len(active),
            merged=merged,
            distance=distance,
        )

    yield state(None, None)
    if len(values) == 1:
        return

    tree = linkage(values, method="ward")
    for merge_index, row in enumerate(tree[: len(values) - n_clusters]):
        left, right = int(row[0]), int(row[1])
        members[len(values) + merge_index] = members.pop(left) + members.pop(right)
        yield state((left, right), float(row[2]))


def dbscan_snapshots(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> Iterator[DBSCANSnapshot]:
    """Simula DBSCAN con vecinos, cola de expansión y marcas de ruido."""

    values = _validate_points(points)
    if eps <= 0 or min_samples < 1:
        raise ValueError("eps debe ser positivo y min_samples debe ser >= 1.")

    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    neighbors_by_point = [np.flatnonzero(row <= eps) for row in distances]
    core_mask = np.array(
        [len(neighbors) >= min_samples for neighbors in neighbors_by_point],
        dtype=bool,
    )
    labels = np.full(len(values), -2, dtype=int)  # -2 = no visitado, -1 = ruido
    visited = np.zeros(len(values), dtype=bool)
    empty = np.array([], dtype=int)
    yield DBSCANSnapshot(
        labels.copy(),
        visited.copy(),
        core_mask.copy(),
        None,
        empty,
        empty,
        None,
        "puntos sin visitar",
    )

    cluster_id = 0
    for seed in range(len(values)):
        if visited[seed]:
            continue

        visited[seed] = True
        seed_neighbors = neighbors_by_point[seed]
        if len(seed_neighbors) < min_samples:
            labels[seed] = -1
            yield DBSCANSnapshot(
                labels.copy(),
                visited.copy(),
                core_mask.copy(),
                seed,
                seed_neighbors.copy(),
                empty,
                None,
                "punto marcado como ruido",
            )
            continue

        labels[seed] = cluster_id
        queue: deque[int] = deque(
            int(index) for index in seed_neighbors if int(index) != seed
        )
        queued = set(queue)
        labels[seed_neighbors] = cluster_id
        yield DBSCANSnapshot(
            labels.copy(),
            visited.copy(),
            core_mask.copy(),
            seed,
            seed_neighbors.copy(),
            np.asarray(queue, dtype=int),
            cluster_id,
            f"inicia expansión del cluster {cluster_id + 1}",
        )

        while queue:
            current = queue.popleft()
            queued.discard(current)
            current_neighbors = neighbors_by_point[current]
            if not visited[current]:
                visited[current] = True
                if core_mask[current]:
                    for neighbor in current_neighbors:
                        neighbor = int(neighbor)
                        if labels[neighbor] in (-2, -1):
                            labels[neighbor] = cluster_id
                        if not visited[neighbor] and neighbor not in queued:
                            queue.append(neighbor)
                            queued.add(neighbor)
            if labels[current] in (-2, -1):
                labels[current] = cluster_id

            yield DBSCANSnapshot(
                labels.copy(),
                visited.copy(),
                core_mask.copy(),
                current,
                current_neighbors.copy(),
                np.asarray(queue, dtype=int),
                cluster_id,
                f"expandiendo cluster {cluster_id + 1}",
            )

        cluster_id += 1
        yield DBSCANSnapshot(
            labels.copy(),
            visited.copy(),
            core_mask.copy(),
            None,
            empty,
            empty,
            cluster_id - 1,
            f"cluster {cluster_id} completado",
        )

    labels[labels == -2] = -1
    yield DBSCANSnapshot(
        labels.copy(),
        visited.copy(),
        core_mask.copy(),
        None,
        empty,
        empty,
        None,
        "DBSCAN finalizado",
    )


def clique_snapshots(
    points: np.ndarray,
    n_bins: int,
    min_unit_points: int,
) -> Iterator[CLIQUEAnimationSnapshot]:
    """Genera estados de CLIQUE para grilla, densidad y celdas adyacentes."""

    values = _validate_points(points)
    if n_bins < 1 or min_unit_points < 1:
        raise ValueError("n_bins y min_unit_points deben ser positivos.")

    x_min, y_min = values.min(axis=0)
    x_max, y_max = values.max(axis=0)
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    cell_indices = np.column_stack(
        [
            np.clip(np.digitize(values[:, 0], x_edges) - 1, 0, n_bins - 1),
            np.clip(np.digitize(values[:, 1], y_edges) - 1, 0, n_bins - 1),
        ]
    ).astype(int)
    counts = np.zeros((n_bins, n_bins), dtype=int)
    np.add.at(counts, (cell_indices[:, 0], cell_indices[:, 1]), 1)
    dense_cells = frozenset(
        (int(index[0]), int(index[1]))
        for index in np.argwhere(counts >= min_unit_points)
    )
    labels = np.full(len(values), -1, dtype=int)

    yield CLIQUEAnimationSnapshot(
        labels.copy(),
        cell_indices.copy(),
        counts.copy(),
        x_edges.copy(),
        y_edges.copy(),
        frozenset(),
        frozenset(),
        "espacio discretizado en una grilla",
    )
    yield CLIQUEAnimationSnapshot(
        labels.copy(),
        cell_indices.copy(),
        counts.copy(),
        x_edges.copy(),
        y_edges.copy(),
        dense_cells,
        frozenset(),
        f"celdas densas: densidad >= {min_unit_points}",
    )

    remaining = set(dense_cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        pending = [start]
        while pending:
            current_x, current_y = pending.pop()
            for neighbor in (
                (current_x - 1, current_y),
                (current_x + 1, current_y),
                (current_x, current_y - 1),
                (current_x, current_y + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(component)

    cell_to_cluster = {
        cell: cluster
        for cluster, component in enumerate(components)
        for cell in component
    }
    active_cells: set[tuple[int, int]] = set()
    for component_index, component in enumerate(components):
        active_cells.update(component)
        labels = np.array(
            [
                cell_to_cluster.get((int(cell[0]), int(cell[1])), -1)
                if (int(cell[0]), int(cell[1])) in active_cells
                else -1
                for cell in cell_indices
            ],
            dtype=int,
        )
        yield CLIQUEAnimationSnapshot(
            labels.copy(),
            cell_indices.copy(),
            counts.copy(),
            x_edges.copy(),
            y_edges.copy(),
            dense_cells,
            frozenset(active_cells),
            f"unión de celdas adyacentes: cluster {component_index + 1}",
        )


def animate_kmeans(
    points: np.ndarray,
    output_path: Path,
    n_clusters: int,
    *,
    max_iter: int = 20,
    random_state: int = 42,
    fps: int = 2,
    dpi: int = 120,
) -> Path:
    """Guarda la animación de asignaciones y movimiento de centroides."""

    values = _validate_points(points)
    snapshots = list(
        kmeans_snapshots(
            values,
            n_clusters,
            max_iter=max_iter,
            random_state=random_state,
        )
    )
    figure, axis = plt.subplots(figsize=(7, 5.5))
    _configure_axis(axis, values, "K-Means")
    point_artist = axis.scatter(
        values[:, 0], values[:, 1], s=28, alpha=0.75, edgecolors="none"
    )
    center_artist = axis.scatter(
        snapshots[0].centers[:, 0],
        snapshots[0].centers[:, 1],
        marker="X",
        s=160,
        c=_cluster_colors(n_clusters),
        edgecolors="black",
        linewidths=0.8,
    )
    status = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    arrow_artist = None

    def update(frame: int) -> tuple:
        nonlocal arrow_artist
        snapshot = snapshots[frame]
        point_artist.set_facecolors(_point_colors(snapshot.labels))
        center_artist.set_offsets(snapshot.centers)
        status.set_text(f"Iteración {snapshot.iteration}: {snapshot.phase}")
        if arrow_artist is not None:
            arrow_artist.remove()
            arrow_artist = None
        if snapshot.old_centers is not None:
            displacement = snapshot.centers - snapshot.old_centers
            arrow_artist = axis.quiver(
                snapshot.old_centers[:, 0],
                snapshot.old_centers[:, 1],
                displacement[:, 0],
                displacement[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="black",
                width=0.004,
            )
        return (point_artist, center_artist, status)

    return _save_animation(
        figure,
        update,
        len(snapshots),
        Path(output_path),
        fps,
        dpi,
    )


def animate_agglomerative(
    points: np.ndarray,
    output_path: Path,
    n_clusters: int,
    *,
    fps: int = 2,
    dpi: int = 120,
) -> Path:
    """Guarda la animación de fusiones Ward y sus límites de grupo."""

    values = _validate_points(points)
    snapshots = list(agglomerative_snapshots(values, n_clusters))
    figure, axis = plt.subplots(figsize=(7, 5.5))
    _configure_axis(axis, values, "Agglomerative Ward")
    point_artist = axis.scatter(
        values[:, 0], values[:, 1], s=28, alpha=0.75, edgecolors="none"
    )
    status = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    rectangles: list[Rectangle] = []

    def update(frame: int) -> tuple:
        snapshot = snapshots[frame]
        point_artist.set_facecolors(_point_colors(snapshot.labels))
        for rectangle in rectangles:
            rectangle.remove()
        rectangles.clear()
        colors = _cluster_colors(snapshot.n_clusters)
        for index, (x, y, width, height) in enumerate(snapshot.bounds):
            rectangle = Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                linewidth=1.5,
                edgecolor=colors[index],
                linestyle="--",
            )
            axis.add_patch(rectangle)
            rectangles.append(rectangle)
        if snapshot.merged is None:
            status.set_text(f"Inicio: {snapshot.n_clusters} clusters")
        else:
            left, right = snapshot.merged
            status.set_text(
                f"Fusión {left} + {right} | Ward={snapshot.distance:.3f} | "
                f"quedan {snapshot.n_clusters} clusters"
            )
        return (point_artist, status, *rectangles)

    return _save_animation(
        figure,
        update,
        len(snapshots),
        Path(output_path),
        fps,
        dpi,
    )


def animate_dbscan(
    points: np.ndarray,
    output_path: Path,
    eps: float,
    min_samples: int,
    *,
    fps: int = 2,
    dpi: int = 120,
) -> Path:
    """Guarda la animación de expansión, vecindades y ruido de DBSCAN."""

    values = _validate_points(points)
    snapshots = list(dbscan_snapshots(values, eps, min_samples))
    figure, axis = plt.subplots(figsize=(7, 5.5))
    _configure_axis(axis, values, "DBSCAN")
    point_artist = axis.scatter(
        values[:, 0], values[:, 1], s=28, alpha=0.75, edgecolors="none"
    )
    core_artist = axis.scatter(
        values[snapshots[0].core_mask, 0],
        values[snapshots[0].core_mask, 1],
        s=110,
        facecolors="none",
        edgecolors="dimgray",
        linewidths=1,
        marker="o",
        label="Punto núcleo",
    )
    neighbor_artist = axis.scatter(
        np.empty((0, 2)),
        np.empty((0, 2)),
        s=75,
        facecolors="none",
        edgecolors="crimson",
        linewidths=1.5,
        marker="o",
        label="Vecindad ε",
    )
    current_artist = axis.scatter(
        np.empty((0, 2)),
        np.empty((0, 2)),
        s=180,
        color="gold",
        edgecolors="black",
        marker="*",
        zorder=4,
        label="Punto actual",
    )
    axis.legend(loc="lower left", fontsize="small")
    status = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    circle: Circle | None = None

    def update(frame: int) -> tuple:
        nonlocal circle
        snapshot = snapshots[frame]
        point_artist.set_facecolors(_point_colors(snapshot.labels))
        if snapshot.current_point is None:
            neighbor_artist.set_offsets(np.empty((0, 2)))
            current_artist.set_offsets(np.empty((0, 2)))
        else:
            current = snapshot.current_point
            neighbor_artist.set_offsets(values[snapshot.neighbors])
            current_artist.set_offsets(values[[current]])
        if circle is not None:
            circle.remove()
            circle = None
        if snapshot.current_point is not None:
            current = snapshot.current_point
            circle = Circle(
                values[current],
                eps,
                fill=False,
                color="crimson",
                linewidth=1.2,
                alpha=0.8,
            )
            axis.add_patch(circle)
        visited_count = int(snapshot.visited.sum())
        status.set_text(
            f"{snapshot.phase}\nvisitados: {visited_count}/{len(values)} | "
            f"ε={eps:g}, min_samples={min_samples}"
        )
        return (point_artist, core_artist, neighbor_artist, current_artist, status)

    return _save_animation(
        figure,
        update,
        len(snapshots),
        Path(output_path),
        fps,
        dpi,
    )


def animate_clique(
    points: np.ndarray,
    output_path: Path,
    n_bins: int,
    min_unit_points: int,
    *,
    fps: int = 2,
    dpi: int = 120,
) -> Path:
    """Guarda la animación de grilla, densidad y unión de CLIQUE."""

    values = _validate_points(points)
    snapshots = list(clique_snapshots(values, n_bins, min_unit_points))
    figure, axis = plt.subplots(figsize=(7, 5.5))
    _configure_axis(axis, values, "CLIQUE")
    point_artist = axis.scatter(
        values[:, 0], values[:, 1], s=28, alpha=0.78, edgecolors="none"
    )
    status = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    rectangles: list[Rectangle] = []
    annotations = []

    def update(frame: int) -> tuple:
        snapshot = snapshots[frame]
        point_artist.set_facecolors(_point_colors(snapshot.labels))
        for rectangle in rectangles:
            rectangle.remove()
        rectangles.clear()
        for annotation in annotations:
            annotation.remove()
        annotations.clear()

        cluster_colors = _cluster_colors(max(int(snapshot.labels.max()) + 1, 1))
        for cell_x in range(len(snapshot.x_edges) - 1):
            for cell_y in range(len(snapshot.y_edges) - 1):
                cell = (cell_x, cell_y)
                if cell in snapshot.active_cells:
                    in_cell = (
                        (snapshot.cell_indices[:, 0] == cell_x)
                        & (snapshot.cell_indices[:, 1] == cell_y)
                    )
                    cell_labels = snapshot.labels[in_cell]
                    assigned_labels = cell_labels[cell_labels >= 0]
                    cluster = int(assigned_labels[0]) if len(assigned_labels) else 0
                    face_color = cluster_colors[cluster % len(cluster_colors)]
                    alpha = 0.22
                elif cell in snapshot.dense_cells:
                    face_color = "gold"
                    alpha = 0.25
                else:
                    face_color = "none"
                    alpha = 0.0
                rectangle = Rectangle(
                    (snapshot.x_edges[cell_x], snapshot.y_edges[cell_y]),
                    snapshot.x_edges[cell_x + 1] - snapshot.x_edges[cell_x],
                    snapshot.y_edges[cell_y + 1] - snapshot.y_edges[cell_y],
                    facecolor=face_color,
                    alpha=alpha,
                    edgecolor="gray",
                    linewidth=0.5,
                )
                axis.add_patch(rectangle)
                rectangles.append(rectangle)

                count = snapshot.counts[cell_x, cell_y]
                if count:
                    annotation = axis.text(
                        (snapshot.x_edges[cell_x] + snapshot.x_edges[cell_x + 1]) / 2,
                        (snapshot.y_edges[cell_y] + snapshot.y_edges[cell_y + 1]) / 2,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="dimgray",
                    )
                    annotations.append(annotation)

        axis.set_xticks(snapshot.x_edges)
        axis.set_yticks(snapshot.y_edges)
        axis.tick_params(labelsize=7)
        status.set_text(
            f"{snapshot.phase}\n"
            f"celdas densas: {len(snapshot.dense_cells)} | "
            f"celdas unidas: {len(snapshot.active_cells)}"
        )
        return (point_artist, status, *rectangles, *annotations)

    return _save_animation(
        figure,
        update,
        len(snapshots),
        Path(output_path),
        fps,
        dpi,
    )


def save_clustering_animations(
    points: np.ndarray,
    output_dir: Path,
    *,
    file_prefix: str = "movies",
    n_clusters: int,
    dbscan_eps: float = 0.8,
    dbscan_min_samples: int = 10,
    clique_bins: int = 8,
    clique_min_unit_points: int = 5,
    max_points: int | None = 120,
    random_state: int = 42,
    fps: int = 2,
    dpi: int = 120,
) -> dict[str, Path]:
    """Genera y guarda las cuatro animaciones en ``output_dir``.

    ``file_prefix`` permite reutilizar la función con varios datasets sin
    sobrescribir los GIF existentes (por ejemplo, ``movies`` e ``iris``).
    ``points`` debe ser una representación 2D. Si contiene más observaciones
    que ``max_points``, se usa una muestra aleatoria reproducible para evitar
    GIF demasiado pesados en presentaciones.
    """

    if not file_prefix or Path(file_prefix).name != file_prefix:
        raise ValueError("file_prefix debe ser un nombre simple, sin rutas.")

    values = _validate_points(points)
    sampled, _ = _sample_points(values, max_points, random_state)
    destination = Path(output_dir)
    return {
        "K-Means": animate_kmeans(
            sampled,
            destination / f"{file_prefix}-kmeans.gif",
            n_clusters,
            random_state=random_state,
            fps=fps,
            dpi=dpi,
        ),
        "DBSCAN": animate_dbscan(
            sampled,
            destination / f"{file_prefix}-dbscan.gif",
            dbscan_eps,
            dbscan_min_samples,
            fps=fps,
            dpi=dpi,
        ),
        "Agglomerative Ward": animate_agglomerative(
            sampled,
            destination / f"{file_prefix}-agglomerative-ward.gif",
            n_clusters,
            fps=fps,
            dpi=dpi,
        ),
        "CLIQUE": animate_clique(
            sampled,
            destination / f"{file_prefix}-clique.gif",
            clique_bins,
            clique_min_unit_points,
            fps=fps,
            dpi=dpi,
        ),
    }
