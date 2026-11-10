"""Run the U4 classification experiments and update their Markdown report."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
PREPARED_DIR = ROOT / "Datos preparados"
SPLITS_DIR = ROOT / "Particiones"
RESULTS_DIR = ROOT / "Resultados"
REPORT_PATH = ROOT / "casificacion.md"
SEED = 42

METRIC_NAMES = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "specificity",
    "mcc",
    "kappa",
    "roc_auc",
    "log_loss",
    "brier_score",
]


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    file_name: str
    split_folder: str
    labels: tuple[int, ...]
    class_names: tuple[str, ...]
    selected_k: int


DATASETS = [
    Dataset(
        "iris",
        "Iris binario",
        "iris_binario.csv",
        "iris",
        (0, 1),
        ("no versicolor", "versicolor"),
        2,
    ),
    Dataset(
        "mobile_price",
        "Mobile Price Classification",
        "mobile_price.csv",
        "mobile_price",
        (0, 1, 2, 3),
        ("0", "1", "2", "3"),
        10,
    ),
]

MODEL_NAMES = [
    "Regresión logística",
    "Red neuronal (MLP)",
    "KNN",
    "SVM lineal",
    "Árbol de decisión",
]

MODEL_PARAMETERS = {
    "Regresión logística": "StandardScaler; LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)",
    "Red neuronal (MLP)": "StandardScaler; MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=50, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1, random_state=42)",
    "KNN": "StandardScaler; KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)",
    "SVM lineal": "StandardScaler; SVC(C=1.0, kernel='linear', probability=True, random_state=42)",
    "Árbol de decisión": "DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)",
}


def make_model(name: str) -> tuple[Any, bool]:
    """Return an estimator and whether it needs standard scaling."""
    if name == "Regresión logística":
        return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED), True
    if name == "Red neuronal (MLP)":
        return (
            MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=50,
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.1,
                random_state=SEED,
            ),
            True,
        )
    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2,
        ), True
    if name == "SVM lineal":
        return SVC(C=1.0, kernel="linear", probability=True, random_state=SEED), True
    if name == "Árbol de decisión":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            random_state=SEED,
        ), False
    raise ValueError(f"Clasificador desconocido: {name}")


def make_pipeline(model_name: str, selected_k: int | None) -> Pipeline:
    estimator, use_scaler = make_model(model_name)
    steps: list[tuple[str, Any]] = []
    if selected_k is not None:
        steps.append(("anova", SelectKBest(score_func=f_classif, k=selected_k)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def load_splits(dataset: Dataset, row_count: int) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Read the saved manifests and return (protocol, split id, train, test)."""
    base = SPLITS_DIR / dataset.split_folder
    specifications = [
        ("Hold-Out 60/40", "holdout_60_40.csv", "partition"),
        ("Entrenamiento/Prueba", "train_test_75_75.csv" if dataset.key == "iris" else "train_test_100_100.csv", "partition"),
        ("Random Subsampling (N=30)", "random_subsampling_30.csv", "repetition"),
        ("K-Fold (K=10)", "kfold_10.csv", "test_fold"),
        ("Leave-One-Out (LOO)", "leave_one_out.csv", "test_fold"),
        ("Stratified Cross-Validation (K=10)", "stratified_cross_validation_10.csv", "test_fold"),
    ]
    splits: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    universe = np.arange(row_count, dtype=int)

    for protocol, filename, mode in specifications:
        manifest = pd.read_csv(base / filename)
        if mode == "partition":
            train_ids = manifest.loc[manifest["partition"] == "train", "row_id"].to_numpy(dtype=int)
            test_ids = manifest.loc[manifest["partition"] == "test", "row_id"].to_numpy(dtype=int)
            splits.append((protocol, "1", train_ids, test_ids))
        elif mode == "repetition":
            for repetition, group in manifest.groupby("repetition", sort=True):
                train_ids = group.loc[group["partition"] == "train", "row_id"].to_numpy(dtype=int)
                test_ids = group.loc[group["partition"] == "test", "row_id"].to_numpy(dtype=int)
                splits.append((protocol, str(int(repetition)), train_ids, test_ids))
        else:
            for fold, group in manifest.groupby("test_fold", sort=True):
                test_ids = group["row_id"].to_numpy(dtype=int)
                train_ids = np.setdiff1d(universe, test_ids, assume_unique=True)
                splits.append((protocol, str(int(fold) + 1 if protocol == "Leave-One-Out (LOO)" else int(fold)), train_ids, test_ids))

    return splits


def class_probability_matrix(estimator: Pipeline, x_test: pd.DataFrame, labels: tuple[int, ...]) -> np.ndarray:
    raw = estimator.predict_proba(x_test)
    model = estimator.named_steps["model"]
    output = np.zeros((len(x_test), len(labels)), dtype=float)
    positions = {label: index for index, label in enumerate(labels)}
    for source_index, label in enumerate(model.classes_):
        output[:, positions[int(label)]] = raw[:, source_index]
    output = np.clip(output, 1e-15, 1.0)
    output /= output.sum(axis=1, keepdims=True)
    return output


def metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[int, ...],
) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    total = int(cm.sum())
    for index in range(len(labels)):
        tp = int(cm[index, index])
        fn = int(cm[index, :].sum() - tp)
        fp = int(cm[:, index].sum() - tp)
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp) if tn + fp else 0.0)

    has_all_labels = len(np.unique(y_true)) == len(labels)
    if has_all_labels and len(labels) == 2:
        roc_auc = roc_auc_score(y_true, probabilities[:, 1])
        brier = float(np.mean((probabilities[:, 1] - y_true) ** 2))
    elif has_all_labels:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=list(labels),
            multi_class="ovr",
            average="macro",
        )
        one_hot = np.eye(len(labels), dtype=float)[np.searchsorted(labels, y_true)]
        brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    elif len(labels) == 2:
        roc_auc = float("nan")
        brier = float(np.mean((probabilities[:, 1] - y_true) ** 2))
    else:
        roc_auc = float("nan")
        one_hot = np.eye(len(labels), dtype=float)[np.searchsorted(labels, y_true)]
        brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "specificity": float(specificities[1] if len(labels) == 2 else np.mean(specificities)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=list(labels))),
        "roc_auc": float(roc_auc),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(labels))),
        "brier_score": brier,
    }


def run_configuration(
    dataset: Dataset,
    features: pd.DataFrame,
    targets: np.ndarray,
    splits: list[tuple[str, str, np.ndarray, np.ndarray]],
    model_name: str,
    feature_variant: str,
    progress_interval: int = 250,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    selected_k = dataset.selected_k if feature_variant == "ANOVA F-test" else None
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    probability_rows: list[np.ndarray] = []
    split_metrics: list[dict[str, Any]] = []
    selected_features: list[dict[str, Any]] = []
    start = time.perf_counter()
    last_protocol = ""
    protocol_counts: dict[str, int] = {}
    config = f"{dataset.key} | {model_name} | {feature_variant}"

    for index, (protocol, split_id, train_ids, test_ids) in enumerate(splits, start=1):
        protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        estimator = make_pipeline(model_name, selected_k)
        x_train = features.iloc[train_ids]
        x_test = features.iloc[test_ids]
        y_train = targets[train_ids]
        y_test = targets[test_ids]
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test).astype(int)
        probabilities = class_probability_matrix(estimator, x_test, dataset.labels)
        if protocol == "Leave-One-Out (LOO)":
            # Metrics such as AUC are undefined for one observation. LOO is
            # evaluated after pooling its complete set of out-of-fold values.
            scores = {metric: float("nan") for metric in METRIC_NAMES}
        else:
            scores = metric_values(y_test, y_pred, probabilities, dataset.labels)

        split_metrics.append(
            {
                "dataset": dataset.key,
                "validacion": protocol,
                "particion": split_id,
                "clasificador": model_name,
                "seleccion_caracteristicas": feature_variant,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                **scores,
            }
        )
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        probability_rows.append(probabilities)

        if (
            feature_variant == "ANOVA F-test"
            and model_name == MODEL_NAMES[0]
            and "anova" in estimator.named_steps
        ):
            mask = estimator.named_steps["anova"].get_support()
            selected_features.append(
                {
                    "dataset": dataset.key,
                    "validacion": protocol,
                    "particion": split_id,
                    "k": selected_k,
                    "variables_seleccionadas": ", ".join(features.columns[mask].tolist()),
                }
            )

        if protocol != last_protocol:
            if last_protocol:
                print(f"    {last_protocol}: terminado", flush=True)
            last_protocol = protocol
        if protocol == "Leave-One-Out (LOO)" and protocol_counts[protocol] % progress_interval == 0:
            elapsed = time.perf_counter() - start
            print(
                f"    {config}: LOO {protocol_counts[protocol]}/"
                f"{sum(1 for part in splits if part[0] == protocol)} particiones ({elapsed:.0f} s)",
                flush=True,
            )

    if last_protocol:
        print(f"    {last_protocol}: terminado", flush=True)

    all_y_true = np.asarray(y_true_all, dtype=int)
    all_y_pred = np.asarray(y_pred_all, dtype=int)
    all_probabilities = np.vstack(probability_rows)
    confusion = confusion_matrix(all_y_true, all_y_pred, labels=dataset.labels)

    protocol_records: list[dict[str, Any]] = []
    for protocol, group in pd.DataFrame(split_metrics).groupby("validacion", sort=False):
        first = group.iloc[0]
        pooled_mask = np.concatenate(
            [
                np.repeat(record["validacion"] == protocol, record["n_test"])
                for record in split_metrics
            ]
        )
        pooled_metrics = metric_values(
            all_y_true[pooled_mask],
            all_y_pred[pooled_mask],
            all_probabilities[pooled_mask],
            dataset.labels,
        )
        is_leave_one_out = protocol == "Leave-One-Out (LOO)"
        record: dict[str, Any] = {
            "dataset": dataset.key,
            "dataset_nombre": dataset.name,
            "validacion": protocol,
            "clasificador": model_name,
            "parametros": MODEL_PARAMETERS[model_name],
            "seleccion_caracteristicas": feature_variant,
            "n_splits": len(group),
            "n_predicciones_oof": int(group["n_test"].sum()),
            "matriz_confusion": json.dumps(
                confusion_matrix(
                    all_y_true[pooled_mask],
                    all_y_pred[pooled_mask],
                    labels=dataset.labels,
                ).tolist()
            ),
        }
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy(dtype=float)
            # The instructions specify pooled metrics for LOO and mean/std by
            # split for repeated subsampling and k-fold cross-validation.
            if is_leave_one_out:
                mean_value = pooled_metrics[metric]
                std_value = np.nan
            else:
                mean_value = float(np.mean(values))
                std_value = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            record[f"{metric}_mean"] = mean_value
            record[f"{metric}_std"] = std_value
        record["tamano_train_representativo"] = int(first["n_train"])
        record["tamano_test_representativo"] = int(first["n_test"])
        protocol_records.append(record)

    return protocol_records, split_metrics, confusion, selected_features


def write_outputs(
    summaries: list[dict[str, Any]],
    split_scores: list[dict[str, Any]],
    matrices: list[dict[str, Any]],
    selected_features: list[dict[str, Any]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    split_frame = pd.DataFrame(split_scores)
    matrix_frame = pd.DataFrame(matrices)
    selected_frame = pd.DataFrame(selected_features)
    summary_frame.to_csv(RESULTS_DIR / "resultados.csv", index=False, encoding="utf-8")
    split_frame.to_csv(RESULTS_DIR / "resultados_por_particion.csv", index=False, encoding="utf-8")
    matrix_frame.to_csv(RESULTS_DIR / "matrices_confusion.csv", index=False, encoding="utf-8")
    selected_frame.to_csv(RESULTS_DIR / "variables_anova_por_particion.csv", index=False, encoding="utf-8")


def load_existing_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resume complete configurations saved by an earlier run."""
    paths = [
        RESULTS_DIR / "resultados.csv",
        RESULTS_DIR / "resultados_por_particion.csv",
        RESULTS_DIR / "matrices_confusion.csv",
        RESULTS_DIR / "variables_anova_por_particion.csv",
    ]
    if not all(path.exists() for path in paths):
        return [], [], [], []

    summary_frame, split_frame, matrix_frame, selected_frame = [
        pd.read_csv(path) for path in paths
    ]
    # The earlier slow RBF SVM results are deliberately discarded so every
    # dataset is compared with the same, faster linear SVM configuration.
    old_name = "SVM RBF"
    summary_frame = summary_frame.loc[summary_frame["clasificador"] != old_name]
    split_frame = split_frame.loc[split_frame["clasificador"] != old_name]
    matrix_frame = matrix_frame.loc[matrix_frame["clasificador"] != old_name]
    return (
        summary_frame.to_dict(orient="records"),
        split_frame.to_dict(orient="records"),
        matrix_frame.to_dict(orient="records"),
        selected_frame.to_dict(orient="records"),
    )


def format_metric(row: pd.Series, metric: str) -> str:
    mean_value = float(row[f"{metric}_mean"])
    std_value = row[f"{metric}_std"]
    if pd.isna(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {float(std_value):.4f}"


def confusion_short(value: str) -> str:
    matrix = json.loads(value)
    return "`" + json.dumps(matrix, separators=(",", ":")) + "`"


def markdown_experiment_table(results: pd.DataFrame) -> str:
    columns = [
        ("dataset_nombre", "Dataset"),
        ("validacion", "Validación"),
        ("clasificador", "Clasificador"),
        ("seleccion_caracteristicas", "Selección"),
    ]
    metric_headers = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precisión macro"),
        ("recall_macro", "Recall macro"),
        ("f1_macro", "F1 macro"),
        ("specificity", "Especificidad"),
        ("mcc", "MCC"),
        ("kappa", "Kappa"),
        ("roc_auc", "ROC-AUC"),
        ("log_loss", "Log loss"),
        ("brier_score", "Brier"),
    ]
    headers = [label for _, label in columns] + [label for _, label in metric_headers] + ["Matriz de confusión"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in results.iterrows():
        cells = [str(row[column]) for column, _ in columns]
        cells.extend(format_metric(row, metric) for metric, _ in metric_headers)
        cells.append(confusion_short(row["matriz_confusion"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_result(results: pd.DataFrame, dataset_key: str, cross_validation_only: bool = False) -> pd.Series:
    subset = results.loc[results["dataset"] == dataset_key].copy()
    if cross_validation_only:
        cv_names = [
            "Random Subsampling (N=30)",
            "K-Fold (K=10)",
            "Leave-One-Out (LOO)",
            "Stratified Cross-Validation (K=10)",
        ]
        subset = subset.loc[subset["validacion"].isin(cv_names)]
    return subset.sort_values(
        ["f1_macro_mean", "roc_auc_mean", "accuracy_mean"], ascending=False
    ).iloc[0]


def render_findings(results: pd.DataFrame) -> str:
    chunks = [
        "## Hallazgos y conclusiones",
        "",
        "Se usa **F1 macro medio** como criterio principal para comparar configuraciones porque da el mismo peso a cada clase. En los protocolos de una sola partición se presenta una medición; en Random Subsampling y K-Fold se presenta media ± desviación estándar entre repeticiones/folds. LOO se resume con las predicciones fuera de muestra de todas las observaciones. Las matrices incluidas en la tabla agregan las predicciones de prueba del protocolo; Random Subsampling cuenta cada observación una vez por repetición.",
        "",
        "Las filas de Entrenamiento/Prueba usan muestras de prueba pequeñas (75 en Iris y 100 en Mobile), por lo que se leen como una comparación acordada y no como estimación más estable que la validación cruzada. Para decidir qué configuración generaliza mejor, también se informa el mejor resultado entre los protocolos con repeticiones o folds.",
        "",
    ]

    for dataset in DATASETS:
        all_best = best_result(results, dataset.key)
        cv_best = best_result(results, dataset.key, cross_validation_only=True)
        all_cm = np.asarray(json.loads(all_best["matriz_confusion"]), dtype=int)
        cv_cm = np.asarray(json.loads(cv_best["matriz_confusion"]), dtype=int)
        chunks.extend(
            [
                f"### {dataset.name}",
                "",
                f"La mayor media de F1 macro entre todos los protocolos corresponde a **{all_best['clasificador']}**, con **{all_best['seleccion_caracteristicas']}**, bajo **{all_best['validacion']}**: F1 macro {all_best['f1_macro_mean']:.4f}, accuracy {all_best['accuracy_mean']:.4f}, recall macro {all_best['recall_macro_mean']:.4f} y ROC-AUC {all_best['roc_auc_mean']:.4f}. Su matriz agregada es `{json.dumps(all_cm.tolist(), separators=(',', ':'))}`.",
                "",
                f"Entre Random Subsampling, K-Fold, LOO y Stratified Cross-Validation, el mayor resultado es de **{cv_best['clasificador']}**, con **{cv_best['seleccion_caracteristicas']}**, bajo **{cv_best['validacion']}**: F1 macro {format_metric(cv_best, 'f1_macro')}, accuracy {format_metric(cv_best, 'accuracy')}, recall macro {cv_best['recall_macro_mean']:.4f} y ROC-AUC {cv_best['roc_auc_mean']:.4f}. La matriz de confusión agregada de sus predicciones fuera de muestra es `{json.dumps(cv_cm.tolist(), separators=(',', ':'))}`.",
                "",
            ]
        )
        if dataset.key == "iris":
            tn, fp, fn, tp = cv_cm.ravel()
            positive_recall = tp / (tp + fn) if tp + fn else 0.0
            chunks.append(
                f"Para esta configuración de validación cruzada, la sensibilidad de la clase positiva `versicolor` es {positive_recall:.4f} ({tp} de {tp + fn} casos). La diagonal de la matriz muestra las clasificaciones correctas; los falsos negativos son los `versicolor` clasificados como `no versicolor`."
            )
        else:
            chunks.append(
                "Las cuatro clases tienen el mismo tamaño en el conjunto completo. La especificidad y el Brier score se calcularon macro uno-contra-el-resto y con suma de errores cuadráticos por observación, respectivamente."
            )

        paired = results.loc[results["dataset"] == dataset.key].pivot_table(
            index=["validacion", "clasificador"],
            columns="seleccion_caracteristicas",
            values="f1_macro_mean",
            aggfunc="first",
        ).dropna()
        if {"Sin selección", "ANOVA F-test"}.issubset(paired.columns):
            delta = paired["ANOVA F-test"] - paired["Sin selección"]
            improved = int((delta > 0).sum())
            chunks.extend(
                [
                    "",
                    f"ANOVA aumentó la media de F1 macro en {improved} de {len(delta)} comparaciones pareadas (misma partición y clasificador). La diferencia media fue {delta.mean():+.4f}; el efecto depende del método y el modelo, por lo que la selección se conserva solo cuando los resultados la respaldan.",
                ]
            )
        chunks.extend(["", ""])

    chunks.extend(
        [
            "Los [resultados agregados](Resultados/resultados.csv), las [métricas por partición](Resultados/resultados_por_particion.csv), las [matrices de confusión](Resultados/matrices_confusion.csv) y las [variables elegidas por ANOVA](Resultados/variables_anova_por_particion.csv) quedan en `Resultados/`. El script [`ejecutar_experimentos.py`](ejecutar_experimentos.py) reproduce el análisis con semilla 42 y las particiones guardadas.",
            "",
        ]
    )
    return "\n".join(chunks)


def update_report(results: pd.DataFrame) -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    procedure_heading = "## Procedimiento experimental"
    table_heading = "## Tabla comparativa de experimentos"
    findings_heading = "## Hallazgos y conclusiones"
    procedure_start = report.index(procedure_heading)
    start = report.index(table_heading)
    procedure = f"""{procedure_heading}

### Datos y particiones

- **Iris:** 150 observaciones, cuatro predictores numéricos y etiqueta binaria; `versicolor=1` y `no versicolor=0`. La partición Entrenamiento/Prueba usa filas alternadas, 75/75.
- **Mobile Price Classification:** 2,000 filas etiquetadas de `train.csv`, 20 predictores y `price_range` (0–3). Se excluyó `row_id` y no se usó el `test.csv` oficial. Entrenamiento/Prueba toma 100 filas para train y otras 100 para test.
- Se reutilizaron las particiones guardadas para Hold-Out 60/40, 30 repeticiones de Random Subsampling, K-Fold 10 barajado, LOO y Stratified Cross-Validation 10. Hold-Out y Random Subsampling son estratificados; también lo es Entrenamiento/Prueba de Mobile. Las particiones usan semilla 42, salvo Entrenamiento/Prueba de Iris (43) y Mobile (44).

### Preparación de variables y clasificadores

Las variantes se ejecutaron sobre cada misma partición: todas las variables, o `SelectKBest(f_classif)` dentro del pipeline de entrenamiento (`k=2` para Iris y `k=10` para Mobile). El selector y el escalador se ajustaron exclusivamente con train en cada partición. Regresión logística, MLP, KNN y SVM incluyen `StandardScaler`; el árbol no se escala.

| Clasificador | Configuración registrada |
|---|---|
{chr(10).join(f"| {name} | `{parameters}` |" for name, parameters in MODEL_PARAMETERS.items())}

SVM entrega probabilidades con `probability=True`. La red MLP usa una capa oculta de 16 neuronas y máximo 50 épocas con parada temprana para que LOO sobre Mobile sea ejecutable; ese límite forma parte de la configuración comparada.

### Cálculo de métricas

Las métricas macro dan el mismo peso a cada clase. Especificidad usa la clase positiva `versicolor` en Iris y promedio uno-contra-el-resto en Mobile. ROC-AUC usa probabilidades (binario para Iris y macro OvR para Mobile); log loss usa las etiquetas posibles completas. Brier es el error cuadrático binario en Iris y la suma multiclase por observación en Mobile. Para Random Subsampling y validaciones cruzadas se presenta media y desviación estándar entre particiones; sus matrices reúnen predicciones de prueba. LOO calcula las métricas usando juntas sus predicciones fuera de muestra. Las métricas de un fold LOO individual quedan vacías en `resultados_por_particion.csv`, porque un único caso no permite calcularlas de forma válida.

"""
    report = report[:procedure_start] + procedure + report[start:]
    start = report.index(table_heading)
    findings_start = report.find(findings_heading, start)
    if findings_start < 0:
        findings_start = report.find("## Resultados y conclusiones", start)
    if findings_start < 0:
        findings_start = len(report)

    table = (
        f"{table_heading}\n\n"
        "Cada fila corresponde a una combinación de dataset, protocolo, clasificador y variante de características. En protocolos con varias particiones, las métricas son media ± desviación estándar entre particiones; LOO usa las predicciones fuera de muestra reunidas. La matriz presenta conteos con clases reales en filas y predichas en columnas. En Mobile, Brier es la suma de los errores cuadráticos por observación.\n\n"
        + markdown_experiment_table(results)
        + "\n\n"
    )
    updated = report[:start] + table + render_findings(results)
    REPORT_PATH.write_text(updated, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # scikit-learn 1.9 currently warns once for every SVC fit because
    # probability=True is scheduled for removal; the assignment explicitly
    # requires SVC probability estimates, so keep the required setting.
    warnings.filterwarnings("ignore", category=FutureWarning, message="The `probability` parameter was deprecated.*")
    summaries, split_scores, matrices, selected_features = load_existing_outputs()
    completed = {
        (record["dataset"], record["clasificador"], record["seleccion_caracteristicas"], record["validacion"])
        for record in summaries
    }

    for dataset in DATASETS:
        prepared = pd.read_csv(PREPARED_DIR / dataset.file_name)
        targets = prepared["target"].to_numpy(dtype=int)
        features = prepared.drop(columns=["row_id", "target"])
        if len(features.columns) != (4 if dataset.key == "iris" else 20):
            raise ValueError(f"Número inesperado de variables en {dataset.name}.")
        if tuple(sorted(np.unique(targets).tolist())) != dataset.labels:
            raise ValueError(f"Etiquetas inesperadas en {dataset.name}.")

        splits = load_splits(dataset, len(prepared))
        print(f"\n{dataset.name}: {len(prepared)} observaciones, {len(features.columns)} variables", flush=True)
        for model_name in MODEL_NAMES:
            for variant in ("Sin selección", "ANOVA F-test"):
                expected_protocols = {
                    "Hold-Out 60/40",
                    "Entrenamiento/Prueba",
                    "Random Subsampling (N=30)",
                    "K-Fold (K=10)",
                    "Leave-One-Out (LOO)",
                    "Stratified Cross-Validation (K=10)",
                }
                actual_protocols = {
                    protocol
                    for dataset_key, classifier, feature_variant, protocol in completed
                    if dataset_key == dataset.key
                    and classifier == model_name
                    and feature_variant == variant
                }
                if expected_protocols.issubset(actual_protocols):
                    print(f"  {model_name} | {variant}: ya estaba completo; se reutiliza", flush=True)
                    continue

                print(f"  {model_name} | {variant}", flush=True)
                records, partition_records, _, selections = run_configuration(
                    dataset,
                    features,
                    targets,
                    splits,
                    model_name,
                    variant,
                )
                summaries.extend(records)
                split_scores.extend(partition_records)
                selected_features.extend(selections)
                completed.update(
                    (record["dataset"], record["clasificador"], record["seleccion_caracteristicas"], record["validacion"])
                    for record in records
                )

                for record in records:
                    matrix = np.asarray(json.loads(record["matriz_confusion"]), dtype=int)
                    for actual_index, actual_name in enumerate(dataset.class_names):
                        for predicted_index, predicted_name in enumerate(dataset.class_names):
                            matrices.append(
                                {
                                    "dataset": dataset.key,
                                    "validacion": record["validacion"],
                                    "clasificador": model_name,
                                    "seleccion_caracteristicas": variant,
                                    "clase_real": actual_name,
                                    "clase_predicha": predicted_name,
                                    "conteo": int(matrix[actual_index, predicted_index]),
                                }
                            )

                write_outputs(summaries, split_scores, matrices, selected_features)
                print(f"  {model_name} | {variant}: completado", flush=True)

    results = pd.DataFrame(summaries)
    results = results.sort_values(
        ["dataset", "validacion", "clasificador", "seleccion_caracteristicas"],
        kind="stable",
    ).reset_index(drop=True)
    write_outputs(results.to_dict(orient="records"), split_scores, matrices, selected_features)
    update_report(results)
    print(f"\nResultados guardados en {RESULTS_DIR}", flush=True)
    print(f"Informe actualizado: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
