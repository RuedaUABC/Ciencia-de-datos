"""Prepare the U4 classification datasets and save reusable split manifests."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datasets"
PREPARED_DIR = ROOT / "Datos preparados"
SPLITS_DIR = ROOT / "Particiones"
SEED = 42
N_SPLITS = 10
N_RANDOM_SUBSAMPLES = 30


def validate_dataset(frame: pd.DataFrame, name: str, target: str) -> None:
    if target not in frame.columns:
        raise ValueError(f"{name}: falta la columna objetivo {target!r}.")
    if frame.isna().any().any():
        missing = frame.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"{name}: hay valores ausentes: {missing}")


def write_manifest(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values([c for c in ("repetition", "row_id") if c in frame.columns] or ["row_id"]).to_csv(
        path, index=False, encoding="utf-8"
    )


def make_partition_manifest(
    row_ids: np.ndarray,
    targets: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> pd.DataFrame:
    assignment = {int(row_id): "unused" for row_id in row_ids}
    assignment.update({int(row_id): "train" for row_id in train_ids})
    assignment.update({int(row_id): "test" for row_id in test_ids})
    return pd.DataFrame(
        {
            "row_id": row_ids,
            "partition": [assignment[int(row_id)] for row_id in row_ids],
            "target": targets,
        }
    )


def create_manifests(
    name: str,
    targets: np.ndarray,
    train_test_size: int,
    train_test_seed: int,
) -> list[dict[str, object]]:
    row_ids = np.arange(len(targets), dtype=int)
    out_dir = SPLITS_DIR / name
    summaries: list[dict[str, object]] = []

    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=0.4,
        random_state=SEED,
        stratify=targets,
    )
    write_manifest(
        out_dir / "holdout_60_40.csv",
        make_partition_manifest(row_ids, targets, train_ids, test_ids),
    )
    summaries.append(
        {"dataset": name, "protocol": "Hold-Out 60/40", "train": len(train_ids), "test": len(test_ids), "folds_or_repetitions": 1, "seed": SEED}
    )

    train_ids, test_ids = train_test_split(
        row_ids,
        train_size=train_test_size,
        test_size=train_test_size,
        random_state=train_test_seed,
        stratify=targets,
    )
    write_manifest(
        out_dir / f"train_test_{train_test_size}_{train_test_size}.csv",
        make_partition_manifest(row_ids, targets, train_ids, test_ids),
    )
    summaries.append(
        {"dataset": name, "protocol": f"Entrenamiento/Prueba {train_test_size}/{train_test_size}", "train": len(train_ids), "test": len(test_ids), "folds_or_repetitions": 1, "seed": train_test_seed}
    )

    random_rows = []
    for repetition in range(1, N_RANDOM_SUBSAMPLES + 1):
        train_ids, test_ids = train_test_split(
            row_ids,
            test_size=0.4,
            random_state=SEED + repetition,
            stratify=targets,
        )
        assignment = np.full(len(row_ids), "train", dtype=object)
        assignment[test_ids] = "test"
        random_rows.append(
            pd.DataFrame(
                {
                    "repetition": repetition,
                    "row_id": row_ids,
                    "partition": assignment,
                    "target": targets,
                }
            )
        )
    write_manifest(out_dir / "random_subsampling_30.csv", pd.concat(random_rows, ignore_index=True))
    summaries.append(
        {"dataset": name, "protocol": "Random Subsampling 60/40", "train": int(round(len(row_ids) * 0.6)), "test": int(round(len(row_ids) * 0.4)), "folds_or_repetitions": N_RANDOM_SUBSAMPLES, "seed": SEED}
    )

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    kfold_rows = []
    for fold, (_, test_index) in enumerate(kfold.split(row_ids), start=1):
        kfold_rows.append(pd.DataFrame({"row_id": row_ids[test_index], "test_fold": fold, "target": targets[test_index]}))
    write_manifest(out_dir / "kfold_10.csv", pd.concat(kfold_rows, ignore_index=True))
    summaries.append(
        {"dataset": name, "protocol": "K-Fold 10", "train": len(row_ids) - int(np.ceil(len(row_ids) / N_SPLITS)), "test": int(np.ceil(len(row_ids) / N_SPLITS)), "folds_or_repetitions": N_SPLITS, "seed": SEED}
    )

    stratified = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    stratified_rows = []
    for fold, (_, test_index) in enumerate(stratified.split(row_ids, targets), start=1):
        stratified_rows.append(pd.DataFrame({"row_id": row_ids[test_index], "test_fold": fold, "target": targets[test_index]}))
    write_manifest(out_dir / "stratified_cross_validation_10.csv", pd.concat(stratified_rows, ignore_index=True))
    summaries.append(
        {"dataset": name, "protocol": "Stratified Cross-Validation 10", "train": len(row_ids) - int(np.ceil(len(row_ids) / N_SPLITS)), "test": int(np.ceil(len(row_ids) / N_SPLITS)), "folds_or_repetitions": N_SPLITS, "seed": SEED}
    )

    loo_rows = pd.DataFrame({"row_id": row_ids, "test_fold": row_ids, "target": targets})
    write_manifest(out_dir / "leave_one_out.csv", loo_rows)
    summaries.append(
        {"dataset": name, "protocol": "Leave-One-Out", "train": len(row_ids) - 1, "test": 1, "folds_or_repetitions": len(row_ids), "seed": "N/A"}
    )
    return summaries


def main() -> None:
    iris_source = pd.read_csv(DATA_DIR / "iris.csv")
    mobile_source = pd.read_csv(DATA_DIR / "Mobile Price Classification" / "train.csv")
    mobile_test = pd.read_csv(DATA_DIR / "Mobile Price Classification" / "test.csv")

    validate_dataset(iris_source, "Iris", "species")
    validate_dataset(mobile_source, "Mobile Price Classification train", "price_range")
    if "price_range" in mobile_test.columns:
        raise ValueError("Se esperaba que el test.csv de Mobile no tuviera price_range.")

    iris_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    mobile_features = [column for column in mobile_source.columns if column != "price_range"]
    if iris_source[iris_features].select_dtypes(exclude="number").shape[1]:
        raise ValueError("Las variables predictoras de Iris deben ser numéricas.")
    if mobile_source[mobile_features].select_dtypes(exclude="number").shape[1]:
        raise ValueError("Las variables predictoras de Mobile deben ser numéricas.")

    iris_prepared = iris_source[iris_features].copy()
    iris_prepared.insert(0, "row_id", np.arange(len(iris_source), dtype=int))
    iris_prepared["target"] = iris_source["species"].eq("versicolor").astype(int)

    mobile_prepared = mobile_source[mobile_features].copy()
    mobile_prepared.insert(0, "row_id", np.arange(len(mobile_source), dtype=int))
    mobile_prepared["target"] = mobile_source["price_range"].astype(int)

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    iris_prepared.to_csv(PREPARED_DIR / "iris_binario.csv", index=False, encoding="utf-8")
    mobile_prepared.to_csv(PREPARED_DIR / "mobile_price.csv", index=False, encoding="utf-8")

    summaries = []
    summaries.extend(create_manifests("iris", iris_prepared["target"].to_numpy(), 75, SEED + 1))
    summaries.extend(create_manifests("mobile_price", mobile_prepared["target"].to_numpy(), 100, SEED + 2))

    # Replace Iris' random 75/75 assignment with the requested deterministic
    # alternating-row split while keeping all other protocols stratified.
    iris_ids = iris_prepared["row_id"].to_numpy()
    iris_targets = iris_prepared["target"].to_numpy()
    iris_train_ids = iris_ids[::2]
    iris_test_ids = iris_ids[1::2]
    write_manifest(
        SPLITS_DIR / "iris" / "train_test_75_75.csv",
        make_partition_manifest(iris_ids, iris_targets, iris_train_ids, iris_test_ids),
    )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(SPLITS_DIR / "resumen_particiones.csv", index=False, encoding="utf-8")

    (SPLITS_DIR / "README.md").write_text(
        "# Particiones guardadas\n\n"
        "Los `row_id` son índices desde cero y corresponden al orden de las filas en cada CSV de `Datos preparados`; excluye `row_id` de las variables predictoras. En Iris, `target=1` significa versicolor y `target=0` significa no versicolor. En Mobile, `target` conserva `price_range` (0 a 3).\n\n"
        "- `holdout_60_40.csv` y `train_test_*.csv`: `partition` indica `train`, `test` o `unused`.\n"
        "- `random_subsampling_30.csv`: cada `repetition` contiene la asignación train/test de una repetición.\n"
        "- `kfold_10.csv` y `stratified_cross_validation_10.csv`: `test_fold` identifica el fold de prueba; en cada fold, las demás filas forman el entrenamiento.\n"
        "- `leave_one_out.csv`: cada fila es una observación de prueba en el fold indicado por `test_fold`; las demás observaciones forman el entrenamiento.\n"
        "- `resumen_particiones.csv`: tamaños de entrenamiento/prueba, número de folds o repeticiones y semilla.\n\n"
        "Mobile usa únicamente `train.csv` para estas particiones porque el `test.csv` original no tiene etiquetas.\n",
        encoding="utf-8",
    )

    print("Datos preparados:")
    print(f"  Iris: {len(iris_prepared)} filas, {len(iris_features)} variables, positivo={int(iris_prepared['target'].sum())}")
    print(f"  Mobile: {len(mobile_prepared)} filas, {len(mobile_features)} variables, {mobile_prepared['target'].nunique()} clases")
    print(f"Archivos guardados en: {PREPARED_DIR}")
    print(f"Particiones guardadas en: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
