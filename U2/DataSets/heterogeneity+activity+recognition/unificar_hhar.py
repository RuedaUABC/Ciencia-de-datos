#!/usr/bin/env python3
"""Normaliza y unifica los CSV del conjunto HHAR.

El archivo descargado contiene dos experimentos con estructuras distintas y
cada uno puede estar distribuido en varias subcarpetas:

* ``Still exp``: acelerómetro estático repartido en muchos archivos, con tres
  variantes de cabecera y sin etiqueta de actividad explícita.
* ``Activity recognition exp``: archivos de actividad con usuario, dispositivo
  y ground truth.

El descubrimiento no depende de que exista una carpeta con un nombre concreto:
se recorren todos los subdirectorios de la carpeta de entrada y el formato se
determina por la cabecera de cada CSV. Así también se admiten variantes del
dataset con nombres de carpetas diferentes.

La salida es un CSV largo con un esquema estable. No se aplica un z-score a
los ejes por defecto: los valores se mantienen en sus unidades físicas para
no eliminar la información de orientación del experimento estático. La
normalización realizada aquí es estructural y de unidades de tiempo.

Uso desde la raíz del proyecto:

    .venv/bin/python \
      "U2/DataSets/heterogeneity+activity+recognition/unificar_hhar.py" \
      --overwrite

Se producen ``hhar_unificado.csv`` y ``hhar_unificado_auditoria.csv`` en la
carpeta de entrada, salvo que se indiquen otras rutas con ``--output`` y
``--audit``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


ORIENTATIONS = {
    "phoneonback": "back",
    "phoneonfront": "front",
    "phoneonbottom": "bottom",
    "phoneontop": "top",
    "phoneonleft": "left",
    "phoneonright": "right",
}

CANONICAL_COLUMNS = [
    "experiment",
    "activity",
    "activity_raw",
    "sensor_type",
    "sensor_unit",
    "user",
    "device_type",
    "device_id",
    "device_model",
    "orientation",
    "sample_index",
    "source_row_number",
    "arrival_time_ms",
    "creation_time_raw",
    "creation_time_ms",
    "creation_time_unit",
    "creation_time_origin",
    "sensor_time_raw",
    "sensor_time_ms",
    "x",
    "y",
    "z",
    "source_schema",
    "source_file",
]

MACOS_METADATA_DIR = "__macosx"

AUDIT_COLUMNS = [
    "source_file",
    "original_header",
    "source_schema",
    "experiment",
    "orientation",
    "device_id",
    "device_model",
    "creation_time_unit",
    "rows_read",
    "rows_written",
    "rows_skipped",
    "extra_fields_discarded",
]


def clean_header_name(value: str | None) -> str:
    """Convierte nombres de columna a una forma comparable."""

    if value is None:
        return ""
    value = value.strip().lstrip("\ufeff").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    aliases = {
        "arrivaltime": "arrival_time",
        "creationtime": "creation_time",
        "sensortime": "sensor_time",
        "sample_index": "index",
        "groundtruth": "gt",
    }
    return aliases.get(value, value)


def header_mapping(header: Sequence[str]) -> dict[str, int]:
    """Devuelve el índice de cada columna conocida, ignorando extras."""

    mapping: dict[str, int] = {}
    for index, raw_name in enumerate(header):
        name = clean_header_name(raw_name)
        if name and name not in mapping:
            mapping[name] = index
    return mapping


def cell(row: Sequence[str], mapping: Mapping[str, int], name: str) -> str:
    index = mapping.get(name)
    if index is None or index >= len(row):
        return ""
    return row[index].strip()


def parse_float(value: str) -> float | None:
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def format_number(value: float | int | None) -> str:
    """Formatea números para que pandas/Excel los reconozcan al leer la salida."""

    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def format_raw_number(value: str) -> str:
    """Normaliza blancos sin alterar el valor original del timestamp."""

    return value.strip() if value else ""


def discover_csv_files(
    input_dir: Path,
    excluded_paths: Iterable[Path],
) -> list[Path]:
    """Encuentra mediciones CSV en cualquier profundidad del dataset.

    ``Path.rglob("*.csv")`` es sensible a mayúsculas en algunos sistemas y
    puede incluir archivos de recursos de macOS. Iterar sobre todos los
    archivos permite aceptar extensiones como ``.CSV`` y filtrar esos metadatos
    sin restringir el descubrimiento a ``Still exp``.
    """

    excluded = {path.resolve() for path in excluded_paths}
    files = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() != ".csv":
            continue
        resolved = path.resolve()
        if resolved in excluded:
            continue
        if any(part.casefold() == MACOS_METADATA_DIR for part in path.parts):
            continue
        if path.name.startswith("._"):
            continue
        files.append(path)
    return files


def read_header_and_samples(
    path: Path,
    sample_limit: int = 1000,
) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        samples = []
        for row in reader:
            if not row or not any(value.strip() for value in row):
                continue
            samples.append(row)
            if len(samples) >= sample_limit:
                break
    return header, samples


def infer_creation_unit(
    mapping: Mapping[str, int],
    samples: Sequence[Sequence[str]],
) -> str:
    """Infiere la unidad del Creation_Time por archivo.

    HHAR mezcla nanosegundos, microsegundos y milisegundos. X-Sense es el caso
    reconocible como epoch-ms porque Creation_Time y Arrival_Time coinciden.
    Para los otros archivos, el salto entre muestras permite distinguir los
    microsegundos del resto; el conjunto entregado tiene unidades constantes
    por archivo.
    """

    creation_values = []
    arrival_values = []
    for row in samples:
        creation = parse_float(cell(row, mapping, "creation_time"))
        arrival = parse_float(cell(row, mapping, "arrival_time"))
        if creation is not None:
            creation_values.append(creation)
        if arrival is not None:
            arrival_values.append(arrival)

    if not creation_values:
        return "missing"

    paired = min(len(creation_values), len(arrival_values))
    if paired:
        equal_pairs = sum(
            abs(creation_values[i] - arrival_values[i]) <= 1
            for i in range(paired)
        )
        if equal_pairs / paired >= 0.95:
            return "epoch_ms"

    deltas = [
        abs(right - left)
        for left, right in zip(creation_values, creation_values[1:])
        if right != left
    ]
    median_delta = statistics.median(deltas) if deltas else 0
    median_value = statistics.median(abs(value) for value in creation_values)

    if median_value >= 1e12 or median_delta >= 1e5:
        return "nanoseconds"
    if median_value >= 1e8 or median_delta >= 100:
        return "microseconds"
    return "unknown"


def convert_creation_time(value: str, unit: str) -> float | None:
    raw = parse_float(value)
    if raw is None:
        return None
    if unit == "epoch_ms":
        return raw
    if unit == "nanoseconds":
        return raw / 1_000_000
    if unit == "microseconds":
        return raw / 1_000
    return None


def orientation_from_path(path: Path) -> str:
    for part in path.parts:
        normalized = part.lower().replace(" ", "")
        if normalized in ORIENTATIONS:
            return ORIENTATIONS[normalized]
    return "unknown"


def experiment_from_path(path: Path, mapping: Mapping[str, int]) -> str:
    """Identifica el experimento usando la estructura del CSV.

    La cabecera es la fuente de verdad: los archivos de actividad incluyen
    ``gt`` y metadatos del usuario/dispositivo, mientras que los del
    experimento estático solo contienen las mediciones y timestamps. Esto
    permite procesar ambos experimentos aunque se cambien los nombres o la
    profundidad de sus carpetas.
    """

    activity_columns = {"gt", "user", "model", "device"}
    if "gt" in mapping or activity_columns.intersection(mapping):
        return "activity_recognition"
    return "still"


def normalize_activity(value: str, experiment: str) -> str:
    if experiment == "still":
        return "still"
    normalized = value.strip().lower()
    if normalized in {"", "null", "none", "nan", "unknown"}:
        return "unknown"
    normalized = re.sub(r"[\s_-]+", "", normalized)
    labels = {
        "bike": "biking",
        "biking": "biking",
        "sit": "sitting",
        "sitting": "sitting",
        "stand": "standing",
        "standing": "standing",
        "walk": "walking",
        "walking": "walking",
        "stairsup": "stair_up",
        "stairup": "stair_up",
        "stairsdown": "stair_down",
        "stairdown": "stair_down",
    }
    return labels.get(normalized, value.strip().lower())


def infer_sensor_type(path: Path, experiment: str) -> str:
    if experiment == "still":
        return "accelerometer"
    return "gyroscope" if "gyro" in path.stem.lower() else "accelerometer"


def infer_device_type(device_model: str, device_id: str, path: Path) -> str:
    text = " ".join((device_model, device_id, path.stem, "/".join(path.parts))).lower()
    if "gear" in text or "watch" in text:
        return "smartwatch"
    if "x-sense" in text:
        return "other"
    return "smartphone"


def path_metadata(
    path: Path,
    input_dir: Path,
    mapping: Mapping[str, int],
    first_sample: Sequence[str],
) -> dict[str, str]:
    experiment = experiment_from_path(path, mapping)
    source_model = cell(first_sample, mapping, "model")
    source_device = cell(first_sample, mapping, "device")

    if experiment == "activity_recognition":
        device_model = source_model or path.stem
        device_id = source_device or path.stem
    elif path.stem.lower() == "x-sense":
        device_model = "x-sense"
        device_id = "x-sense"
    else:
        device_model = path.stem
        device_id = path.parent.name

    sensor_type = infer_sensor_type(path, experiment)
    return {
        "experiment": experiment,
        "activity": normalize_activity(cell(first_sample, mapping, "gt"), experiment),
        "activity_raw": (
            cell(first_sample, mapping, "gt") if experiment != "still" else "still"
        ),
        "sensor_type": sensor_type,
        "sensor_unit": "m/s2" if sensor_type == "accelerometer" else "unknown",
        "user": cell(first_sample, mapping, "user"),
        "device_type": infer_device_type(device_model, device_id, path),
        "device_id": device_id,
        "device_model": device_model,
        "orientation": orientation_from_path(path),
        "source_file": path.relative_to(input_dir).as_posix(),
    }


def source_schema(experiment: str, mapping: Mapping[str, int]) -> str:
    if experiment == "activity_recognition":
        return "activity_recognition"
    if "sensor_time" in mapping:
        return "still_creation_sensor_arrival"
    if "index" in mapping:
        return "still_index_arrival_creation"
    return "still_arrival_creation"


def row_is_empty(row: Sequence[str]) -> bool:
    return not row or not any(value.strip() for value in row)


def normalized_row(
    row: Sequence[str],
    mapping: Mapping[str, int],
    metadata: Mapping[str, str],
    creation_unit: str,
    schema: str,
    source_row_number: int,
) -> list[str] | None:
    axes = [parse_float(cell(row, mapping, axis)) for axis in ("x", "y", "z")]
    if any(axis is None for axis in axes):
        return None

    row_metadata = dict(metadata)
    if metadata["experiment"] == "activity_recognition":
        row_metadata["activity_raw"] = cell(row, mapping, "gt")
        row_metadata["activity"] = normalize_activity(
            row_metadata["activity_raw"], metadata["experiment"]
        )
        row_metadata["user"] = cell(row, mapping, "user") or metadata["user"]
        row_metadata["device_model"] = (
            cell(row, mapping, "model") or metadata["device_model"]
        )
        row_metadata["device_id"] = (
            cell(row, mapping, "device") or metadata["device_id"]
        )
        row_metadata["device_type"] = infer_device_type(
            row_metadata["device_model"], row_metadata["device_id"], Path()
        )

    arrival = parse_float(cell(row, mapping, "arrival_time"))
    creation_raw = format_raw_number(cell(row, mapping, "creation_time"))
    creation_ms = convert_creation_time(creation_raw, creation_unit)
    sensor_raw = format_raw_number(cell(row, mapping, "sensor_time"))
    sensor_ms = parse_float(sensor_raw) if sensor_raw else None
    origin = "epoch" if creation_unit == "epoch_ms" else "device_clock_relative"

    values = {
        **row_metadata,
        "sample_index": format_raw_number(cell(row, mapping, "index")),
        "source_row_number": str(source_row_number),
        "arrival_time_ms": format_number(arrival),
        "creation_time_raw": creation_raw,
        "creation_time_ms": format_number(creation_ms),
        "creation_time_unit": creation_unit,
        "creation_time_origin": origin,
        "sensor_time_raw": sensor_raw,
        "sensor_time_ms": format_number(sensor_ms),
        "x": format_number(axes[0]),
        "y": format_number(axes[1]),
        "z": format_number(axes[2]),
        "source_schema": schema,
    }
    return [values[column] for column in CANONICAL_COLUMNS]


def process_file(
    path: Path,
    input_dir: Path,
    output_writer: csv.writer,
) -> dict[str, str | int]:
    header, samples = read_header_and_samples(path)
    mapping = header_mapping(header)
    if not samples:
        first_sample: list[str] = []
    else:
        first_sample = samples[0]

    metadata = path_metadata(path, input_dir, mapping, first_sample)
    experiment = metadata["experiment"]
    schema = source_schema(experiment, mapping)
    creation_unit = infer_creation_unit(mapping, samples)

    rows_read = 0
    rows_written = 0
    rows_skipped = 0
    extra_fields_discarded = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for source_row_number, row in enumerate(reader):
            if row_is_empty(row):
                continue
            rows_read += 1
            extra_fields_discarded += max(0, len(row) - len(header))
            result = normalized_row(
                row,
                mapping,
                metadata,
                creation_unit,
                schema,
                source_row_number,
            )
            if result is None:
                rows_skipped += 1
                continue
            output_writer.writerow(result)
            rows_written += 1

    return {
        "source_file": metadata["source_file"],
        "original_header": "|".join(header),
        "source_schema": schema,
        "experiment": metadata["experiment"],
        "orientation": metadata["orientation"],
        "device_id": metadata["device_id"],
        "device_model": metadata["device_model"],
        "creation_time_unit": creation_unit,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "rows_skipped": rows_skipped,
        "extra_fields_discarded": extra_fields_discarded,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normaliza y unifica los CSV del dataset HHAR."
    )
    default_input = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Carpeta raíz que contiene los CSV (por defecto, la del script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV unificado; por defecto hhar_unificado.csv dentro de input-dir.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Auditoría; por defecto hhar_unificado_auditoria.csv dentro de input-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar las salidas si ya existen.",
    )
    return parser.parse_args()


def absolute_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"No existe la carpeta de entrada: {input_dir}")

    output_path = (
        input_dir / "hhar_unificado.csv"
        if args.output is None
        else absolute_path(args.output, input_dir)
    )
    audit_path = (
        input_dir / "hhar_unificado_auditoria.csv"
        if args.audit is None
        else absolute_path(args.audit, input_dir)
    )
    if not args.overwrite:
        existing = [path for path in (output_path, audit_path) if path.exists()]
        if existing:
            raise SystemExit(
                "Ya existen salidas. Usa --overwrite para reemplazarlas: "
                + ", ".join(str(path) for path in existing)
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    files = discover_csv_files(input_dir, (output_path, audit_path))
    if not files:
        raise SystemExit(f"No se encontraron CSV válidos dentro de {input_dir}")

    with output_path.open("w", encoding="utf-8", newline="") as output_handle, audit_path.open(
        "w", encoding="utf-8", newline=""
    ) as audit_handle:
        output_writer = csv.writer(output_handle)
        audit_writer = csv.DictWriter(audit_handle, fieldnames=AUDIT_COLUMNS)
        output_writer.writerow(CANONICAL_COLUMNS)
        audit_writer.writeheader()

        total_written = 0
        total_skipped = 0
        for number, path in enumerate(files, start=1):
            audit = process_file(path, input_dir, output_writer)
            audit_writer.writerow(audit)
            total_written += int(audit["rows_written"])
            total_skipped += int(audit["rows_skipped"])
            print(
                f"[{number:03d}/{len(files):03d}] {audit['source_file']}: "
                f"{audit['rows_written']} filas"
            )

    print(f"\nCSV unificado: {output_path}")
    print(f"Auditoría: {audit_path}")
    print(f"Archivos procesados: {len(files)}")
    print(f"Filas escritas: {total_written}")
    print(f"Filas omitidas por ejes inválidos: {total_skipped}")


if __name__ == "__main__":
    main()
