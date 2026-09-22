#!/usr/bin/env python3
"""Unifica los CSV de HHAR sin normalizar sus datos.

El script concatena todos los CSV encontrados debajo de ``--input-dir``.
Conserva los valores y los nombres de las columnas de los archivos originales.
Como los experimentos de HHAR tienen encabezados diferentes, la salida usa la
union de todos los encabezados y deja vacias las columnas que no existan en un
archivo concreto.

No se convierten timestamps, no se renombran actividades, no se interpretan
unidades y no se descartan filas por valores invalidos. Solo se omiten filas
completamente vacias. Se agregan ``source_file`` y ``source_row_number`` para
trazabilidad.

Uso desde la raiz del proyecto::

    .venv/Scripts/python.exe \
      "U2/DataSets/heterogeneity+activity+recognition/unificar_hhar.py" \
      --overwrite

La salida predeterminada es ``hhar_unificado.csv`` y su auditoria es
``hhar_unificado_auditoria.csv`` dentro de ``--input-dir``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


AUDIT_COLUMNS = [
    "source_file",
    "original_header",
    "rows_read",
    "rows_written",
    "blank_rows_skipped",
    "extra_values_preserved",
]

EXPECTED_ACTIVITY_FILES = {
    "phones_accelerometer.csv",
    "phones_gyroscope.csv",
    "watch_accelerometer.csv",
    "watch_gyroscope.csv",
}


def discover_csv_files(
    input_dir: Path,
    excluded_paths: Iterable[Path],
) -> list[Path]:
    """Encuentra todos los CSV, incluyendo extensiones escritas como ``.CSV``."""

    excluded = {path.resolve() for path in excluded_paths}
    files: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        if path.resolve() in excluded:
            continue
        if any(part.lower() == "__macosx" for part in path.parts):
            continue
        if path.name.startswith("._"):
            continue
        files.append(path)
    return files


def missing_activity_files(files: Sequence[Path]) -> list[str]:
    """Informa si la descarga no contiene los cuatro CSV de actividad HHAR."""

    discovered = {path.name.lower() for path in files}
    return sorted(EXPECTED_ACTIVITY_FILES - discovered)


def read_header(path: Path) -> list[str]:
    """Lee solo el encabezado, sin cambiar sus nombres."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def column_key(name: str, index: int) -> str:
    """Crea una clave interna para alinear columnas sin cambiar su texto."""

    key = name.lstrip("\ufeff").strip().lower()
    return key or f"__unnamed_{index + 1}"


def header_keys(header: Sequence[str]) -> list[str]:
    """Devuelve claves estables y conserva columnas duplicadas si existieran."""

    occurrences: dict[str, int] = {}
    keys: list[str] = []
    for index, name in enumerate(header):
        base = column_key(name, index)
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        keys.append(base if occurrence == 0 else f"{base}__duplicate_{occurrence + 1}")
    return keys


def display_header_name(name: str, index: int) -> str:
    """Conserva el nombre original; solo asigna nombre a una cabecera vacia."""

    return name if name else f"column_{index + 1}"


def unique_name(preferred: str, used: set[str]) -> str:
    candidate = preferred
    suffix = 2
    while candidate in used:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def build_unified_header(
    headers_by_file: Mapping[Path, Sequence[str]],
    max_extra_fields: int,
) -> tuple[list[str], dict[str, int], list[int], list[int]]:
    """Construye la union de encabezados y las posiciones de trazabilidad."""

    unified_headers: list[str] = []
    unified_keys: list[str] = []
    seen_keys: set[str] = set()
    used_names: set[str] = set()

    for header in headers_by_file.values():
        keys = header_keys(header)
        for index, key in enumerate(keys):
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unified_keys.append(key)
            unified_headers.append(
                unique_name(display_header_name(header[index], index), used_names)
            )

    source_file_position = len(unified_headers)
    unified_headers.append(unique_name("source_file", used_names))
    source_row_position = len(unified_headers)
    unified_headers.append(unique_name("source_row_number", used_names))

    extra_positions: list[int] = []
    for index in range(max_extra_fields):
        extra_positions.append(len(unified_headers))
        unified_headers.append(unique_name(f"__extra_{index + 1}", used_names))

    positions = {key: index for index, key in enumerate(unified_keys)}
    return (
        unified_headers,
        positions,
        [source_file_position, source_row_position],
        extra_positions,
    )


def inspect_max_extra_fields(path: Path, header_length: int) -> int:
    """Calcula cuantas celdas excedentes necesita la salida."""

    maximum = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            extra_values = row[header_length:]
            while extra_values and not extra_values[-1].strip():
                extra_values.pop()
            maximum = max(maximum, len(extra_values))
    return maximum


def row_is_empty(row: Sequence[str]) -> bool:
    return not row or not any(value.strip() for value in row)


def process_file(
    path: Path,
    input_dir: Path,
    header: Sequence[str],
    output_writer: csv.writer,
    unified_positions: Mapping[str, int],
    source_positions: Sequence[int],
    extra_positions: Sequence[int],
) -> dict[str, str | int]:
    """Copia un CSV al esquema union sin convertir ninguna celda."""

    keys = header_keys(header)
    source_file = path.relative_to(input_dir).as_posix()
    rows_read = 0
    rows_written = 0
    blank_rows_skipped = 0
    extra_values_preserved = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for source_row_number, row in enumerate(reader, start=2):
            if row_is_empty(row):
                blank_rows_skipped += 1
                continue

            rows_read += 1
            output_row = [""] * (
                len(unified_positions) + 2 + len(extra_positions)
            )
            for index, key in enumerate(keys):
                if index >= len(row):
                    continue
                output_row[unified_positions[key]] = row[index]

            for extra_index, output_position in enumerate(extra_positions):
                source_index = len(header) + extra_index
                if source_index >= len(row):
                    continue
                output_row[output_position] = row[source_index]
                if row[source_index].strip():
                    extra_values_preserved += 1

            output_row[source_positions[0]] = source_file
            output_row[source_positions[1]] = str(source_row_number)
            output_writer.writerow(output_row)
            rows_written += 1

    return {
        "source_file": source_file,
        "original_header": "|".join(header),
        "rows_read": rows_read,
        "rows_written": rows_written,
        "blank_rows_skipped": blank_rows_skipped,
        "extra_values_preserved": extra_values_preserved,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unifica los CSV del dataset HHAR sin normalizar sus datos."
    )
    default_input = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Carpeta raiz que contiene los CSV (por defecto, la del script).",
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
        help="Auditoria; por defecto hhar_unificado_auditoria.csv dentro de input-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar las salidas si ya existen.",
    )
    parser.add_argument(
        "--require-complete-dataset",
        action="store_true",
        help="Falla si falta alguno de los cuatro CSV del experimento de actividad.",
    )
    return parser.parse_args()


def absolute_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


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

    files = discover_csv_files(input_dir, (output_path, audit_path))
    if not files:
        raise SystemExit(f"No se encontraron CSV validos dentro de {input_dir}")

    missing = missing_activity_files(files)
    if missing:
        message = (
            "ADVERTENCIA: faltan CSV del experimento de actividad: "
            + ", ".join(missing)
            + "."
        )
        if args.require_complete_dataset:
            raise SystemExit(message)
        print(message, file=sys.stderr)

    headers_by_file = {path: read_header(path) for path in files}
    max_extra_fields = max(
        inspect_max_extra_fields(path, len(headers_by_file[path])) for path in files
    )
    (
        output_headers,
        unified_positions,
        source_positions,
        extra_positions,
    ) = build_unified_header(headers_by_file, max_extra_fields)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_handle, audit_path.open(
        "w", encoding="utf-8", newline=""
    ) as audit_handle:
        output_writer = csv.writer(output_handle)
        audit_writer = csv.DictWriter(audit_handle, fieldnames=AUDIT_COLUMNS)
        output_writer.writerow(output_headers)
        audit_writer.writeheader()

        total_read = 0
        total_written = 0
        total_blank = 0
        total_extra = 0
        for number, path in enumerate(files, start=1):
            audit = process_file(
                path,
                input_dir,
                headers_by_file[path],
                output_writer,
                unified_positions,
                source_positions,
                extra_positions,
            )
            audit_writer.writerow(audit)
            total_read += int(audit["rows_read"])
            total_written += int(audit["rows_written"])
            total_blank += int(audit["blank_rows_skipped"])
            total_extra += int(audit["extra_values_preserved"])
            print(
                f"[{number:03d}/{len(files):03d}] {audit['source_file']}: "
                f"{audit['rows_written']} filas"
            )

    print(f"\nCSV unificado: {output_path}")
    print(f"Auditoria: {audit_path}")
    print(f"Archivos procesados: {len(files)}")
    print(f"Columnas de salida: {len(output_headers)}")
    print(f"Filas leidas: {total_read}")
    print(f"Filas escritas: {total_written}")
    print(f"Filas vacias omitidas: {total_blank}")
    print(f"Valores extra preservados: {total_extra}")


if __name__ == "__main__":
    main()
