"""Imputa los valores faltantes del dataset de permisos de construcción de SF.

El script implementa las reglas descritas en ``SFBuildingPermits.md`` y no
sobrescribe valores que ya existen. Además del dataset imputado, genera un
CSV de auditoría con los valores faltantes antes y después del proceso.

Uso con KaggleHub:
    python imputar_SFBuildingPermits.py

Uso con un CSV local:
    python imputar_SFBuildingPermits.py --input Building_Permits.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


KAGGLE_DATASET = "aparnashastry/building-permit-applications-data"
KAGGLE_FILE = "Building_Permits.csv"
MIN_OBSERVACIONES_MEDIANA = 5
MAX_IQR_MEDIANA_FECHAS = 1.0
UMBRAL_NULOS_FILAS = 0.50
UMBRAL_NULOS_COLUMNAS = 0.60

FECHAS = [
    "Current Status Date",
    "Filed Date",
    "Permit Creation Date",
    "Issued Date",
    "Completed Date",
    "First Construction Document Date",
    "Permit Expiration Date",
]

# Orden cronológico de los eventos del trámite. La fecha de expiración se
# trata aparte porque puede coincidir con el fin del permiso, no con el fin
# del proyecto.
ORDEN_EVENTOS = [
    "Filed Date",
    "Permit Creation Date",
    "Issued Date",
    "First Construction Document Date",
    "Completed Date",
]


def registrar(auditoria: list[dict[str, Any]], regla: str, columna: str, cantidad: int) -> None:
    """Agrega a la auditoría una regla que haya imputado al menos un valor."""

    if cantidad:
        auditoria.append(
            {
                "tipo": "regla",
                "regla": regla,
                "columna": columna,
                "cantidad": int(cantidad),
            }
        )


def asignar_solo_nulos(
    df: pd.DataFrame,
    columna: str,
    candidatos: pd.Series,
    auditoria: list[dict[str, Any]],
    regla: str,
    mascara: pd.Series | None = None,
) -> int:
    """Asigna candidatos únicamente donde la columna destino está vacía."""

    if columna not in df.columns:
        return 0

    candidatos = candidatos.reindex(df.index)
    seleccion = df[columna].isna() & candidatos.notna()
    if mascara is not None:
        seleccion &= mascara.reindex(df.index, fill_value=False)

    cantidad = int(seleccion.sum())
    if cantidad:
        df.loc[seleccion, columna] = candidatos.loc[seleccion]
        registrar(auditoria, regla, columna, cantidad)
    return cantidad


def normalizar_nulos(df: pd.DataFrame) -> None:
    """Convierte marcadores textuales comunes de ausencia a valores nulos."""

    marcadores = {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "null": pd.NA, "N/A": pd.NA, "NA": pd.NA}
    for columna in df.select_dtypes(include=["object", "string"]).columns:
        df[columna] = df[columna].replace(marcadores)


def cargar_dataset(ruta_entrada: Path | None) -> pd.DataFrame:
    """Carga un CSV local o la versión de Kaggle usada por el análisis."""

    if ruta_entrada is not None:
        kwargs: dict[str, Any] = {"encoding": "latin1", "low_memory": False}
        if ruta_entrada.suffix.lower() == ".zip":
            kwargs["compression"] = "zip"
        return pd.read_csv(ruta_entrada, **kwargs)

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as error:
        raise RuntimeError(
            "No se encontró kagglehub. Usa --input para indicar un CSV local "
            "o instala kagglehub."
        ) from error

    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET,
        KAGGLE_FILE,
        pandas_kwargs={
            "encoding": "latin1",
            "compression": "zip",
            "low_memory": False,
        },
    )


def convertir_fechas(df: pd.DataFrame) -> None:
    """Convierte las columnas de fecha disponibles sin fallar por formatos mixtos."""

    for columna in FECHAS:
        if columna not in df.columns:
            continue
        try:
            df[columna] = pd.to_datetime(df[columna], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            # ``format='mixed'`` no existe en versiones antiguas de pandas.
            df[columna] = pd.to_datetime(df[columna], errors="coerce")


def obtener_moda(serie: pd.Series) -> Any:
    """Obtiene la moda de una serie, usando el primer valor en caso de empate."""

    valores = serie.dropna().mode()
    return valores.iloc[0] if not valores.empty else pd.NA


def clave_texto(valor: Any) -> str:
    return str(valor).strip().casefold()


def completar_par_codigo_descripcion(
    df: pd.DataFrame,
    columna_codigo: str,
    columna_descripcion: str,
    auditoria: list[dict[str, Any]],
) -> None:
    """Completa un código o su descripción a partir de los pares observados."""

    if columna_codigo not in df.columns or columna_descripcion not in df.columns:
        return

    observados = df[[columna_codigo, columna_descripcion]].dropna()
    if observados.empty:
        return

    mapa_codigo = observados.groupby(columna_codigo, dropna=False)[columna_descripcion].agg(obtener_moda)
    candidatos_descripcion = df[columna_codigo].map(mapa_codigo)
    asignar_solo_nulos(
        df,
        columna_descripcion,
        candidatos_descripcion,
        auditoria,
        f"{columna_codigo} -> {columna_descripcion}",
    )

    # Para la dirección inversa se normaliza el texto, evitando que diferencias
    # de mayúsculas o espacios impidan encontrar una equivalencia.
    observados = df[[columna_codigo, columna_descripcion]].dropna()
    claves_descripcion = observados[columna_descripcion].map(clave_texto)
    mapa_descripcion = (
        pd.DataFrame(
            {
                "clave": claves_descripcion,
                "valor": observados[columna_codigo].to_numpy(),
            }
        )
        .groupby("clave", dropna=False)["valor"]
        .agg(obtener_moda)
    )
    candidatos_codigo = df[columna_descripcion].map(clave_texto).map(mapa_descripcion)
    asignar_solo_nulos(
        df,
        columna_codigo,
        candidatos_codigo,
        auditoria,
        f"{columna_descripcion} -> {columna_codigo}",
    )


def construir_clave(df: pd.DataFrame, columnas: Iterable[str]) -> pd.Series:
    """Construye una clave compuesta; una clave incompleta se marca como nula."""

    columnas = list(columnas)
    if any(columna not in df.columns for columna in columnas):
        return pd.Series(pd.NA, index=df.index, dtype="string")

    partes = df[columnas].astype("string").apply(lambda serie: serie.str.strip().str.casefold())
    # Se reemplazan temporalmente los faltantes para que ``join`` no intente
    # concatenar objetos ``pd.NA``; después se invalidan esas claves.
    clave = partes.fillna("<NA>").agg("\x1f".join, axis=1)
    return clave.mask(partes.isna().any(axis=1))


def completar_por_ubicacion(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Completa datos geográficos usando otras filas de la misma dirección."""

    objetivos = [
        "Zipcode",
        "Supervisor District",
        "Neighborhoods - Analysis Boundaries",
        "Location",
    ]
    claves = [
        ["Block", "Lot"],
        ["Street Number", "Street Name", "Street Name Suffix"],
        ["Street Number", "Street Name"],
    ]

    for columnas_clave in claves:
        clave = construir_clave(df, columnas_clave)
        validas = clave.notna()
        if not validas.any():
            continue

        for objetivo in objetivos:
            if objetivo not in df.columns:
                continue
            observados = pd.DataFrame({"clave": clave, "valor": df[objetivo]})
            observados = observados.loc[validas & observados["valor"].notna()]
            if observados.empty:
                continue
            mapa = observados.groupby("clave", dropna=False)["valor"].agg(obtener_moda)
            candidatos = clave.map(mapa)
            completar = int((df[objetivo].isna() & candidatos.notna()).sum())
            asignar_solo_nulos(
                df,
                objetivo,
                candidatos,
                auditoria,
                f"lookup por {' + '.join(columnas_clave)}",
            )
            if not completar:
                continue


def detectar_cambio_estructural(df: pd.DataFrame) -> pd.Series:
    """Identifica descripciones donde no es seguro copiar existente a propuesto."""

    textos = pd.Series("", index=df.index, dtype="string")
    for columna in ["Permit Type Definition", "Description"]:
        if columna in df.columns:
            textos = textos + " " + df[columna].astype("string").fillna("")

    patron = (
        r"change\s+of\s+use|new\s+(?:construction|building)|vertical\s+addition|"
        r"horizontal\s+addition|\baddition\b|\bdemolition\b|\bdemolish\w*|"
        r"\brebuild\w*|\breconstruction\b|\bmajor\b|\bextension\b"
    )
    return textos.str.contains(patron, case=False, regex=True, na=False)


def completar_inercia_existente_propuesto(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Copia propiedades existente/propuesta solo en permisos sin cambio evidente."""

    pares = [
        ("Existing Use", "Proposed Use"),
        ("Existing Units", "Proposed Units"),
        ("Number of Existing Stories", "Number of Proposed Stories"),
        ("Existing Construction Type", "Proposed Construction Type"),
    ]
    sin_cambio = ~detectar_cambio_estructural(df)

    for izquierda, derecha in pares:
        if izquierda not in df.columns or derecha not in df.columns:
            continue

        asignar_solo_nulos(
            df,
            derecha,
            df[izquierda],
            auditoria,
            f"{izquierda} -> {derecha} por inercia",
            sin_cambio,
        )
        asignar_solo_nulos(
            df,
            izquierda,
            df[derecha],
            auditoria,
            f"{derecha} -> {izquierda} por inercia",
            sin_cambio,
        )


def ajustar_regresion_lineal(
    df: pd.DataFrame,
    origen: str,
    destino: str,
    permit_type: Any = None,
) -> tuple[float, float] | None:
    """Ajusta destino = intercepto + pendiente * origen."""

    if origen not in df.columns or destino not in df.columns:
        return None

    x = pd.to_numeric(df[origen], errors="coerce")
    y = pd.to_numeric(df[destino], errors="coerce")
    validos = x.notna() & y.notna()
    if permit_type is not None and "Permit Type" in df.columns:
        por_tipo = df["Permit Type"].isna() if pd.isna(permit_type) else df["Permit Type"].eq(permit_type)
        muestra = validos & por_tipo
        if int(muestra.sum()) >= MIN_OBSERVACIONES_MEDIANA:
            validos = muestra

    if int(validos.sum()) < MIN_OBSERVACIONES_MEDIANA:
        return None

    x_validos = x.loc[validos]
    y_validos = y.loc[validos]
    variacion_x = ((x_validos - x_validos.mean()) ** 2).sum()
    if not variacion_x:
        return None
    pendiente = ((x_validos - x_validos.mean()) * (y_validos - y_validos.mean())).sum() / variacion_x
    intercepto = y_validos.mean() - pendiente * x_validos.mean()
    return float(intercepto), float(pendiente)


def completar_por_regresion(
    df: pd.DataFrame,
    origen: str,
    destino: str,
    auditoria: list[dict[str, Any]],
    tipo_variable: str,
) -> None:
    """Completa un par correlacionado con modelos por tipo y respaldo global."""

    if origen not in df.columns or destino not in df.columns:
        return

    candidatos = pd.Series(float("nan"), index=df.index)
    faltantes = df[destino].isna()
    if not faltantes.any():
        return

    if "Permit Type" in df.columns:
        filas_validas = df.loc[faltantes & df[origen].notna()]
        for permit_type, grupo in filas_validas.groupby("Permit Type", dropna=False):
            modelo = ajustar_regresion_lineal(df, origen, destino, permit_type)
            if modelo is None:
                modelo = ajustar_regresion_lineal(df, origen, destino)
            if modelo is None:
                continue
            intercepto, pendiente = modelo
            indices = grupo.index
            x = pd.to_numeric(df.loc[indices, origen], errors="coerce")
            candidatos.loc[indices] = intercepto + pendiente * x
    else:
        modelo = ajustar_regresion_lineal(df, origen, destino)
        if modelo is not None:
            intercepto, pendiente = modelo
            x = pd.to_numeric(df[origen], errors="coerce")
            candidatos = intercepto + pendiente * x

    if tipo_variable in {"entero", "categoria"}:
        candidatos = candidatos.round()
        candidatos = candidatos.clip(lower=0)

    if tipo_variable == "categoria":
        valores_observados = pd.to_numeric(df[destino], errors="coerce").dropna().unique()
        if len(valores_observados):
            candidatos = candidatos.map(
                lambda valor: min(valores_observados, key=lambda observado: abs(observado - valor))
                if pd.notna(valor)
                else valor
            )

    asignar_solo_nulos(
        df,
        destino,
        candidatos,
        auditoria,
        f"regresión lineal {origen} -> {destino} por Permit Type",
    )


def completar_pares_correlacionados(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Aplica regresión a los pares con correlación alta reportados en las notas."""

    pares = [
        ("Number of Existing Stories", "Number of Proposed Stories", "entero"),
        ("Estimated Cost", "Revised Cost", "continuo"),
        ("Existing Units", "Proposed Units", "entero"),
        ("Existing Construction Type", "Proposed Construction Type", "categoria"),
    ]
    for origen, destino, tipo_variable in pares:
        completar_por_regresion(df, origen, destino, auditoria, tipo_variable)
        completar_por_regresion(df, destino, origen, auditoria, tipo_variable)


def estadisticas_diferencia(
    df: pd.DataFrame,
    inicio: str,
    fin: str,
    mascara: pd.Series | None = None,
) -> tuple[float, float, float] | None:
    """Devuelve mediana, IQR e IQR/mediana de una diferencia de fechas."""

    if inicio not in df.columns or fin not in df.columns:
        return None

    validos = df[inicio].notna() & df[fin].notna()
    if mascara is not None:
        validos &= mascara.reindex(df.index, fill_value=False)
    if int(validos.sum()) < MIN_OBSERVACIONES_MEDIANA:
        return None

    dias = (
        (df.loc[validos, fin] - df.loc[validos, inicio])
        .abs()
        .dt.total_seconds()
        .div(86400)
    )
    mediana = dias.median()
    iqr = dias.quantile(0.75) - dias.quantile(0.25)
    if pd.isna(mediana) or pd.isna(iqr):
        return None

    # Replica los valores de las notas: 0/0 se considera una diferencia
    # perfectamente estable; cualquier IQR distinto de cero con mediana 0
    # queda fuera por tener una razón infinita.
    if mediana == 0:
        razon = 0.0 if iqr == 0 else float("inf")
    else:
        razon = float(iqr) / abs(float(mediana))
    return float(mediana), float(iqr), razon


def mediana_diferencia(
    df: pd.DataFrame,
    inicio: str,
    fin: str,
    permit_type: Any = None,
) -> float | None:
    """Calcula una mediana de días, primero por tipo y luego globalmente."""

    if permit_type is not None:
        if pd.isna(permit_type):
            por_tipo = df.get("Permit Type", pd.Series(pd.NA, index=df.index)).isna()
        else:
            por_tipo = df.get("Permit Type", pd.Series(pd.NA, index=df.index)).eq(permit_type)
        estadisticas = estadisticas_diferencia(df, inicio, fin, por_tipo)
        if estadisticas is not None:
            mediana, _iqr, razon = estadisticas
            # Si hay suficientes observaciones del tipo, no se sustituye por
            # la estadística global cuando la dispersión del tipo es alta.
            return mediana if razon < MAX_IQR_MEDIANA_FECHAS else None

    estadisticas = estadisticas_diferencia(df, inicio, fin)
    if estadisticas is None:
        return None
    mediana, _iqr, razon = estadisticas
    return mediana if razon < MAX_IQR_MEDIANA_FECHAS else None


def candidatos_por_diferencia(
    df: pd.DataFrame,
    ancla: str,
    inicio_delta: str,
    fin_delta: str,
    factor: int,
) -> pd.Series:
    """Calcula fechas candidatas usando la mediana por tipo de permiso."""

    candidatos = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if ancla not in df.columns:
        return candidatos

    validos = df[ancla].notna()
    if not validos.any():
        return candidatos

    if "Permit Type" not in df.columns:
        delta = mediana_diferencia(df, inicio_delta, fin_delta)
        if delta is not None:
            candidatos.loc[validos] = df.loc[validos, ancla] + pd.to_timedelta(factor * delta, unit="D")
        return candidatos

    for permit_type, grupo in df.loc[validos].groupby("Permit Type", dropna=False):
        delta = mediana_diferencia(df, inicio_delta, fin_delta, permit_type)
        if delta is None:
            continue
        indices = grupo.index
        candidatos.loc[indices] = df.loc[indices, ancla] + pd.to_timedelta(factor * delta, unit="D")
    return candidatos


def limitar_fecha(df: pd.DataFrame, indice: Any, columna: str, candidata: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    """Ajusta una fecha imputada al intervalo cronológico disponible en la fila."""

    if pd.isna(candidata):
        return pd.NaT

    anteriores: list[str] = []
    posteriores: list[str] = []
    if columna in ORDEN_EVENTOS:
        posicion = ORDEN_EVENTOS.index(columna)
        anteriores = ORDEN_EVENTOS[:posicion]
        posteriores = ORDEN_EVENTOS[posicion + 1 :]
    elif columna == "Permit Expiration Date":
        # La expiración se calcula a partir de Issued Date y no se fuerza a
        # ser posterior a Completed Date, que puede ser una fecha posterior.
        anteriores = ["Issued Date"]

    valores_anteriores = [df.at[indice, c] for c in anteriores if c in df.columns and pd.notna(df.at[indice, c])]
    valores_posteriores = [df.at[indice, c] for c in posteriores if c in df.columns and pd.notna(df.at[indice, c])]

    limite_inferior = max(valores_anteriores) if valores_anteriores else None
    limite_superior = min(valores_posteriores) if valores_posteriores else None
    if limite_inferior is not None and limite_superior is not None and limite_inferior > limite_superior:
        return pd.NaT
    if limite_inferior is not None and candidata < limite_inferior:
        candidata = limite_inferior
    if limite_superior is not None and candidata > limite_superior:
        candidata = limite_superior
    return candidata


def completar_fecha_desde_referencia(
    df: pd.DataFrame,
    columna: str,
    referencias: list[tuple[str, str, str, int, str]],
    auditoria: list[dict[str, Any]],
    permitidas: pd.Series | None = None,
) -> None:
    """Imputa una fecha desde varias referencias, en orden de confiabilidad."""

    if columna not in df.columns:
        return

    for ancla, inicio_delta, fin_delta, factor, nombre in referencias:
        if ancla not in df.columns:
            continue
        candidatas = candidatos_por_diferencia(df, ancla, inicio_delta, fin_delta, factor)
        seleccion = df[columna].isna() & candidatas.notna()
        if permitidas is not None:
            seleccion &= permitidas.reindex(df.index, fill_value=False)

        cantidad = 0
        for indice in df.index[seleccion]:
            candidata = limitar_fecha(df, indice, columna, candidatas.at[indice])
            if pd.notna(candidata):
                df.at[indice, columna] = candidata
                cantidad += 1
        registrar(auditoria, nombre, columna, cantidad)


def completar_fechas(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Completa fechas usando medianas robustas por Permit Type."""

    estado = (
        df["Current Status"].astype("string").str.strip().str.casefold()
        if "Current Status" in df.columns
        else pd.Series("", index=df.index, dtype="string")
    )
    # El dataset usa "complete"; se conserva "completed" por compatibilidad
    # con la etiqueta descrita en las notas.
    completado = estado.isin(["complete", "completed"])
    no_aplica = estado.isin(["withdrawn", "cancelled", "canceled"])

    # Reglas directas indicadas en las notas.
    if "Completed Date" in df.columns and "Current Status Date" in df.columns:
        asignar_solo_nulos(
            df,
            "Completed Date",
            df["Current Status Date"],
            auditoria,
            "Current Status = completed -> Current Status Date",
            completado,
        )

    completar_fecha_desde_referencia(
        df,
        "Permit Creation Date",
        [("Filed Date", "Filed Date", "Permit Creation Date", 1, "mediana Filed -> Creation por Permit Type")],
        auditoria,
    )

    completar_fecha_desde_referencia(
        df,
        "Issued Date",
        [
            ("Filed Date", "Filed Date", "Issued Date", 1, "mediana Filed -> Issued por Permit Type"),
            ("Permit Creation Date", "Permit Creation Date", "Issued Date", 1, "mediana Creation -> Issued por Permit Type"),
            ("Current Status Date", "Issued Date", "Current Status Date", -1, "mediana Status -> Issued por Permit Type"),
        ],
        auditoria,
        ~no_aplica,
    )

    completar_fecha_desde_referencia(
        df,
        "First Construction Document Date",
        [
            ("Issued Date", "Issued Date", "First Construction Document Date", 1, "mediana Issued -> First Construction por Permit Type"),
            ("Completed Date", "First Construction Document Date", "Completed Date", -1, "mediana Completed -> First Construction por Permit Type"),
            ("Current Status Date", "First Construction Document Date", "Current Status Date", -1, "mediana Status -> First Construction por Permit Type"),
        ],
        auditoria,
        ~no_aplica,
    )

    # Completed Date se estima con la mediana por Permit Type para los
    # permisos no retirados ni cancelados, siempre que IQR/Mediana < 1.
    completar_fecha_desde_referencia(
        df,
        "Completed Date",
        [
            ("Issued Date", "Issued Date", "Completed Date", 1, "mediana Issued -> Completed por Permit Type"),
            ("First Construction Document Date", "First Construction Document Date", "Completed Date", 1, "mediana First Construction -> Completed por Permit Type"),
            ("Filed Date", "Filed Date", "Completed Date", 1, "mediana Filed -> Completed por Permit Type"),
        ],
        auditoria,
        ~no_aplica,
    )

    completar_fecha_desde_referencia(
        df,
        "Permit Expiration Date",
        [
            ("Issued Date", "Issued Date", "Permit Expiration Date", 1, "mediana Issued -> Expiration por Permit Type"),
            ("First Construction Document Date", "First Construction Document Date", "Permit Expiration Date", 1, "mediana First Construction -> Expiration por Permit Type"),
            ("Filed Date", "Filed Date", "Permit Expiration Date", 1, "mediana Filed -> Expiration por Permit Type"),
        ],
        auditoria,
        ~no_aplica,
    )


def grupo_descripcion(df: pd.DataFrame) -> pd.Series:
    """Clasifica Description en grupos útiles para las medianas de costos."""

    if "Description" not in df.columns:
        return pd.Series("sin_descripcion", index=df.index, dtype="string")

    texto = df["Description"].astype("string").fillna("").str.casefold()
    patrones = [
        ("reroofing", r"\breroof\w*\b|\bre-roof\w*\b"),
        ("bathroom", r"\bbathroom\w*\b"),
        ("kitchen", r"\bkitchen\w*\b"),
        ("fire", r"\bsprinkler\w*\b|fire\s+alarm|fire\s+suppression"),
        ("seismic", r"soft[- ]story|seismic\s+retrofit"),
        ("addition", r"\baddition\b|vertical\s+addition|horizontal\s+addition"),
        ("new_construction", r"new\s+(?:construction|building)"),
    ]
    resultado = pd.Series("other", index=df.index, dtype="string")
    for nombre, patron in patrones:
        resultado = resultado.mask(resultado.eq("other") & texto.str.contains(patron, regex=True, na=False), nombre)
    resultado = resultado.mask(texto.str.strip().eq(""), "sin_descripcion")
    return resultado


def completar_por_mediana(
    df: pd.DataFrame,
    columna: str,
    grupos: list[str],
    minimo: int,
    auditoria: list[dict[str, Any]],
    regla: str,
) -> None:
    """Completa una columna numérica usando mediana de grupos con evidencia."""

    if columna not in df.columns:
        return

    valores = pd.to_numeric(df[columna], errors="coerce")
    validos = valores.notna()
    if not validos.any():
        return

    trabajo = pd.DataFrame({"__valor": valores}, index=df.index)
    for grupo in grupos:
        trabajo[grupo] = df[grupo] if grupo in df.columns else pd.NA

    if grupos:
        agrupado = trabajo.loc[validos].groupby(grupos, dropna=False)["__valor"]
        medianas = agrupado.transform("median")
        cantidades = agrupado.transform("count")
        medianas = medianas.where(cantidades >= minimo)
    else:
        medianas = pd.Series(valores.loc[validos].median(), index=valores.loc[validos].index)

    candidatos = pd.Series(float("nan"), index=df.index)
    candidatos.loc[medianas.index] = medianas
    asignar_solo_nulos(df, columna, candidatos, auditoria, regla)


def copiar_costo(df: pd.DataFrame, origen: str, destino: str, auditoria: list[dict[str, Any]]) -> None:
    if origen not in df.columns or destino not in df.columns:
        return
    origen_numerico = pd.to_numeric(df[origen], errors="coerce")
    asignar_solo_nulos(
        df,
        destino,
        origen_numerico,
        auditoria,
        f"{origen} -> {destino} como costo relacionado",
    )


def completar_costos(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Completa costos por relación directa y medianas de grupos."""

    copiar_costo(df, "Estimated Cost", "Revised Cost", auditoria)
    copiar_costo(df, "Revised Cost", "Estimated Cost", auditoria)

    trabajo = df.copy()
    trabajo["__description_group"] = grupo_descripcion(df)
    if "Proposed Units" in df.columns:
        trabajo["__proposed_units_group"] = pd.to_numeric(df["Proposed Units"], errors="coerce").round().astype("string")

    grupos = []
    if "Proposed Units" in df.columns:
        grupos.append(["Permit Type", "__description_group", "__proposed_units_group"])
    grupos.extend(
        [
            ["Permit Type", "__description_group"],
            ["Permit Type"],
            [],
        ]
    )

    # Se agregan las columnas temporales solo durante el cálculo para no
    # alterar el esquema original del dataset de salida.
    temporales = [columna for columna in trabajo.columns if columna.startswith("__") and columna not in df.columns]
    for columna in temporales:
        df[columna] = trabajo[columna]

    for columna in ["Estimated Cost", "Revised Cost"]:
        for grupos_actuales in grupos:
            grupos_df = [grupo for grupo in grupos_actuales if grupo in df.columns]
            completar_por_mediana(
                df,
                columna,
                grupos_df,
                3 if grupos_df else 10,
                auditoria,
                f"mediana de {columna} por " + (" + ".join(grupos_df) if grupos_df else "dataset"),
            )

    if temporales:
        df.drop(columns=temporales, inplace=True)

    copiar_costo(df, "Estimated Cost", "Revised Cost", auditoria)
    copiar_costo(df, "Revised Cost", "Estimated Cost", auditoria)


def completar_banderas_por_descripcion(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Marca como afirmativos los permisos inferibles desde Description."""

    if "Description" not in df.columns:
        return
    texto = df["Description"].astype("string").fillna("")
    reglas = [
        (
            "Fire Only Permit",
            r"\bsprinkler\w*\b|fire\s+alarm|fire\s+suppression|fire[- ]only",
            "Description -> Fire Only Permit = Y",
        ),
        (
            "Voluntary Soft-Story Retrofit",
            r"soft[- ]story|seismic\s+retrofit",
            "Description -> Voluntary Soft-Story Retrofit = Y",
        ),
    ]
    for columna, patron, regla in reglas:
        if columna not in df.columns:
            continue
        candidatos = pd.Series(pd.NA, index=df.index, dtype="string")
        candidatos.loc[texto.str.contains(patron, case=False, regex=True, na=False)] = "Y"
        asignar_solo_nulos(df, columna, candidatos, auditoria, regla)


def completar_plansets(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Imputa Plansets solo con una mediana por tipo de permiso."""

    if "Plansets" in df.columns and "Permit Type" in df.columns:
        completar_por_mediana(
            df,
            "Plansets",
            ["Permit Type"],
            5,
            auditoria,
            "mediana de Plansets por Permit Type",
        )


def filtrar_por_porcentaje_nulos(df: pd.DataFrame, auditoria: list[dict[str, Any]]) -> None:
    """Elimina filas y columnas según su porcentaje de valores nulos."""

    if df.empty:
        return

    porcentajes = df.isna().mean()
    columnas_para_filas = [
        columna
        for columna in df.columns
        if porcentajes[columna] < UMBRAL_NULOS_FILAS
    ]
    columnas_para_eliminar = [
        columna
        for columna in df.columns
        if porcentajes[columna] >= UMBRAL_NULOS_COLUMNAS
    ]

    if columnas_para_filas:
        filas_validas = df[columnas_para_filas].notna().all(axis=1)
        filas_eliminadas = int((~filas_validas).sum())
        if filas_eliminadas:
            df.drop(index=df.index[~filas_validas], inplace=True)
            auditoria.append(
                {
                    "tipo": "filtro",
                    "regla": f"eliminar tuplas con nulos en columnas con menos de {UMBRAL_NULOS_FILAS:.0%} de nulos",
                    "columna": ", ".join(columnas_para_filas),
                    "cantidad": filas_eliminadas,
                }
            )

    if columnas_para_eliminar:
        df.drop(columns=columnas_para_eliminar, inplace=True)
        auditoria.append(
            {
                "tipo": "filtro",
                "regla": f"eliminar columnas con al menos {UMBRAL_NULOS_COLUMNAS:.0%} de nulos",
                "columna": ", ".join(columnas_para_eliminar),
                "cantidad": len(columnas_para_eliminar),
            }
        )


def resumen_auditoria(original: pd.DataFrame, resultado: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in original.columns:
        antes = int(original[columna].isna().sum())
        despues = int(resultado[columna].isna().sum())
        filas.append(
            {
                "tipo": "resumen",
                "regla": "faltantes antes/después de imputar (antes del filtro)",
                "columna": columna,
                "faltantes_antes": antes,
                "faltantes_despues": despues,
                "imputados": antes - despues,
            }
        )
    return pd.DataFrame(filas)


def imputar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta todas las reglas y devuelve dataset imputado y auditoría."""

    resultado = df.copy()
    normalizar_nulos(resultado)
    convertir_fechas(resultado)
    original = resultado.copy()
    auditoria: list[dict[str, Any]] = []

    completar_par_codigo_descripcion(resultado, "Permit Type", "Permit Type Definition", auditoria)
    completar_par_codigo_descripcion(
        resultado,
        "Existing Construction Type",
        "Existing Construction Type Description",
        auditoria,
    )
    completar_par_codigo_descripcion(
        resultado,
        "Proposed Construction Type",
        "Proposed Construction Type Description",
        auditoria,
    )
    completar_por_ubicacion(resultado, auditoria)
    completar_inercia_existente_propuesto(resultado, auditoria)
    completar_pares_correlacionados(resultado, auditoria)
    completar_fechas(resultado, auditoria)
    completar_costos(resultado, auditoria)
    completar_banderas_por_descripcion(resultado, auditoria)
    completar_plansets(resultado, auditoria)

    resumen = resumen_auditoria(original, resultado)
    filtrar_por_porcentaje_nulos(resultado, auditoria)
    detalle = pd.DataFrame(auditoria)
    if detalle.empty:
        detalle = pd.DataFrame(columns=["tipo", "regla", "columna", "cantidad"])
    return resultado, pd.concat([detalle, resumen], ignore_index=True, sort=False)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV local. Si se omite, se descarga/carga con KaggleHub.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Building_Permits_imputed.csv"),
        help="Ruta del CSV imputado.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("Building_Permits_imputation_audit.csv"),
        help="Ruta del CSV de auditoría.",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    df = cargar_dataset(args.input)
    resultado, auditoria = imputar(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    auditoria.to_csv(args.audit, index=False, encoding="utf-8-sig")

    resumen = auditoria.loc[auditoria["tipo"].eq("resumen")]
    total = int(resumen["imputados"].fillna(0).sum())
    print(f"Dataset imputado: {args.output}")
    print(f"Auditoría: {args.audit}")
    print(f"Celdas imputadas: {total:,}")
    print(f"Tuplas del dataset resultante: {len(resultado):,}")

    porcentaje_nulos = (
        resultado.isna()
        .mean()
        .mul(100)
        .loc[lambda serie: serie.gt(0)]
        .sort_values(ascending=False)
    )
    print("\nPorcentaje de nulos por columna:")
    if porcentaje_nulos.empty:
        print("No quedan valores nulos.")
    else:
        reporte_nulos = porcentaje_nulos.rename("porcentaje_nulos").to_frame()
        print(
            reporte_nulos.to_string(
                header=["Porcentaje"],
                float_format=lambda valor: f"{valor:.2f}%",
            )
        )


if __name__ == "__main__":
    main()
