"""
===========================================================
 NORMALIZACIÓN DE DATOS - MEN SHOES
===========================================================

Entrada:
    MEN_SHOES(1).csv

Salida:
    MEN_SHOES_normalized.csv

Procesos:
    1. Limpieza de nombres de columnas
    2. Normalización de texto
    3. Normalización de marcas
    4. Conversión de variables numéricas
    5. Eliminación de duplicados
    6. Limpieza de Product_details
    7. Extracción de características
    8. Extracción de adjetivos significativos
    9. Creación de variables binarias
===========================================================
"""

import re
import unicodedata

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURACIÓN
# =========================================================

INPUT_FILE = "MEN_SHOES.csv"  # Asegúrate de que coincida con el nombre de tu archivo
OUTPUT_FILE = "MEN_SHOES_NORMALIZED.csv"


# =========================================================
# CARACTERÍSTICAS DEL PRODUCTO
# =========================================================

FEATURE_DICTIONARY = {

    # Materiales
    "leather": [
        "leather",
        "genuine leather",
        "synthetic leather",
        "faux leather"
    ],

    "mesh": [
        "mesh"
    ],

    "canvas": [
        "canvas"
    ],

    "suede": [
        "suede"
    ],

    # Comodidad
    "comfortable": [
        "comfortable",
        "comfort"
    ],

    "lightweight": [
        "lightweight",
        "light weight"
    ],

    "breathable": [
        "breathable"
    ],

    "cushioned": [
        "cushioned",
        "cushion"
    ],

    "soft": [
        "soft"
    ],

    # Protección
    "waterproof": [
        "waterproof",
        "water proof"
    ],

    "slip_resistant": [
        "slip resistant",
        "non slip",
        "anti slip"
    ],

    "shock_absorbing": [
        "shock absorbing",
        "shock absorption"
    ],

    # Actividad
    "running": [
        "running",
        "runner"
    ],

    "walking": [
        "walking"
    ],

    "training": [
        "training",
        "trainer"
    ],

    "gym": [
        "gym",
        "workout"
    ],

    "sports": [
        "sports",
        "sport"
    ],

    "hiking": [
        "hiking"
    ],

    # Estilo
    "casual": [
        "casual"
    ],

    "stylish": [
        "stylish"
    ],

    "fashionable": [
        "fashionable",
        "fashion"
    ],

    "classic": [
        "classic"
    ],

    "modern": [
        "modern"
    ],

    "retro": [
        "retro"
    ],

    # Construcción
    "lace_up": [
        "lace up",
        "lace-up",
        "laces"
    ],

    "slip_on": [
        "slip on",
        "slip-on"
    ],

    "high_top": [
        "high top",
        "high-top"
    ],

    "low_top": [
        "low top",
        "low-top"
    ],

    # Uso
    "outdoor": [
        "outdoor"
    ],

    "indoor": [
        "indoor"
    ],

    "daily_use": [
        "daily",
        "everyday",
        "daily wear",
        "everyday wear"
    ],

    # Características comerciales
    "new": [
        "new",
        "latest"
    ],

    "premium": [
        "premium"
    ],

    "durable": [
        "durable",
        "durability"
    ]
}


# =========================================================
# ADJETIVOS SIGNIFICATIVOS
# =========================================================

SIGNIFICANT_ADJECTIVES = {
    "waterproof",
    "lightweight",
    "comfortable",
    "breathable",
    "cushioned",
    "soft",
    "stylish",
    "casual",
    "classic",
    "modern",
    "retro",
    "durable",
    "premium",
    "fashionable",
    "new",
    "latest",
    "sports",
    "running",
    "walking",
    "training",
    "outdoor",
    "indoor",
    "leather",
    "mesh",
    "canvas",
    "suede"
}


# =========================================================
# NORMALIZACIÓN DE TEXTO
# =========================================================

def normalizar_texto(texto):

    if pd.isna(texto):
        return ""

    texto = str(texto)

    texto = texto.lower().strip()

    # Eliminar acentos
    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    # Normalizar guiones
    texto = re.sub(
        r"[-–—]+",
        "-",
        texto
    )

    # Normalizar espacios
    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


# =========================================================
# NORMALIZACIÓN DE MARCA
# =========================================================

def normalizar_marca(marca):

    if pd.isna(marca):
        return "unknown"

    marca = normalizar_texto(marca)

    return marca if marca else "unknown"


# =========================================================
# NORMALIZACIÓN DE PRECIO
# =========================================================

def convertir_precio(valor):

    if pd.isna(valor):
        return np.nan

    valor = str(valor)

    # Eliminar símbolos monetarios
    valor = re.sub(
        r"[^0-9.\-]",
        "",
        valor
    )

    if not valor:
        return np.nan

    try:
        return float(valor)

    except ValueError:
        return np.nan


# =========================================================
# NORMALIZACIÓN DE VENTAS
# =========================================================

def convertir_ventas(valor):

    if pd.isna(valor):
        return np.nan

    valor = str(valor)

    valor = re.sub(
        r"[^0-9]",
        "",
        valor
    )

    if not valor:
        return np.nan

    try:
        return int(valor)

    except ValueError:
        return np.nan


# =========================================================
# NORMALIZACIÓN DE RATING
# =========================================================

def convertir_rating(valor):

    if pd.isna(valor):
        return np.nan

    valor = str(valor)

    valor = re.sub(
        r"[^0-9.]",
        "",
        valor
    )

    if not valor:
        return np.nan

    try:
        return float(valor)

    except ValueError:
        return np.nan


# =========================================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# =========================================================

def extraer_caracteristicas(texto):

    texto = normalizar_texto(texto)

    caracteristicas = []

    for nombre, patrones in FEATURE_DICTIONARY.items():

        for patron in patrones:

            patron = normalizar_texto(patron)

            # Buscar palabras o frases completas
            patron_regex = (
                r"(?<!\w)"
                + re.escape(patron)
                + r"(?!\w)"
            )

            if re.search(
                patron_regex,
                texto
            ):
                caracteristicas.append(nombre)
                break

    return caracteristicas


# =========================================================
# EXTRACCIÓN DE ADJETIVOS
# =========================================================

def extraer_adjetivos(texto):

    texto = normalizar_texto(texto)

    palabras = re.findall(
        r"[a-z]+",
        texto
    )

    encontrados = sorted(
        {
            palabra
            for palabra in palabras
            if palabra in SIGNIFICANT_ADJECTIVES
        }
    )

    return encontrados


# =========================================================
# CARGAR DATASET
# =========================================================

def cargar_dataset():

    print("Cargando dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Registros originales: {len(df):,}"
    )

    return df


# =========================================================
# LIMPIAR COLUMNAS
# =========================================================

def limpiar_nombres_columnas(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )

    return df


# =========================================================
# NORMALIZAR DATASET
# =========================================================

def normalizar_dataset(df):

    # -----------------------------------------------------
    # Marca
    # -----------------------------------------------------

    if "brand" in df.columns:

        df["brand"] = (
            df["brand"]
            .apply(normalizar_marca)
        )

    # -----------------------------------------------------
    # Precio
    # -----------------------------------------------------

    if "current_price" in df.columns:

        df["current_price"] = (
            df["current_price"]
            .apply(convertir_precio)
        )

    # -----------------------------------------------------
    # Ventas
    # -----------------------------------------------------

    if "how_many_sold" in df.columns:

        df["how_many_sold"] = (
            df["how_many_sold"]
            .apply(convertir_ventas)
        )

    # -----------------------------------------------------
    # Rating
    # -----------------------------------------------------

    if "rating" in df.columns:

        df["rating"] = (
            df["rating"]
            .apply(convertir_rating)
        )

    # -----------------------------------------------------
    # Product details
    # -----------------------------------------------------

    if "product_details" in df.columns:

        df["product_details"] = (
            df["product_details"]
            .apply(normalizar_texto)
        )

        # Crear texto limpio adicional
        df["product_details_clean"] = (
            df["product_details"]
            .fillna("")
        )

        # Extraer características
        df["features"] = (
            df["product_details_clean"]
            .apply(
                extraer_caracteristicas
            )
        )

        # Extraer adjetivos
        df["significant_adjectives"] = (
            df["product_details_clean"]
            .apply(
                extraer_adjetivos
            )
        )

        # Número de características
        df["feature_count"] = (
            df["features"]
            .apply(len)
        )

        # -------------------------------------------------
        # Variables binarias
        # -------------------------------------------------

        for caracteristica in FEATURE_DICTIONARY:

            df[
                f"is_{caracteristica}"
            ] = (
                df["features"]
                .apply(
                    lambda lista:
                    int(
                        caracteristica
                        in lista
                    )
                )
            )

        # -------------------------------------------------
        # Convertir listas a strings para permitir drop_duplicates
        # -------------------------------------------------
        df["features"] = df["features"].apply(lambda x: ", ".join(x))
        df["significant_adjectives"] = df["significant_adjectives"].apply(lambda x: ", ".join(x))

    return df


# =========================================================
# ELIMINACIÓN DE DUPLICADOS
# =========================================================

def eliminar_duplicados(df):

    antes = len(df)

    # Duplicados exactos
    df = df.drop_duplicates()

    eliminados = antes - len(df)

    print(
        f"Duplicados eliminados: {eliminados:,}"
    )

    return df


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():

    # 1. Cargar
    df = cargar_dataset()

    # 2. Limpiar nombres de columnas
    df = limpiar_nombres_columnas(df)

    # 3. Normalizar datos
    df = normalizar_dataset(df)

    # 4. Eliminar duplicados
    df = eliminar_duplicados(df)

    # 5. Guardar CSV
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("NORMALIZACIÓN COMPLETADA")
    print("=" * 60)
    print(
        f"Registros finales: {len(df):,}"
    )
    print(
        f"Columnas finales: {len(df.columns):,}"
    )
    print(
        f"Archivo generado: {OUTPUT_FILE}"
    )
    print("=" * 60)


# =========================================================
# EJECUCIÓN
# =========================================================

if __name__ == "__main__":
    main()