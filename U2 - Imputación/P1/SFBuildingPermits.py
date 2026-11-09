import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter


file_path = "Building_Permits.csv"
columnas_omitir = [
  "Voluntary Soft-Story Retrofit",
  "TIDF Compliance",
  "Street Number Suffix",
  "Unit Suffix",
  "Site Permit",
  "Structural Notification",
  "Fire Only Permit",
  "Unit",
]

# Cargar la version mas reciente del conjunto de datos.
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "aparnashastry/building-permit-applications-data",
  file_path,
  pandas_kwargs={
    "encoding": "latin1",
    "compression": "zip",
    "low_memory": False,
  },
)

def obtener_moda(serie):
  modas = serie.mode()
  return modas.iloc[0] if not modas.empty else float("nan")


# Calcular la correlacion entre las variables numericas.
df_correlacion = (
  df.drop(columns=columnas_omitir, errors="ignore")
    .select_dtypes(include="number")
)
matriz_correlacion = df_correlacion.corr()
resultados_correlacion = []

for i, variable_1 in enumerate(matriz_correlacion.columns):
  for variable_2 in matriz_correlacion.columns[i + 1:]:
    correlacion = matriz_correlacion.loc[variable_1, variable_2]
    if correlacion > 0.8:
      nulo_1 = df[variable_1].isna()
      nulo_2 = df[variable_2].isna()
      datos_validos = df.loc[
        ~nulo_1 & ~nulo_2,
        [variable_1, variable_2],
      ]

      desviacion_estandar = (
        f"{datos_validos[variable_1].std():.4f} / "
        f"{datos_validos[variable_2].std():.4f}"
      )
      minimo = (
        f"{datos_validos[variable_1].min():.4f} / "
        f"{datos_validos[variable_2].min():.4f}"
      )
      maximo = (
        f"{datos_validos[variable_1].max():.4f} / "
        f"{datos_validos[variable_2].max():.4f}"
      )
      mediana = (
        f"{datos_validos[variable_1].median():.4f} / "
        f"{datos_validos[variable_2].median():.4f}"
      )
      moda = (
        f"{obtener_moda(datos_validos[variable_1]):.4f} / "
        f"{obtener_moda(datos_validos[variable_2]):.4f}"
      )

      solo_variable_1_nula = (nulo_1 & ~nulo_2).sum()
      solo_variable_2_nula = (~nulo_1 & nulo_2).sum()
      ambas_nulas = (nulo_1 & nulo_2).sum()

      resultados_correlacion.append(
        (
          variable_1,
          variable_2,
          desviacion_estandar,
          minimo,
          maximo,
          mediana,
          moda,
          solo_variable_1_nula,
          solo_variable_2_nula,
          ambas_nulas,
        )
      )

print("| Variable 1 | Variable 2 | Desviacion estandar (V1 / V2) | Min (V1 / V2) | Max (V1 / V2) | Mediana (V1 / V2) | Moda (V1 / V2) | Solo V1 nula | Solo V2 nula | Ambas nulas |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for resultado in resultados_correlacion:
  (
    variable_1,
    variable_2,
    desviacion_estandar,
    minimo,
    maximo,
    mediana,
    moda,
    solo_variable_1_nula,
    solo_variable_2_nula,
    ambas_nulas,
  ) = resultado
  print(
    f"| {variable_1} | {variable_2} | {desviacion_estandar} | "
    f"{minimo} | {maximo} | {mediana} | {moda} | "
    f"{solo_variable_1_nula} | {solo_variable_2_nula} | {ambas_nulas} |"
  )

columnas_fechas = [
  "Current Status Date",
  "Filed Date",
  "Issued Date",
  "Completed Date",
  "First Construction Document Date",
  "Permit Expiration Date",
]
df_fechas = df.reindex(columns=["Permit Type", *columnas_fechas]).copy()
df_fechas["Permit Type"] = (
  df_fechas["Permit Type"].astype("string").fillna("Sin dato")
)
df_fechas[columnas_fechas] = df_fechas[columnas_fechas].apply(
  pd.to_datetime,
  errors="coerce",
)
resultados_correlacion_fechas = []

for i, fecha_1 in enumerate(columnas_fechas):
  for fecha_2 in columnas_fechas[i + 1:]:
    nulo_1 = df_fechas[fecha_1].isna()
    nulo_2 = df_fechas[fecha_2].isna()

    solo_fecha_1_nula = (nulo_1 & ~nulo_2).sum()
    solo_fecha_2_nula = (~nulo_1 & nulo_2).sum()
    ambas_nulas = (nulo_1 & nulo_2).sum()

    # Distancia absoluta en dias para la estadistica y los boxplots.
    distancias = (
      (df_fechas[fecha_2] - df_fechas[fecha_1])
        .abs()
        .dt.total_seconds()
        .div(86400)
        .dropna()
    )
    rango_intercuartilico = (
      distancias.quantile(0.75) - distancias.quantile(0.25)
    )
    mediana = distancias.median()
    if pd.isna(mediana):
      iqr_mediana = float("nan")
    elif mediana != 0:
      iqr_mediana = rango_intercuartilico / abs(mediana)
    else:
      iqr_mediana = (
        0 if rango_intercuartilico == 0 else float("inf")
      )

    resultados_correlacion_fechas.append(
      (
        fecha_1,
        fecha_2,
        rango_intercuartilico,
        mediana,
        iqr_mediana,
        solo_fecha_1_nula,
        solo_fecha_2_nula,
        ambas_nulas,
      )
    )

print("\n| Fecha 1 | Fecha 2 | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana | Solo Fecha 1 nula | Solo Fecha 2 nula | Ambas nulas |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for resultado in resultados_correlacion_fechas:
  (
    fecha_1,
    fecha_2,
    rango_intercuartilico,
    mediana,
    iqr_mediana,
    solo_fecha_1_nula,
    solo_fecha_2_nula,
    ambas_nulas,
  ) = resultado
  print(
    f"| {fecha_1} | {fecha_2} | {rango_intercuartilico:.4f} | "
    f"{mediana:.4f} | {iqr_mediana:.4f} | "
    f"{solo_fecha_1_nula} | {solo_fecha_2_nula} | {ambas_nulas} |"
  )

# Crear una tabla independiente con las estadisticas de fechas de cada permiso.
def formatear_estadistica(valor):
  return "Sin dato" if pd.isna(valor) else f"{valor:.4f}"


print("\n## Fechas por Permit Type")
distancias_boxplots_por_tipo = {}
for permit_type, datos_tipo in df_fechas.groupby("Permit Type", sort=True):
  print(f"\n### Permit Type: {permit_type}")
  print(
    "| Fecha 1 | Fecha 2 | Rango intercuartilico (dias) | "
    "Mediana (dias) | IQR / Mediana |"
  )
  print("|---|---|---:|---:|---:|")
  distancias_seleccionadas = []

  for i, fecha_1 in enumerate(columnas_fechas):
    for fecha_2 in columnas_fechas[i + 1:]:
      distancias = (
        (datos_tipo[fecha_2] - datos_tipo[fecha_1])
          .abs()
          .dt.total_seconds()
          .div(86400)
          .dropna()
      )
      rango_intercuartilico = (
        distancias.quantile(0.75) - distancias.quantile(0.25)
      )
      mediana = distancias.median()
      if pd.isna(mediana):
        iqr_mediana = float("nan")
      elif mediana != 0:
        iqr_mediana = rango_intercuartilico / abs(mediana)
      else:
        iqr_mediana = (
          0 if rango_intercuartilico == 0 else float("inf")
        )

      if (
        not distancias.empty
        and (iqr_mediana < 1 or iqr_mediana == float("inf"))
      ):
        distancias_seleccionadas.append((fecha_1, fecha_2, distancias))

      print(
        f"| {fecha_1} | {fecha_2} | "
        f"{formatear_estadistica(rango_intercuartilico)} | "
        f"{formatear_estadistica(mediana)} | "
        f"{formatear_estadistica(iqr_mediana)} |"
      )

  distancias_boxplots_por_tipo[permit_type] = distancias_seleccionadas

# Mostrar un boxplot independiente por Permit Type.
figuras_creadas = 0
for permit_type, pares_de_fecha in distancias_boxplots_por_tipo.items():
  if not pares_de_fecha:
    continue

  figura, eje = plt.subplots(figsize=(20, 9))
  datos_boxplot = [distancias for _, _, distancias in pares_de_fecha]
  etiquetas = [
    f"{fecha_1}\nvs\n{fecha_2}"
    for fecha_1, fecha_2, _ in pares_de_fecha
  ]
  eje.boxplot(
    datos_boxplot,
    patch_artist=True,
    boxprops={"facecolor": "steelblue", "alpha": 0.7},
    medianprops={"color": "darkred", "linewidth": 2},
    whiskerprops={"color": "black"},
    capprops={"color": "black"},
  )
  eje.set_title(
    f"Permit Type {permit_type}: IQR / Mediana < 1 o inf",
    fontsize=16,
  )
  eje.set_ylabel("Distancia absoluta (dias)")
  eje.set_xticks(range(1, len(etiquetas) + 1))
  eje.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
  eje.grid(axis="y", linestyle="--", alpha=0.5)
  figura.tight_layout()
  figuras_creadas += 1

if figuras_creadas:
  plt.show()
