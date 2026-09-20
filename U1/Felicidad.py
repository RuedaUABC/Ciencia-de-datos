from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["savefig.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.titlecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["legend.labelcolor"] = "white"


def aplicar_fondo_transparente(figura):
    """Hace que el fondo de la figura sea transparente cuando se guarde o muestre."""
    figura.patch.set_alpha(0)
    for eje in figura.axes:
        eje.patch.set_alpha(0)
        eje.set_facecolor("none")


RUTA_DATASET = (
    Path(__file__).resolve().parent.parent
    / "Datasets"
    / "WHR26_Data_Figure_2.1.csv"
)
df = pd.read_csv(RUTA_DATASET)

# Limitar el análisis a los años solicitados.
df_periodo = df[df["Year"].between(2019, 2025)].copy()

# Usar las variables numéricas, excluyendo los límites del whisker.
columnas_omitir = [
    "Lower whisker",
    "Upper whisker",
    "Year",
    "Rank",
]


def mostrar_matriz_correlacion(datos, titulo, metodo="pearson"):
    columnas_numericas = datos.select_dtypes(include="number").columns
    datos_matriz = datos[columnas_numericas].drop(columns=columnas_omitir)
    correlaciones = datos_matriz.corr(method=metodo)
    print(f"\n=== Correlación {metodo.upper()} - {titulo} ===")
    print(correlaciones.round(3).to_string())
    figura, eje = plt.subplots(figsize=(12, 10))
    imagen = eje.imshow(
        correlaciones,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    eje.set_xticks(range(len(correlaciones.columns)))
    eje.set_yticks(range(len(correlaciones.index)))
    eje.set_xticklabels(correlaciones.columns, rotation=45, ha="right")
    eje.set_yticklabels(correlaciones.index)
    eje.set_title(
        f"Matriz de correlación {metodo.capitalize()} de indicadores de felicidad: {titulo}"
    )

    for fila in range(len(correlaciones.index)):
        for columna in range(len(correlaciones.columns)):
            eje.text(
                columna,
                fila,
                f"{correlaciones.iloc[fila, columna]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    figura.colorbar(imagen, ax=eje, label="Coeficiente de correlación")
    figura.tight_layout()
    aplicar_fondo_transparente(figura)
    plt.show()


def mostrar_matrices(datos, titulo):
    columnas_numericas = datos.select_dtypes(include="number").columns
    datos_matriz = datos[columnas_numericas].drop(columns=columnas_omitir)

    pd.plotting.scatter_matrix(
        datos_matriz,
        figsize=(18, 18),
        diagonal="hist",
        alpha=0.6,
        grid=True,
    )
    plt.suptitle(f"Matriz de dispersión de indicadores de felicidad: {titulo}", y=0.9)
    aplicar_fondo_transparente(plt.gcf())
    plt.show()

    mostrar_matriz_correlacion(datos, titulo)


# Definir variables a analizar
variables_correlacion = [
    "Life evaluation (3-year average)",
    "Explained by: Log GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
]
variables_diagrama_lineas = [
    "Explained by: Log GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
    "Explained by: Generosity",
    "Explained by: Perceptions of corruption",
]
nombre_variable = {
    "Life evaluation (3-year average)": "Life evaluation",
    "Explained by: Log GDP per capita": "Log GDP per capita",
    "Explained by: Social support": "Social support",
    "Explained by: Healthy life expectancy": "Healthy life expectancy",
    "Explained by: Freedom to make life choices": "Freedom to make life choices",
    "Explained by: Generosity": "Generosity",
    "Explained by: Perceptions of corruption": "Perceptions of corruption",
    "Dystopia + residual": "Dystopia + residual",
}


def mostrar_diagrama_lineas(datos, titulo):
    """Muestra un diagrama de líneas con la evolución de variables por año."""
    promedios_por_año = datos.groupby("Year")[variables_diagrama_lineas].median()
    promedios_por_año = promedios_por_año.rename(columns=nombre_variable)

    plt.figure(figsize=(14, 8))
    for variable in promedios_por_año.columns:
        plt.plot(
            promedios_por_año.index,
            promedios_por_año[variable],
            marker="o",
            linewidth=2,
            label=variable,
        )

    plt.title(f"Evolución anual de variables explicativas: {titulo}")
    plt.xlabel("Año")
    plt.ylabel("Valor promedio")
    plt.xticks(sorted(datos["Year"].unique()))
    plt.ylim(0, 2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    aplicar_fondo_transparente(plt.gcf())
    plt.show()


def mostrar_diagrama_lineas_life_evaluation(datos, titulo):
    """Muestra la evolución de la variable Life evaluation por año."""
    promedios_vida = (
        datos.groupby("Year")["Life evaluation (3-year average)"].median().rename("Life evaluation")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        promedios_vida.index,
        promedios_vida.values,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="Life evaluation",
    )

    plt.title(f"Evolución anual de Life evaluation: {titulo}")
    plt.xlabel("Año")
    plt.ylabel("Life evaluation (3-year average)")
    plt.xticks(sorted(datos["Year"].unique()))
    minimo = promedios_vida.min() * 0.95 if len(promedios_vida) > 0 else 0
    maximo = promedios_vida.max() * 1.05 if len(promedios_vida) > 0 else 1
    plt.ylim(minimo, maximo)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    aplicar_fondo_transparente(plt.gcf())
    plt.show()


# Matrices para todos los países.
mostrar_matrices(df_periodo, "todos los países (2019-2025)")
mostrar_matriz_correlacion(
    df_periodo,
    "todos los países (2019-2025)",
    metodo="spearman",
)

# Las mismas matrices, pero únicamente para México.
df_mexico = df_periodo[df_periodo["Country name"].str.strip() == "Mexico"].copy()
mostrar_matrices(df_mexico, "México (2019-2025)")
mostrar_diagrama_lineas(df_mexico, "México (2019-2025)")

# Diagrama de líneas de promedios globales (2019-2025) con todas las variables excepto Life evaluation y Dystopia
mostrar_diagrama_lineas(df_periodo, "todos los países (2019-2025)")

# Diagrama de líneas exclusivo para Life evaluation (2019-2025)
mostrar_diagrama_lineas_life_evaluation(df_periodo, "todos los países (2019-2025)")

# Diagrama de líneas de Costa Rica (2019-2025)
df_costarica = df_periodo[df_periodo["Country name"].str.strip() == "Costa Rica"].copy()
if len(df_costarica) > 0:
    mostrar_diagrama_lineas(df_costarica, "Costa Rica (2019-2025)")
else:
    print("No hay datos para Costa Rica en el período 2019-2025")

# Diagrama de líneas de Ucrania (2019-2025)
df_ucrania = df_periodo[df_periodo["Country name"].str.strip() == "Ukraine"].copy()
if len(df_ucrania) > 0:
    mostrar_diagrama_lineas(df_ucrania, "Ucrania (2019-2025)")
else:
    print("No hay datos para Ucrania en el período 2019-2025")

# Diagrama de líneas de Rusia (2019-2025)
df_rusia = df_periodo[df_periodo["Country name"].str.strip() == "Russian Federation"].copy()
if len(df_rusia) > 0:
    mostrar_diagrama_lineas(df_rusia, "Rusia (2019-2025)")
else:
    print("No hay datos para Rusia en el período 2019-2025")

# Seguir por año los valores promedio de las variables seleccionadas.

# Comparar los 20 países mejor y peor posicionados en 2024.
df_2024 = df[df["Year"] == 2024].copy()
top_20 = df_2024.nsmallest(20, "Rank")
bottom_20 = df_2024.nlargest(20, "Rank")
combinados = pd.concat([top_20, bottom_20])

columnas_factores = [
    "Explained by: Log GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
    "Explained by: Generosity",
    "Explained by: Perceptions of corruption",
    "Dystopia + residual",
]

datos_apilados = combinados.sort_values("Rank").copy()
datos_apilados[columnas_factores] = datos_apilados[columnas_factores].fillna(0)
datos_apilados["Country name"] = (
    datos_apilados["Country name"].str.strip()
    + " ("
    + datos_apilados["Life evaluation (3-year average)"].map("{:.2f}".format)
    + ")"
)

figura_factores, eje_factores = plt.subplots(figsize=(14, 12))
posiciones = range(len(datos_apilados))
acumulado = pd.Series(0.0, index=datos_apilados.index)
colores = plt.get_cmap("tab10").colors

for indice, factor in enumerate(columnas_factores):
    nombre_factor = factor.replace("Explained by: ", "")
    valores = datos_apilados[factor]
    eje_factores.barh(
        posiciones,
        valores,
        left=acumulado,
        color=colores[indice],
        label=nombre_factor,
    )
    acumulado = acumulado + valores

eje_factores.set_yticks(list(posiciones))
eje_factores.set_yticklabels(datos_apilados["Country name"])
eje_factores.invert_yaxis()
eje_factores.set_xlabel("Contribución a la evaluación de vida")
eje_factores.set_ylabel("")
eje_factores.set_title(
    "Factores de evaluación: 20 mejores y 20 peores países (2024)"
)
eje_factores.legend(title="Factor", bbox_to_anchor=(1.02, 1), loc="upper left")
eje_factores.grid(axis="x", alpha=0.3)
figura_factores.tight_layout()
aplicar_fondo_transparente(figura_factores)
plt.show()

# Comparar la evaluación de vida de Rusia y Ucrania por año.
paises_comparar = ["Russian Federation", "Ukraine"]
df_rusia_ucrania = df_periodo[
    df_periodo["Country name"].str.strip().isin(paises_comparar)
].copy()
df_rusia_ucrania = df_rusia_ucrania.sort_values("Year")

plt.figure(figsize=(10, 6))
for pais in paises_comparar:
    datos_pais = df_rusia_ucrania[
        df_rusia_ucrania["Country name"].str.strip() == pais
    ].dropna(subset=["Life evaluation (3-year average)"])
    plt.plot(
        datos_pais["Year"],
        datos_pais["Life evaluation (3-year average)"],
        marker="o",
        linewidth=2,
        label=pais,
    )

plt.title("Evaluación de vida: Rusia y Ucrania (2019-2025)")
plt.xlabel("Año")
plt.ylabel("Life evaluation (3-year average)")
plt.xticks(sorted(df_rusia_ucrania["Year"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
aplicar_fondo_transparente(plt.gcf())
plt.show()
