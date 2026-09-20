import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Importar el dataset
df = pd.read_csv("Datasets/iris.csv")

# 2. Definir las características numéricas
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 3. Crear figura con 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 4. Generar un boxplot por cada característica
for ax, feature in zip(axes.flatten(), features):
    df.boxplot(column=feature, by="species", ax=ax)
    ax.set_title(f"Distribución de {feature} por especie")
    ax.set_xlabel("Especie")
    ax.set_ylabel(feature)

# 5. Ajustar diseño
plt.suptitle("Boxplots de características del Iris por especie")
plt.tight_layout()
plt.show()

# 2. Identificar las columnas cuantitativas
quant_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# 3. Crear la Matriz de Dispersión (Pair Plot)
# Utilizamos 'species' para colorear los puntos y agrupar visualmente los datos
g = sns.pairplot(
    df,
    vars=quant_cols,      # Variables a graficar en la matriz
    hue='species',        # Variable categórica para el color
    palette='viridis',    # Paleta de colores
    height=2.5            # Tamaño de cada subgráfico
)

# 4. Añadir un título general
g.fig.suptitle('Matriz de Dispersión (Scatter Plot Matrix)', y=1.02)

# Reemplazar los histogramas de la diagonal x = y por boxplots coloreados.
especies = sorted(df["species"].unique())
colores_boxplot = sns.color_palette("viridis", len(especies))
for indice, feature in enumerate(quant_cols):
    eje = g.axes[indice, indice]
    eje.clear()
    grupos = [
        df.loc[df["species"] == especie, feature].dropna()
        for especie in especies
    ]
    boxplot = eje.boxplot(
        grupos,
        patch_artist=True,
        boxprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black", "linewidth": 2},
    )
    for caja, color in zip(boxplot["boxes"], colores_boxplot):
        caja.set_facecolor(color)
    eje.set_xticks(range(1, len(especies) + 1))
    eje.set_xticklabels(especies, rotation=45, ha="right")
    eje.set_title(feature)

# 5. Mostrar la matriz
plt.show()

# 6. Calcular e imprimir la matriz de correlación
correlaciones = df[quant_cols].corr()
print("\n=== Matriz de correlación de las características del Iris ===")
print(correlaciones.round(3).to_string())

# 7. Graficar la matriz de correlación
figura_correlacion, eje_correlacion = plt.subplots(figsize=(9, 7))
imagen = eje_correlacion.imshow(correlaciones, cmap="coolwarm", vmin=-1, vmax=1)
eje_correlacion.set_xticks([])
eje_correlacion.set_yticks([])
eje_correlacion.set_title("Matriz de correlación de las características del Iris")
eje_correlacion.title.set_color("white")

for fila in range(len(correlaciones.index)):
    for columna in range(len(correlaciones.columns)):
        eje_correlacion.text(
            columna,
            fila,
            f"{correlaciones.iloc[fila, columna]:.2f}",
            ha="center",
            va="center",
            fontweight="bold",
        )

figura_correlacion.colorbar(
    imagen,
    ax=eje_correlacion,
    label="Coeficiente de correlación",
)
barra_color = figura_correlacion.axes[-1]
barra_color.yaxis.label.set_color("white")
barra_color.tick_params(colors="white")
for spine in barra_color.spines.values():
    spine.set_color("white")
figura_correlacion.tight_layout()
figura_correlacion.patch.set_alpha(0)
eje_correlacion.set_facecolor("none")
plt.show()