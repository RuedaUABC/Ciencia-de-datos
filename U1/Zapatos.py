import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fondo general del proyecto y texto blanco
plt.rcParams['figure.facecolor'] = '#76A5B0'
plt.rcParams['axes.facecolor'] = '#76A5B0'
plt.rcParams['savefig.facecolor'] = '#76A5B0'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['legend.labelcolor'] = 'white'

# 1. Cargar el dataset NORMALIZADO
df = pd.read_csv("Datasets/MEN_SHOES_NORMALIZED.csv") # Asegúrate de que la ruta sea correcta

# Si vuelve a haber un error de columnas, esta línea te mostrará los nombres exactos:
# print(df.columns)

# 2. Limpieza de datos (Manejo de nulos)
df = df.dropna(subset=['how_many_sold', 'current_price', 'rating'])

# Variables cuantitativas a analizar
quant_cols = ['current_price', 'how_many_sold', 'rating']

# 4. Seleccionar las cinco categorías "is_..." más frecuentes
is_cols = [col for col in df.columns if col.startswith('is_')]
top_categories = df[is_cols].sum().nlargest(3).index.tolist()

# 5. Generar Matriz de Dispersión y Correlación por categoría
for category in top_categories:
    subset = df[df[category] == 1]

    if len(subset) > 1:
        print(f"{'=' * 40}")
        print(f"CATEGORÍA: {category.upper()} (Registros: {len(subset)})")
        print(f"{'=' * 40}")

        # --- A. CÁLCULO DE LA MATRIZ DE CORRELACIÓN ---
        matriz_corre = subset[quant_cols].corr(method='pearson')
        print("Matriz de Correlación (Valores):")
        print(matriz_corre)
        print("\n")

        # --- B. GRÁFICO 1: MATRIZ DE DISPERSIÓN (PAIRPLOT) ---
        g = sns.pairplot(
            subset,
            vars=quant_cols,
            hue='brand_name',
            palette='tab20',
            diag_kind='hist',
            height=2.5,
            plot_kws={'alpha': 0.7},
            diag_kws={'alpha': 0.6, 'bins': 12}
        )
        g.fig.suptitle(f'Matriz de Dispersión - {category}', y=1.05)
        plt.show()

        # --- C. GRÁFICO 2: MATRIZ DE CORRELACIÓN (HEATMAP) ---
        plt.figure(figsize=(6, 4))
        sns.heatmap(matriz_corre, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
        plt.title(f'Matriz de Correlación (Heatmap) - {category}')
        plt.tight_layout()
        plt.show()


# Resumen tabular de las cinco categorías más frecuentes
print(f"{'=' * 40}\nRESUMEN DE CORRELACIONES POR CATEGORÍA\n{'=' * 40}")
for category in top_categories:
    subset = df[df[category] == 1]
    print(f"\n{category.upper()}:")
    print(subset[quant_cols].corr(method='pearson'))

# Histograma de frecuencias de las categorías binarias "is_..."
if is_cols:
    frecuencias = df[is_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(14, 7))
    frecuencias.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Frecuencia de las categorías de características')
    plt.xlabel('Categoría')
    plt.ylabel('Número de productos')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron columnas con el prefijo 'is_'.")