# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


MISSING_COLUMNS = ("reviews_per_month", "last_review")


def validate_missing_rows(dataframe: pd.DataFrame, columns: tuple[str, str]) -> None:
  """Valida si dos columnas tienen valores faltantes en las mismas filas."""
  missing_masks = {
    column: dataframe[column].isna()
    for column in columns
  }

  print("\nValidación de datos faltantes")
  print(f"Filas totales: {len(dataframe):,}")

  for column, mask in missing_masks.items():
    count = int(mask.sum())
    percentage = count / len(dataframe) * 100
    print(f"{column}: {count:,} faltantes ({percentage:.2f}%)")

  first_column, second_column = columns
  first_missing = missing_masks[first_column]
  second_missing = missing_masks[second_column]
  same_rows = first_missing.equals(second_missing)
  shared_rows = int((first_missing & second_missing).sum())
  only_first = int((first_missing & ~second_missing).sum())
  only_second = int((~first_missing & second_missing).sum())

  print(f"\n¿Los faltantes están en las mismas tuplas? {'Sí' if same_rows else 'No'}")
  print(f"Faltantes compartidos: {shared_rows:,}")
  print(f"Solo en {first_column}: {only_first:,}")
  print(f"Solo en {second_column}: {only_second:,}")

  if not same_rows:
    different_rows = dataframe.index[first_missing != second_missing]
    print("Primeros índices donde difieren:")
    print(different_rows[:10].tolist())

# Set the path to the file you'd like to load
file_path = "listings.csv"

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "jojoker/singapore-airbnb",
  file_path,
  pandas_kwargs={
    "encoding": "latin1",
    "compression": "zip",
    "low_memory": False,
  },
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:")
print(df.head().to_string(index=False))

missing_columns_not_found = [column for column in MISSING_COLUMNS if column not in df.columns]
if missing_columns_not_found:
  raise KeyError(f"No se encontraron las columnas: {missing_columns_not_found}")

validate_missing_rows(df, MISSING_COLUMNS)

# Los anuncios sin reseñas registradas se consideran sin actividad de
# reseñas; ``last_review`` se elimina porque no se utilizará en el análisis.
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df.drop(columns=["last_review"], inplace=True)

print("\nDataset después de la limpieza:")
print(f"reviews_per_month nulos: {int(df['reviews_per_month'].isna().sum()):,}")
print(f"last_review presente: {'last_review' in df.columns}")
