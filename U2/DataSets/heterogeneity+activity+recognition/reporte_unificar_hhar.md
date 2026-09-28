# Reporte de unificación del conjunto HHAR

## 1. Propósito

El script [`unificar_hhar.py`](./unificar_hhar.py) integra los archivos CSV del conjunto **Heterogeneity Dataset for Human Activity Recognition (HHAR)** en un único archivo tabular con una estructura homogénea.

El dataset contiene dos experimentos con formatos diferentes:

- `Still exp`: mediciones estáticas del acelerómetro distribuidas en muchos archivos y subdirectorios.
- `Activity recognition exp`: mediciones de acelerómetro y giroscopio realizadas durante actividades humanas, con información de usuario, dispositivo y actividad real (`gt`).

El resultado es una tabla en formato largo. Cada fila representa una medición y conserva la información necesaria para identificar su experimento, sensor, dispositivo, orientación, actividad y archivo de origen.

## 2. Descubrimiento de archivos

El script recibe como carpeta de entrada, por defecto, la carpeta donde se encuentra el propio archivo. La búsqueda se realiza recursivamente en todos sus subdirectorios, por lo que no depende de que los archivos estén ubicados directamente en `Still exp` o `Activity recognition exp`.

Durante el descubrimiento:

- Se aceptan extensiones `.csv` sin distinguir mayúsculas y minúsculas.
- Se excluyen los archivos de salida `hhar_unificado.csv` y `hhar_unificado_auditoria.csv`.
- Se excluyen los metadatos generados por macOS ubicados en `__MACOSX` o cuyos nombres comienzan con `._`.
- Los archivos se procesan en orden alfabético.

Esta estrategia permite procesar tanto las mediciones estáticas como los cuatro archivos del experimento de actividad descritos en el `readme.txt`:

| Archivo | Sensor identificado |
|---|---|
| `Phones_accelerometer.csv` | Acelerómetro |
| `Phones_gyroscope.csv` | Giroscopio |
| `Watch_accelerometer.csv` | Acelerómetro |
| `Watch_gyroscope.csv` | Giroscopio |

## 3. Identificación de los formatos de entrada

Cada archivo se abre con `csv.reader`. La cabecera se limpia para hacer comparables nombres como `Arrival_Time`, `Creation_Time`, `Sensor_Time`, `Index` y `groundtruth`.

También se aplican alias internos:

- `arrivaltime` se convierte en `arrival_time`.
- `creationtime` se convierte en `creation_time`.
- `sensortime` se convierte en `sensor_time`.
- `sample_index` se convierte en `index`.
- `groundtruth` se convierte en `gt`.

El experimento se identifica principalmente por la cabecera:

- Si contiene `gt`, `user`, `model` o `device`, se clasifica como `activity_recognition`.
- En caso contrario, se clasifica como `still`.

Esto evita depender únicamente del nombre de las carpetas.

El experimento estático puede presentar tres esquemas de origen:

1. `Creation_Time | Sensor_Time | Arrival_Time | x | y | z`
2. `Index | Arrival_Time | Creation_Time | x | y | z`
3. `Arrival_Time | Creation_Time | x | y | z`

El experimento de actividad utiliza el esquema:

```text
Index, Arrival_Time, Creation_Time, x, y, z, User, Model, Device, gt
```

El script registra el esquema detectado en la columna `source_schema`.

## 4. Metadatos derivados

Para cada archivo se obtienen o infieren los siguientes metadatos:

- `experiment`: `still` o `activity_recognition`.
- `sensor_type`: acelerómetro o giroscopio.
- `sensor_unit`: `m/s2` para acelerómetros y `unknown` para giroscopios.
- `user`: usuario indicado en el CSV de actividad.
- `device_type`: teléfono inteligente, reloj inteligente u otro dispositivo.
- `device_id`: identificador específico del dispositivo.
- `device_model`: modelo del dispositivo.
- `orientation`: `back`, `front`, `bottom`, `top`, `left`, `right` o `unknown`.
- `source_file`: ruta relativa del archivo original.

Para `Still exp`, la orientación se obtiene de nombres como `Phoneonback`, `Phoneonfront`, `Phoneontop`, `Phoneonbottom`, `Phoneonleft` y `Phoneonright`. En los archivos de actividad, la orientación normalmente queda como `unknown` porque no forma parte de ese experimento.

El tipo de dispositivo se infiere a partir del modelo, el identificador y la ruta. Los nombres que incluyen `gear` o `watch` se clasifican como relojes inteligentes; `x-sense` se clasifica como otro dispositivo y el resto como teléfono inteligente.

## 5. Normalización de actividades

Los valores originales de `gt` se conservan en `activity_raw` y se normalizan en `activity`.

| Valor original | Valor normalizado |
|---|---|
| `bike`, `biking` | `biking` |
| `sit`, `sitting` | `sitting` |
| `stand`, `standing` | `standing` |
| `walk`, `walking` | `walking` |
| `stairsup`, `stairup` | `stair_up` |
| `stairsdown`, `stairdown` | `stair_down` |
| vacío, `null`, `none`, `nan`, `unknown` | `unknown` |

Las filas del experimento estático reciben `still` como actividad, ya que ese experimento no contiene una etiqueta de actividad humana equivalente a `gt`.

## 6. Normalización de tiempos

HHAR utiliza distintas escalas para `Creation_Time`. El script inspecciona hasta 1,000 muestras por archivo y estima la unidad mediante el valor de los timestamps, sus diferencias entre muestras y la comparación con `Arrival_Time`.

Se reconocen las siguientes unidades:

- `epoch_ms`: cuando `Creation_Time` y `Arrival_Time` coinciden prácticamente en todas las muestras.
- `nanoseconds`: cuando la magnitud o los saltos entre muestras corresponden a nanosegundos.
- `microseconds`: cuando la magnitud o los saltos corresponden a microsegundos.
- `unknown`: cuando no hay suficientes datos para identificar la unidad.

Los valores reconocidos se convierten a milisegundos en `creation_time_ms`. El valor original se conserva en `creation_time_raw`, y la unidad detectada se registra en `creation_time_unit`.

El campo `creation_time_origin` se establece como:

- `epoch` para timestamps identificados como `epoch_ms`.
- `device_clock_relative` para timestamps relativos del reloj del dispositivo.

`Arrival_Time` se conserva como `arrival_time_ms`. Cuando existe `Sensor_Time`, se guardan tanto el valor original (`sensor_time_raw`) como su representación numérica (`sensor_time_ms`).

## 7. Validación de las mediciones

Para cada fila se convierten los ejes `x`, `y` y `z` a números de punto flotante. Los valores no numéricos, infinitos o ausentes provocan que la fila sea omitida.

El script no aplica estandarización estadística ni z-score. Los ejes se mantienen en sus unidades originales para conservar la información física de orientación y magnitud del sensor.

Cada fila válida se escribe con el siguiente esquema canónico de 24 columnas:

```text
experiment
activity
activity_raw
sensor_type
sensor_unit
user
device_type
device_id
device_model
orientation
sample_index
source_row_number
arrival_time_ms
creation_time_raw
creation_time_ms
creation_time_unit
creation_time_origin
sensor_time_raw
sensor_time_ms
x
y
z
source_schema
source_file
```

Además de los datos originales, se conserva `source_row_number` para localizar la fila en el archivo fuente y `source_file` para rastrear su procedencia.

## 8. Archivo de auditoría

Por cada CSV procesado se genera una fila en `hhar_unificado_auditoria.csv`. La auditoría contiene:

- archivo de origen y cabecera original;
- esquema detectado;
- experimento, orientación, dispositivo y modelo;
- unidad de `Creation_Time`;
- filas leídas;
- filas escritas;
- filas omitidas;
- campos adicionales descartados.

La auditoría permite verificar si un archivo fue incluido y distinguir entre filas procesadas y filas descartadas por ejes inválidos.

## 9. Resultados de la ejecución

La ejecución se realizó con sobrescritura de las salidas anteriores:

```bash
python3 \
  "U2/DataSets/heterogeneity+activity+recognition/unificar_hhar.py" \
  --overwrite
```

Resultados obtenidos:

| Métrica | Resultado |
|---|---:|
| Archivos procesados | 202 |
| Archivos de `Still exp` | 198 |
| Archivos de `Activity recognition exp` | 4 |
| Filas leídas | 36,349,812 |
| Filas escritas | 36,349,812 |
| Filas omitidas | 0 |
| Líneas en la salida, incluyendo cabecera | 36,349,813 |

Los cuatro archivos de actividad aportaron las siguientes filas:

| Archivo | Filas escritas |
|---|---:|
| `Phones_accelerometer.csv` | 13,062,475 |
| `Phones_gyroscope.csv` | 13,932,632 |
| `Watch_accelerometer.csv` | 3,540,962 |
| `Watch_gyroscope.csv` | 3,205,431 |

Los archivos generados son:

- [`hhar_unificado.csv`](./hhar_unificado.csv): archivo unificado con todas las mediciones.
- [`hhar_unificado_auditoria.csv`](./hhar_unificado_auditoria.csv): detalle del procesamiento de cada archivo fuente.

## 10. Opciones de ejecución

El script admite las siguientes opciones:

| Opción | Función |
|---|---|
| `--input-dir RUTA` | Cambia la carpeta raíz de búsqueda. |
| `--output RUTA` | Cambia la ubicación del CSV unificado. |
| `--audit RUTA` | Cambia la ubicación de la auditoría. |
| `--overwrite` | Permite reemplazar salidas existentes. |

Sin `--overwrite`, el script se detiene si ya existen los archivos de salida, evitando sobrescrituras accidentales.

## 11. Consideraciones y limitaciones

- La salida completa es considerablemente más grande que los archivos fuente porque incorpora 24 columnas normalizadas y metadatos de trazabilidad.
- El proceso escribe fila por fila; con los archivos de actividad completos puede tardar varios minutos.
- Las filas con ejes `x`, `y` o `z` inválidos no se incluyen en la salida, pero quedan contabilizadas en `rows_skipped`.
- El experimento estático no tiene una etiqueta de actividad humana; por ello se marca como `still`.
- La detección de unidades de tiempo depende de los valores disponibles en cada archivo. Si la unidad no puede inferirse, `creation_time_ms` queda vacío y se conserva el valor original.
- Los archivos de metadatos de macOS se excluyen intencionalmente para evitar incorporarlos como mediciones.

## 12. Conclusión

`unificar_hhar.py` resuelve la heterogeneidad estructural del conjunto HHAR mediante descubrimiento recursivo, identificación del formato por cabecera, normalización de metadatos, conversión de tiempos, validación de los ejes y generación de una auditoría por archivo.

La ejecución verificada incorporó los 202 CSV disponibles y no omitió ninguna fila por datos inválidos en `x`, `y` o `z`.
