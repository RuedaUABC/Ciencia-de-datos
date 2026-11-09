# Unidad 4: Categorización

## Objetivo

Comparar distintos clasificadores y métodos de validación sobre dos conjuntos de datos, con y sin selección de características, para identificar las configuraciones que obtienen los mejores resultados.

## Conjuntos de datos

- [x] **Iris**, disponible en `Datasets/iris.csv`.
- [x] **Mobile Price Classification**, disponible en `Datasets/Mobile Price Classification/train.csv`.

En Iris, transformar la variable `species` en un objetivo binario: **versicolor** (clase positiva) frente a **no versicolor** (setosa y virginica combinadas). En Mobile Price Classification, `train.csv` contiene la variable objetivo `price_range` con cuatro clases (`0` a `3`). Su archivo `test.csv` no incluye esa etiqueta, por lo que no sirve para calcular métricas ni una matriz de confusión.

Aplicar los experimentos y la evaluación a cada conjunto de datos.

## Métodos de validación

Evaluar cada conjunto de datos usando los siguientes métodos:

1. **Hold-Out:** partición 60/40.
2. **Entrenamiento/Prueba:** usar conjuntos distintos con la misma cantidad de registros para comprobar overfitting. Para Iris, por tener 150 registros, usar **75 para entrenamiento y 75 para prueba**. Como el CSV está ordenado por especie, asignar las filas alternadamente: una a entrenamiento, la siguiente a prueba, y repetir. Así quedan 25 registros de cada especie (25 versicolor positivos y 50 no versicolor) en cada conjunto. Para Mobile Price Classification, tomar 100 registros etiquetados distintos para entrenamiento y otros 100 para prueba, ambos desde `train.csv`.
3. **Random subsampling:** `N = 30` repeticiones.
4. **K-Fold:** `K = 10`.
5. **Leave-One-Out (LOO)**.
6. **Stratified Cross-Validation:** `K = 10`.

## Clasificadores

Usar cinco clasificadores en cada caso:

- Regresión logística.
- Redes neuronales.
- Tres técnicas adicionales de elección propia:
  1. KNN
  2. SVM
  3. Árboles de decisión

## Selección de características

Comparar cada experimento en dos condiciones:

- Sin selección de características.
- Con selección de características usando **ANOVA F-test** (`SelectKBest` con `f_classif`).

Aplicar ANOVA por separado a cada conjunto de datos y conservar la mitad de sus variables predictoras con mayor puntuación: las **2 mejores de Iris** (4 variables) y las **10 mejores de Mobile Price Classification** (20 variables). Este filtro es independiente del clasificador, por lo que se usará el mismo criterio al comparar los cinco modelos.

En cada partición, ajustar el selector usando únicamente el conjunto de entrenamiento. Después, aplicar la transformación aprendida al conjunto de prueba o validación correspondiente para evitar filtración de información. La condición sin selección debe usar todas las variables predictoras.

## Evaluación

Para cada combinación de conjunto de datos, método de validación, clasificador y condición de selección de características:

- Calcular estas **10 métricas**:
  1. **Accuracy** (exactitud).
  2. **Precisión macro**.
  3. **Recall macro** (sensibilidad macro).
  4. **F1 macro**.
  5. **Especificidad** para Iris (versicolor como clase positiva) y especificidad macro uno-contra-el-resto para Mobile Price Classification.
  6. **Coeficiente de correlación de Matthews (MCC)**: binario para Iris y multiclase para Mobile Price Classification.
  7. **Kappa de Cohen**.
  8. **ROC-AUC**: binario para Iris (versicolor como clase positiva) y macro uno-contra-el-resto (OvR) para Mobile Price Classification.
  9. **Log loss** (pérdida logarítmica).
  10. **Brier score**, binario para Iris y multiclase para Mobile Price Classification.
- Incluir la **matriz de confusión**.
- Registrar los resultados de forma que se puedan comparar entre experimentos.

Las métricas macro promedian las clases por igual. ROC-AUC, log loss y Brier score requieren probabilidades por clase; obtenerlas de modelos ajustados solo con los datos de entrenamiento de cada partición. Para SVM, habilitar la estimación de probabilidades. En Iris, tratar `versicolor` como la clase positiva para las métricas binarias. En Mobile Price Classification, calcular las métricas en sus cuatro clases.

La matriz de confusión debe mostrar **conteos**, con una fila por clase real y una columna por clase predicha: será de `2 × 2` para Iris (`versicolor` y `no versicolor`) y de `4 × 4` para Mobile Price Classification. Calcular las métricas a partir de las predicciones, usando el mismo esquema para todos los clasificadores de cada conjunto.

## Planeación del procedimiento

Seguir este orden para que las comparaciones sean reproducibles y no haya filtración de información entre entrenamiento y prueba:

1. **Preparar los datos.** Cargar Iris y Mobile Price Classification. En Iris, convertir `species` en la etiqueta binaria `versicolor`/`no versicolor`. En Mobile, usar `train.csv`, separar `price_range` como objetivo y conservar sus 20 variables como predictores. No usar el `test.csv` oficial de Mobile para calcular métricas porque no contiene etiquetas.
2. **Crear y guardar las particiones.** Para Hold-Out, usar una separación aleatoria estratificada 60/40. Para Entrenamiento/Prueba, usar la partición acordada para cada dataset: Iris 75/75 alternando las filas del CSV (una para entrenamiento, la siguiente para prueba); Mobile 100 registros etiquetados distintos para entrenamiento y otros 100 para prueba, muestreados estratificadamente desde `train.csv`. Para Random Subsampling (`N=30`), repetir 30 veces la separación estratificada 60/40. Ejecutar K-Fold (`K=10`) barajando las filas para evitar que el orden original de Iris agrupe las clases, Leave-One-Out y Stratified Cross-Validation (`K=10`). Fijar y registrar la semilla aleatoria. Mantener las mismas particiones al comparar clasificadores y selección de características.
3. **Preparar las variantes de características.** Para cada partición, ejecutar una variante con todas las variables y otra con ANOVA (`SelectKBest(f_classif)`): `k=2` para Iris y `k=10` para Mobile. Ajustar ANOVA solo con el subconjunto de entrenamiento de esa partición y luego transformar su prueba/validación.
4. **Preparar y ajustar los clasificadores.** Entrenar regresión logística, red neuronal, KNN, SVM y árbol de decisión. Estandarizar las variables de regresión logística, red neuronal, KNN y SVM dentro del pipeline de entrenamiento; el árbol no requiere escalado. Registrar los parámetros utilizados y configurar SVM para producir probabilidades.
5. **Ejecutar todos los experimentos.** Para cada dataset, método de validación, clasificador y variante de características, ajustar el modelo con entrenamiento y obtener predicciones (y probabilidades cuando se necesiten). Calcular las diez métricas y construir la matriz de confusión a partir de las predicciones de prueba/validación. En Iris, revisar especialmente el desempeño de la clase positiva `versicolor`.
6. **Resumir los resultados.** En Hold-Out y Entrenamiento/Prueba, registrar el resultado de esa partición. Para Random Subsampling y los métodos con varios folds, reportar promedio y desviación estándar de las métricas; reunir las predicciones fuera de muestra para construir la matriz de confusión total. En Leave-One-Out, calcular las métricas usando en conjunto las predicciones de todas las observaciones dejadas fuera.
7. **Comparar y concluir.** Completar la tabla de resultados, contrastar cada clasificador con y sin ANOVA y comparar los métodos de validación. Identificar las mejores configuraciones por dataset y justificarlo con las métricas, las matrices de confusión y la capacidad de detectar `versicolor` en Iris.

### Estado de los pasos 1 y 2

Los pasos de preparación de datos y creación de particiones ya están ejecutados con `preparar_datos.py`. `row_id` conserva el orden original de cada archivo y debe excluirse de las variables predictoras.

| Dataset preparado | Registros | Variables predictoras | Distribución del objetivo |
|---|---:|---:|---|
| Iris binario | 150 | 4 | 100 no versicolor, 50 versicolor |
| Mobile Price Classification | 2,000 | 20 | 500 por cada clase (`0`–`3`) |

Los archivos preparados están en `Datos preparados/`. Las asignaciones de Hold-Out, Entrenamiento/Prueba, Random Subsampling, K-Fold, Leave-One-Out y Stratified Cross-Validation están en `Particiones/`; `Particiones/resumen_particiones.csv` resume tamaños, repeticiones y semillas. Las particiones 75/75 de Iris tienen 25 positivos y 50 negativos en cada subconjunto; las particiones 100/100 de Mobile tienen 25 ejemplos de cada clase en train y test.

## Procedimiento experimental

Documentar el procedimiento completo de cada experimento, incluyendo:

1. Preparación del conjunto de datos y las variables utilizadas.
2. Método de validación y particiones generadas.
3. Clasificador y configuración empleada.
4. Si se aplicó selección de características y cómo se ajustó dentro del entrenamiento.
5. Métricas calculadas y matriz de confusión.
6. Resultado obtenido y observaciones relevantes.

## Tabla comparativa de experimentos

Completar una fila por cada experimento. Incluir las diez métricas y la información necesaria para identificar la mejor configuración.

| Dataset | Validación | Clasificador | Selección de características | Accuracy | Precisión macro | Recall macro | F1 macro | Especificidad | MCC | Kappa | ROC-AUC | Log loss | Brier score | Matriz de confusión / referencia | Observaciones |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| | | | | | | | | | | | | | | | |

## Resultados y conclusiones

Presentar los resultados obtenidos para ambos conjuntos de datos. Comparar el desempeño entre clasificadores, métodos de validación y condiciones con o sin selección de características. Indicar cuál configuración obtuvo el mejor resultado y justificar la conclusión con las métricas registradas.
