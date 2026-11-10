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

### Datos y particiones

- **Iris:** 150 observaciones, cuatro predictores numéricos y etiqueta binaria; `versicolor=1` y `no versicolor=0`. La partición Entrenamiento/Prueba usa filas alternadas, 75/75.
- **Mobile Price Classification:** 2,000 filas etiquetadas de `train.csv`, 20 predictores y `price_range` (0–3). Se excluyó `row_id` y no se usó el `test.csv` oficial. Entrenamiento/Prueba toma 100 filas para train y otras 100 para test.
- Se reutilizaron las particiones guardadas para Hold-Out 60/40, 30 repeticiones de Random Subsampling, K-Fold 10 barajado, LOO y Stratified Cross-Validation 10. Hold-Out y Random Subsampling son estratificados; también lo es Entrenamiento/Prueba de Mobile. Las particiones usan semilla 42, salvo Entrenamiento/Prueba de Iris (43) y Mobile (44).

### Preparación de variables y clasificadores

Las variantes se ejecutaron sobre cada misma partición: todas las variables, o `SelectKBest(f_classif)` dentro del pipeline de entrenamiento (`k=2` para Iris y `k=10` para Mobile). El selector y el escalador se ajustaron exclusivamente con train en cada partición. Regresión logística, MLP, KNN y SVM incluyen `StandardScaler`; el árbol no se escala.

| Clasificador | Configuración registrada |
|---|---|
| Regresión logística | `StandardScaler; LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)` |
| Red neuronal (MLP) | `StandardScaler; MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=50, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1, random_state=42)` |
| KNN | `StandardScaler; KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)` |
| SVM lineal | `StandardScaler; SVC(C=1.0, kernel='linear', probability=True, random_state=42)` |
| Árbol de decisión | `DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)` |

SVM entrega probabilidades con `probability=True`. La red MLP usa una capa oculta de 16 neuronas y máximo 50 épocas con parada temprana para que LOO sobre Mobile sea ejecutable; ese límite forma parte de la configuración comparada.

### Cálculo de métricas

Las métricas macro dan el mismo peso a cada clase. Especificidad usa la clase positiva `versicolor` en Iris y promedio uno-contra-el-resto en Mobile. ROC-AUC usa probabilidades (binario para Iris y macro OvR para Mobile); log loss usa las etiquetas posibles completas. Brier es el error cuadrático binario en Iris y la suma multiclase por observación en Mobile. Para Random Subsampling y validaciones cruzadas se presenta media y desviación estándar entre particiones; sus matrices reúnen predicciones de prueba. LOO calcula las métricas usando juntas sus predicciones fuera de muestra. Las métricas de un fold LOO individual quedan vacías en `resultados_por_particion.csv`, porque un único caso no permite calcularlas de forma válida.

## Tabla comparativa de experimentos

Cada fila corresponde a una combinación de dataset, protocolo, clasificador y variante de características. En protocolos con varias particiones, las métricas son media ± desviación estándar entre particiones; LOO usa las predicciones fuera de muestra reunidas. La matriz presenta conteos con clases reales en filas y predichas en columnas. En Mobile, Brier es la suma de los errores cuadráticos por observación.

| Dataset | Validación | Clasificador | Selección | Accuracy | Precisión macro | Recall macro | F1 macro | Especificidad | MCC | Kappa | ROC-AUC | Log loss | Brier | Matriz de confusión |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Iris binario | Entrenamiento/Prueba | KNN | ANOVA F-test | 0.9333 | 0.9219 | 0.9300 | 0.9257 | 0.9400 | 0.8519 | 0.8515 | 0.9684 | 0.5856 | 0.0517 | `[[47,3],[2,23]]` |
| Iris binario | Entrenamiento/Prueba | KNN | Sin selección | 0.9600 | 0.9513 | 0.9600 | 0.9554 | 0.9600 | 0.9113 | 0.9109 | 0.9920 | 0.1259 | 0.0389 | `[[48,2],[1,24]]` |
| Iris binario | Entrenamiento/Prueba | Red neuronal (MLP) | ANOVA F-test | 0.5067 | 0.3390 | 0.3900 | 0.3590 | 0.7400 | -0.2662 | -0.2472 | 0.5164 | 0.6644 | 0.2360 | `[[37,13],[24,1]]` |
| Iris binario | Entrenamiento/Prueba | Red neuronal (MLP) | Sin selección | 0.6267 | 0.5000 | 0.5000 | 0.4675 | 0.8800 | 0.0000 | 0.0000 | 0.5264 | 0.6603 | 0.2335 | `[[44,6],[22,3]]` |
| Iris binario | Entrenamiento/Prueba | Regresión logística | ANOVA F-test | 0.7333 | 0.7028 | 0.6600 | 0.6693 | 0.8800 | 0.3603 | 0.3478 | 0.8148 | 0.5106 | 0.1708 | `[[44,6],[14,11]]` |
| Iris binario | Entrenamiento/Prueba | Regresión logística | Sin selección | 0.7467 | 0.7251 | 0.6700 | 0.6811 | 0.9000 | 0.3912 | 0.3736 | 0.8168 | 0.5020 | 0.1662 | `[[45,5],[14,11]]` |
| Iris binario | Entrenamiento/Prueba | SVM lineal | ANOVA F-test | 0.7333 | 0.7308 | 0.6300 | 0.6345 | 0.9400 | 0.3465 | 0.3023 | 0.7900 | 0.5313 | 0.1788 | `[[47,3],[17,8]]` |
| Iris binario | Entrenamiento/Prueba | SVM lineal | Sin selección | 0.7467 | 0.7342 | 0.6600 | 0.6708 | 0.9200 | 0.3871 | 0.3596 | 0.8088 | 0.5403 | 0.1812 | `[[46,4],[15,10]]` |
| Iris binario | Entrenamiento/Prueba | Árbol de decisión | ANOVA F-test | 0.9733 | 0.9808 | 0.9600 | 0.9694 | 1.0000 | 0.9405 | 0.9388 | 0.9584 | 0.9395 | 0.0333 | `[[50,0],[2,23]]` |
| Iris binario | Entrenamiento/Prueba | Árbol de decisión | Sin selección | 0.9600 | 0.9513 | 0.9600 | 0.9554 | 0.9600 | 0.9113 | 0.9109 | 0.9600 | 1.3816 | 0.0400 | `[[48,2],[1,24]]` |
| Iris binario | Hold-Out 60/40 | KNN | ANOVA F-test | 0.9000 | 0.8875 | 0.8875 | 0.8875 | 0.9250 | 0.7750 | 0.7750 | 0.9600 | 0.2294 | 0.0784 | `[[37,3],[3,17]]` |
| Iris binario | Hold-Out 60/40 | KNN | Sin selección | 0.9500 | 0.9493 | 0.9375 | 0.9430 | 0.9750 | 0.8867 | 0.8861 | 0.9838 | 0.6561 | 0.0410 | `[[39,1],[2,18]]` |
| Iris binario | Hold-Out 60/40 | Red neuronal (MLP) | ANOVA F-test | 0.5167 | 0.4316 | 0.4375 | 0.4334 | 0.6750 | -0.1308 | -0.1299 | 0.5600 | 0.6519 | 0.2301 | `[[27,13],[16,4]]` |
| Iris binario | Hold-Out 60/40 | Red neuronal (MLP) | Sin selección | 0.6667 | 0.5862 | 0.5125 | 0.4434 | 0.9750 | 0.0657 | 0.0323 | 0.7638 | 0.6195 | 0.2139 | `[[39,1],[19,1]]` |
| Iris binario | Hold-Out 60/40 | Regresión logística | ANOVA F-test | 0.6167 | 0.5642 | 0.5625 | 0.5632 | 0.7250 | 0.1267 | 0.1266 | 0.6837 | 0.5922 | 0.2125 | `[[29,11],[12,8]]` |
| Iris binario | Hold-Out 60/40 | Regresión logística | Sin selección | 0.6500 | 0.6027 | 0.6000 | 0.6011 | 0.7500 | 0.2027 | 0.2025 | 0.7137 | 0.5759 | 0.2052 | `[[30,10],[11,9]]` |
| Iris binario | Hold-Out 60/40 | SVM lineal | ANOVA F-test | 0.6500 | 0.5889 | 0.5750 | 0.5765 | 0.8000 | 0.1633 | 0.1600 | 0.7462 | 0.5797 | 0.2050 | `[[32,8],[13,7]]` |
| Iris binario | Hold-Out 60/40 | SVM lineal | Sin selección | 0.6833 | 0.6412 | 0.6375 | 0.6391 | 0.7750 | 0.2787 | 0.2785 | 0.7288 | 0.5585 | 0.1960 | `[[31,9],[10,10]]` |
| Iris binario | Hold-Out 60/40 | Árbol de decisión | ANOVA F-test | 0.9167 | 0.9241 | 0.8875 | 0.9023 | 0.9750 | 0.8108 | 0.8052 | 0.8875 | 2.8782 | 0.0833 | `[[39,1],[4,16]]` |
| Iris binario | Hold-Out 60/40 | Árbol de decisión | Sin selección | 0.9000 | 0.8968 | 0.8750 | 0.8845 | 0.9500 | 0.7715 | 0.7692 | 0.8750 | 3.4539 | 0.1000 | `[[38,2],[4,16]]` |
| Iris binario | K-Fold (K=10) | KNN | ANOVA F-test | 0.9267 ± 0.0734 | 0.9174 ± 0.0843 | 0.9192 ± 0.0995 | 0.9147 ± 0.0928 | 0.9385 ± 0.0535 | 0.8360 ± 0.1825 | 0.8302 ± 0.1845 | 0.9468 ± 0.0615 | 1.0396 ± 1.2096 | 0.0648 ± 0.0516 | `[[94,6],[5,45]]` |
| Iris binario | K-Fold (K=10) | KNN | Sin selección | 0.9533 ± 0.0706 | 0.9516 ± 0.0800 | 0.9427 ± 0.0991 | 0.9435 ± 0.0930 | 0.9687 ± 0.0507 | 0.8937 ± 0.1788 | 0.8880 ± 0.1845 | 0.9811 ± 0.0536 | 0.3124 ± 0.7747 | 0.0331 ± 0.0434 | `[[97,3],[4,46]]` |
| Iris binario | K-Fold (K=10) | Red neuronal (MLP) | ANOVA F-test | 0.6533 ± 0.0932 | 0.5907 ± 0.1886 | 0.5556 ± 0.1084 | 0.5258 ± 0.1208 | 0.8745 ± 0.1082 | 0.1491 ± 0.2398 | 0.1179 ± 0.2185 | 0.6232 ± 0.2412 | 0.6505 ± 0.0383 | 0.2292 ± 0.0188 | `[[87,13],[39,11]]` |
| Iris binario | K-Fold (K=10) | Red neuronal (MLP) | Sin selección | 0.7067 ± 0.0900 | 0.6053 ± 0.2317 | 0.5824 ± 0.1399 | 0.5305 ± 0.1550 | 0.9615 ± 0.0502 | 0.2112 ± 0.2816 | 0.1672 ± 0.2639 | 0.6121 ± 0.1503 | 0.6404 ± 0.0470 | 0.2239 ± 0.0226 | `[[96,4],[40,10]]` |
| Iris binario | K-Fold (K=10) | Regresión logística | ANOVA F-test | 0.7000 ± 0.0786 | 0.6580 ± 0.1512 | 0.6346 ± 0.1293 | 0.6234 ± 0.1254 | 0.8458 ± 0.0828 | 0.2881 ± 0.2663 | 0.2666 ± 0.2466 | 0.7977 ± 0.0949 | 0.5213 ± 0.0610 | 0.1784 ± 0.0274 | `[[84,16],[29,21]]` |
| Iris binario | K-Fold (K=10) | Regresión logística | Sin selección | 0.7333 ± 0.0943 | 0.6969 ± 0.1592 | 0.6720 ± 0.1346 | 0.6576 ± 0.1408 | 0.8707 ± 0.0851 | 0.3621 ± 0.2711 | 0.3412 ± 0.2562 | 0.7967 ± 0.0833 | 0.5182 ± 0.0576 | 0.1764 ± 0.0254 | `[[87,13],[27,23]]` |
| Iris binario | K-Fold (K=10) | SVM lineal | ANOVA F-test | 0.7200 ± 0.0689 | 0.6915 ± 0.1527 | 0.6312 ± 0.0969 | 0.6167 ± 0.1147 | 0.9140 ± 0.0683 | 0.3110 ± 0.2234 | 0.2763 ± 0.2000 | 0.8071 ± 0.0863 | 0.5319 ± 0.0522 | 0.1801 ± 0.0241 | `[[91,9],[33,17]]` |
| Iris binario | K-Fold (K=10) | SVM lineal | Sin selección | 0.7267 ± 0.1153 | 0.6978 ± 0.1663 | 0.6732 ± 0.1387 | 0.6592 ± 0.1457 | 0.8364 ± 0.1490 | 0.3640 ± 0.2830 | 0.3431 ± 0.2681 | 0.7768 ± 0.0961 | 0.5332 ± 0.0608 | 0.1818 ± 0.0278 | `[[84,16],[25,25]]` |
| Iris binario | K-Fold (K=10) | Árbol de decisión | ANOVA F-test | 0.9333 ± 0.0889 | 0.9385 ± 0.0895 | 0.9194 ± 0.1075 | 0.9235 ± 0.1006 | 0.9556 ± 0.0777 | 0.8556 ± 0.1925 | 0.8483 ± 0.1997 | 0.9176 ± 0.1082 | 2.3238 ± 3.0626 | 0.0741 ± 0.0868 | `[[96,4],[6,44]]` |
| Iris binario | K-Fold (K=10) | Árbol de decisión | Sin selección | 0.9400 ± 0.0584 | 0.9346 ± 0.0643 | 0.9356 ± 0.0716 | 0.9314 ± 0.0688 | 0.9496 ± 0.0536 | 0.8696 ± 0.1327 | 0.8637 ± 0.1368 | 0.9356 ± 0.0716 | 2.0723 ± 2.0161 | 0.0600 ± 0.0584 | `[[95,5],[4,46]]` |
| Iris binario | Leave-One-Out (LOO) | KNN | ANOVA F-test | 0.9267 | 0.9159 | 0.9200 | 0.9179 | 0.9400 | 0.8359 | 0.8358 | 0.9503 | 1.0354 | 0.0629 | `[[94,6],[5,45]]` |
| Iris binario | Leave-One-Out (LOO) | KNN | Sin selección | 0.9467 | 0.9400 | 0.9400 | 0.9400 | 0.9600 | 0.8800 | 0.8800 | 0.9793 | 0.5327 | 0.0369 | `[[96,4],[4,46]]` |
| Iris binario | Leave-One-Out (LOO) | Red neuronal (MLP) | ANOVA F-test | 0.6600 | 0.5831 | 0.5450 | 0.5295 | 0.8900 | 0.1223 | 0.1053 | 0.7590 | 0.6261 | 0.2172 | `[[89,11],[40,10]]` |
| Iris binario | Leave-One-Out (LOO) | Red neuronal (MLP) | Sin selección | 0.6867 | 0.6773 | 0.5450 | 0.5042 | 0.9700 | 0.1786 | 0.1132 | 0.6590 | 0.6273 | 0.2176 | `[[97,3],[44,6]]` |
| Iris binario | Leave-One-Out (LOO) | Regresión logística | ANOVA F-test | 0.7200 | 0.6838 | 0.6450 | 0.6528 | 0.8700 | 0.3265 | 0.3152 | 0.7782 | 0.5239 | 0.1795 | `[[87,13],[29,21]]` |
| Iris binario | Leave-One-Out (LOO) | Regresión logística | Sin selección | 0.7200 | 0.6827 | 0.6500 | 0.6577 | 0.8600 | 0.3311 | 0.3226 | 0.7866 | 0.5176 | 0.1762 | `[[86,14],[28,22]]` |
| Iris binario | Leave-One-Out (LOO) | SVM lineal | ANOVA F-test | 0.7200 | 0.6984 | 0.6200 | 0.6233 | 0.9200 | 0.3086 | 0.2759 | 0.7720 | 0.5258 | 0.1789 | `[[92,8],[34,16]]` |
| Iris binario | Leave-One-Out (LOO) | SVM lineal | Sin selección | 0.7133 | 0.6739 | 0.6400 | 0.6471 | 0.8600 | 0.3121 | 0.3027 | 0.7720 | 0.5337 | 0.1817 | `[[86,14],[29,21]]` |
| Iris binario | Leave-One-Out (LOO) | Árbol de decisión | ANOVA F-test | 0.9467 | 0.9494 | 0.9300 | 0.9387 | 0.9800 | 0.8792 | 0.8776 | 0.9270 | 1.8633 | 0.0607 | `[[98,2],[6,44]]` |
| Iris binario | Leave-One-Out (LOO) | Árbol de decisión | Sin selección | 0.9400 | 0.9308 | 0.9350 | 0.9328 | 0.9500 | 0.8658 | 0.8657 | 0.9350 | 2.0723 | 0.0600 | `[[95,5],[4,46]]` |
| Iris binario | Random Subsampling (N=30) | KNN | ANOVA F-test | 0.9106 ± 0.0261 | 0.9012 ± 0.0293 | 0.9008 ± 0.0303 | 0.8995 ± 0.0288 | 0.9300 ± 0.0356 | 0.8018 ± 0.0563 | 0.7994 ± 0.0574 | 0.9456 ± 0.0227 | 1.0982 ± 0.5625 | 0.0735 ± 0.0165 | `[[1116,84],[77,523]]` |
| Iris binario | Random Subsampling (N=30) | KNN | Sin selección | 0.9500 ± 0.0205 | 0.9445 ± 0.0249 | 0.9446 ± 0.0234 | 0.9439 ± 0.0229 | 0.9608 ± 0.0252 | 0.8890 ± 0.0459 | 0.8878 ± 0.0458 | 0.9844 ± 0.0138 | 0.3125 ± 0.3136 | 0.0387 ± 0.0113 | `[[1153,47],[43,557]]` |
| Iris binario | Random Subsampling (N=30) | Red neuronal (MLP) | ANOVA F-test | 0.5911 ± 0.0957 | 0.4911 ± 0.1298 | 0.4862 ± 0.0963 | 0.4709 ± 0.1035 | 0.8008 ± 0.1033 | -0.0238 ± 0.2215 | -0.0266 ± 0.2099 | 0.5640 ± 0.1235 | 0.6578 ± 0.0201 | 0.2327 ± 0.0099 | `[[961,239],[497,103]]` |
| Iris binario | Random Subsampling (N=30) | Red neuronal (MLP) | Sin selección | 0.6878 ± 0.0293 | 0.6266 ± 0.1062 | 0.5671 ± 0.0445 | 0.5456 ± 0.0704 | 0.9292 ± 0.0329 | 0.1854 ± 0.1247 | 0.1572 ± 0.1029 | 0.5903 ± 0.0641 | 0.6453 ± 0.0118 | 0.2262 ± 0.0056 | `[[1115,85],[477,123]]` |
| Iris binario | Random Subsampling (N=30) | Regresión logística | ANOVA F-test | 0.7067 ± 0.0403 | 0.6722 ± 0.0615 | 0.6296 ± 0.0440 | 0.6334 ± 0.0496 | 0.8608 ± 0.0625 | 0.2973 ± 0.0985 | 0.2816 ± 0.0938 | 0.7869 ± 0.0477 | 0.5153 ± 0.0376 | 0.1761 ± 0.0166 | `[[1033,167],[361,239]]` |
| Iris binario | Random Subsampling (N=30) | Regresión logística | Sin selección | 0.7211 ± 0.0450 | 0.6917 ± 0.0679 | 0.6512 ± 0.0491 | 0.6568 ± 0.0553 | 0.8608 ± 0.0665 | 0.3392 ± 0.1107 | 0.3251 ± 0.1052 | 0.8000 ± 0.0487 | 0.5108 ± 0.0411 | 0.1736 ± 0.0179 | `[[1033,167],[335,265]]` |
| Iris binario | Random Subsampling (N=30) | SVM lineal | ANOVA F-test | 0.7233 ± 0.0419 | 0.7033 ± 0.0743 | 0.6362 ± 0.0461 | 0.6414 ± 0.0540 | 0.8975 ± 0.0577 | 0.3311 ± 0.1110 | 0.3042 ± 0.1017 | 0.7874 ± 0.0485 | 0.5296 ± 0.0310 | 0.1793 ± 0.0136 | `[[1077,123],[375,225]]` |
| Iris binario | Random Subsampling (N=30) | SVM lineal | Sin selección | 0.7289 ± 0.0408 | 0.7045 ± 0.0647 | 0.6608 ± 0.0494 | 0.6657 ± 0.0532 | 0.8650 ± 0.0697 | 0.3606 ± 0.1043 | 0.3439 ± 0.1000 | 0.7918 ± 0.0515 | 0.5324 ± 0.0305 | 0.1803 ± 0.0135 | `[[1038,162],[326,274]]` |
| Iris binario | Random Subsampling (N=30) | Árbol de decisión | ANOVA F-test | 0.9222 ± 0.0271 | 0.9231 ± 0.0342 | 0.9038 ± 0.0308 | 0.9108 ± 0.0303 | 0.9592 ± 0.0380 | 0.8263 ± 0.0601 | 0.8221 ± 0.0604 | 0.9025 ± 0.0310 | 2.6947 ± 0.9336 | 0.0808 ± 0.0264 | `[[1151,49],[91,509]]` |
| Iris binario | Random Subsampling (N=30) | Árbol de decisión | Sin selección | 0.9389 ± 0.0253 | 0.9335 ± 0.0317 | 0.9321 ± 0.0270 | 0.9314 ± 0.0282 | 0.9525 ± 0.0362 | 0.8654 ± 0.0552 | 0.8631 ± 0.0562 | 0.9321 ± 0.0270 | 2.1107 ± 0.8728 | 0.0611 ± 0.0253 | `[[1143,57],[53,547]]` |
| Iris binario | Stratified Cross-Validation (K=10) | KNN | ANOVA F-test | 0.9200 ± 0.0757 | 0.9163 ± 0.0808 | 0.9100 ± 0.1075 | 0.9050 ± 0.0984 | 0.9400 ± 0.0516 | 0.8244 ± 0.1846 | 0.8128 ± 0.1921 | 0.9530 ± 0.0884 | 1.0397 ± 1.6070 | 0.0660 ± 0.0520 | `[[94,6],[6,44]]` |
| Iris binario | Stratified Cross-Validation (K=10) | KNN | Sin selección | 0.9533 ± 0.0450 | 0.9516 ± 0.0494 | 0.9550 ± 0.0438 | 0.9487 ± 0.0483 | 0.9500 ± 0.0707 | 0.9060 ± 0.0871 | 0.8983 ± 0.0955 | 0.9870 ± 0.0347 | 0.3124 ± 0.7285 | 0.0341 ± 0.0297 | `[[95,5],[2,48]]` |
| Iris binario | Stratified Cross-Validation (K=10) | Red neuronal (MLP) | ANOVA F-test | 0.6333 ± 0.0846 | 0.5673 ± 0.1912 | 0.5200 ± 0.0823 | 0.4999 ± 0.0942 | 0.8600 ± 0.1075 | 0.0736 ± 0.2373 | 0.0500 ± 0.1896 | 0.6120 ± 0.1655 | 0.6531 ± 0.0237 | 0.2304 ± 0.0116 | `[[86,14],[41,9]]` |
| Iris binario | Stratified Cross-Validation (K=10) | Red neuronal (MLP) | Sin selección | 0.6867 ± 0.0450 | 0.5793 ± 0.2349 | 0.5500 ± 0.0624 | 0.5052 ± 0.1021 | 0.9600 ± 0.0516 | 0.1576 ± 0.2053 | 0.1194 ± 0.1489 | 0.6040 ± 0.1791 | 0.6440 ± 0.0305 | 0.2255 ± 0.0145 | `[[96,4],[43,7]]` |
| Iris binario | Stratified Cross-Validation (K=10) | Regresión logística | ANOVA F-test | 0.7200 ± 0.0757 | 0.6577 ± 0.1556 | 0.6450 ± 0.1012 | 0.6369 ± 0.1299 | 0.8700 ± 0.0949 | 0.3166 ± 0.2208 | 0.3061 ± 0.2158 | 0.7940 ± 0.0828 | 0.5188 ± 0.0935 | 0.1773 ± 0.0392 | `[[87,13],[29,21]]` |
| Iris binario | Stratified Cross-Validation (K=10) | Regresión logística | Sin selección | 0.7200 ± 0.1209 | 0.6696 ± 0.2015 | 0.6550 ± 0.1462 | 0.6520 ± 0.1679 | 0.8500 ± 0.1080 | 0.3264 ± 0.3353 | 0.3223 ± 0.3132 | 0.8060 ± 0.0864 | 0.5161 ± 0.1066 | 0.1754 ± 0.0450 | `[[85,15],[27,23]]` |
| Iris binario | Stratified Cross-Validation (K=10) | SVM lineal | ANOVA F-test | 0.7333 ± 0.0770 | 0.6972 ± 0.1760 | 0.6400 ± 0.0966 | 0.6360 ± 0.1280 | 0.9200 ± 0.0789 | 0.3418 ± 0.2260 | 0.3122 ± 0.2141 | 0.8080 ± 0.1088 | 0.5289 ± 0.0877 | 0.1788 ± 0.0359 | `[[92,8],[32,18]]` |
| Iris binario | Stratified Cross-Validation (K=10) | SVM lineal | Sin selección | 0.7267 ± 0.1153 | 0.6852 ± 0.2003 | 0.6650 ± 0.1454 | 0.6582 ± 0.1640 | 0.8500 ± 0.1269 | 0.3516 ± 0.3329 | 0.3389 ± 0.3059 | 0.7950 ± 0.0868 | 0.5331 ± 0.0882 | 0.1816 ± 0.0382 | `[[85,15],[26,24]]` |
| Iris binario | Stratified Cross-Validation (K=10) | Árbol de decisión | ANOVA F-test | 0.9400 ± 0.0663 | 0.9477 ± 0.0542 | 0.9250 ± 0.0950 | 0.9268 ± 0.0882 | 0.9700 ± 0.0483 | 0.8692 ± 0.1478 | 0.8569 ± 0.1687 | 0.9190 ± 0.1113 | 2.0981 ± 2.3001 | 0.0691 ± 0.0710 | `[[97,3],[6,44]]` |
| Iris binario | Stratified Cross-Validation (K=10) | Árbol de decisión | Sin selección | 0.9467 ± 0.0526 | 0.9471 ± 0.0520 | 0.9400 ± 0.0699 | 0.9385 ± 0.0628 | 0.9600 ± 0.0516 | 0.8858 ± 0.1159 | 0.8780 ± 0.1240 | 0.9400 ± 0.0699 | 1.8421 ± 1.8163 | 0.0533 ± 0.0526 | `[[96,4],[4,46]]` |
| Mobile Price Classification | Entrenamiento/Prueba | KNN | ANOVA F-test | 0.5000 | 0.5142 | 0.5000 | 0.5044 | 0.8333 | 0.3345 | 0.3333 | 0.7351 | 5.2400 | 0.6684 | `[[13,8,3,1],[5,8,10,2],[2,8,11,4],[0,1,6,18]]` |
| Mobile Price Classification | Entrenamiento/Prueba | KNN | Sin selección | 0.4800 | 0.4968 | 0.4800 | 0.4851 | 0.8267 | 0.3082 | 0.3067 | 0.6979 | 5.3323 | 0.6981 | `[[14,7,4,0],[6,12,4,3],[1,7,10,7],[0,6,7,12]]` |
| Mobile Price Classification | Entrenamiento/Prueba | Red neuronal (MLP) | ANOVA F-test | 0.2500 | 0.2895 | 0.2500 | 0.2270 | 0.7500 | 0.0000 | 0.0000 | 0.5599 | 1.4414 | 0.7824 | `[[4,1,19,1],[10,1,10,4],[7,1,12,5],[5,0,12,8]]` |
| Mobile Price Classification | Entrenamiento/Prueba | Red neuronal (MLP) | Sin selección | 0.2700 | 0.2700 | 0.2700 | 0.2530 | 0.7567 | 0.0278 | 0.0267 | 0.5251 | 1.5861 | 0.8252 | `[[3,7,5,10],[3,11,3,8],[3,8,4,10],[2,11,3,9]]` |
| Mobile Price Classification | Entrenamiento/Prueba | Regresión logística | ANOVA F-test | 0.7300 | 0.7297 | 0.7300 | 0.7280 | 0.9100 | 0.6411 | 0.6400 | 0.9117 | 0.6724 | 0.3732 | `[[22,3,0,0],[1,18,6,0],[0,6,12,7],[0,0,4,21]]` |
| Mobile Price Classification | Entrenamiento/Prueba | Regresión logística | Sin selección | 0.6700 | 0.6608 | 0.6700 | 0.6597 | 0.8900 | 0.5635 | 0.5600 | 0.8739 | 0.7939 | 0.4413 | `[[20,3,2,0],[6,17,2,0],[0,10,9,6],[0,0,4,21]]` |
| Mobile Price Classification | Entrenamiento/Prueba | SVM lineal | ANOVA F-test | 0.7800 | 0.7956 | 0.7800 | 0.7786 | 0.9267 | 0.7131 | 0.7067 | 0.9639 | 0.4704 | 0.2647 | `[[20,5,0,0],[1,22,2,0],[0,7,14,4],[0,0,3,22]]` |
| Mobile Price Classification | Entrenamiento/Prueba | SVM lineal | Sin selección | 0.8000 | 0.8207 | 0.8000 | 0.7912 | 0.9333 | 0.7446 | 0.7333 | 0.9444 | 0.6095 | 0.3425 | `[[22,3,0,0],[2,23,0,0],[0,9,12,4],[0,0,2,23]]` |
| Mobile Price Classification | Entrenamiento/Prueba | Árbol de decisión | ANOVA F-test | 0.6800 | 0.6698 | 0.6800 | 0.6690 | 0.8933 | 0.5771 | 0.5733 | 0.8087 | 11.0524 | 0.6400 | `[[22,3,0,0],[4,15,5,1],[1,4,10,10],[0,0,4,21]]` |
| Mobile Price Classification | Entrenamiento/Prueba | Árbol de decisión | Sin selección | 0.6800 | 0.6647 | 0.6800 | 0.6670 | 0.8933 | 0.5768 | 0.5733 | 0.8080 | 11.0524 | 0.6400 | `[[22,3,0,0],[4,16,5,0],[1,5,9,10],[0,0,4,21]]` |
| Mobile Price Classification | Hold-Out 60/40 | KNN | ANOVA F-test | 0.6362 | 0.6467 | 0.6362 | 0.6403 | 0.8788 | 0.5156 | 0.5150 | 0.8584 | 2.2543 | 0.4824 | `[[139,57,4,0],[37,112,48,3],[2,45,112,41],[0,6,48,146]]` |
| Mobile Price Classification | Hold-Out 60/40 | KNN | Sin selección | 0.4900 | 0.5039 | 0.4900 | 0.4957 | 0.8300 | 0.3205 | 0.3200 | 0.7514 | 3.5965 | 0.6288 | `[[113,63,24,0],[53,71,63,13],[8,61,82,49],[1,20,53,126]]` |
| Mobile Price Classification | Hold-Out 60/40 | Red neuronal (MLP) | ANOVA F-test | 0.6512 | 0.6350 | 0.6513 | 0.6312 | 0.8838 | 0.5418 | 0.5350 | 0.8858 | 0.8483 | 0.4714 | `[[159,40,1,0],[83,67,47,3],[7,19,108,66],[0,0,13,187]]` |
| Mobile Price Classification | Hold-Out 60/40 | Red neuronal (MLP) | Sin selección | 0.6388 | 0.6187 | 0.6387 | 0.6172 | 0.8796 | 0.5247 | 0.5183 | 0.8612 | 0.9029 | 0.4957 | `[[172,27,1,0],[66,84,43,7],[10,47,74,69],[0,6,13,181]]` |
| Mobile Price Classification | Hold-Out 60/40 | Regresión logística | ANOVA F-test | 0.9575 | 0.9582 | 0.9575 | 0.9575 | 0.9858 | 0.9436 | 0.9433 | 0.9974 | 0.2214 | 0.1108 | `[[193,7,0,0],[1,197,2,0],[0,8,183,9],[0,0,7,193]]` |
| Mobile Price Classification | Hold-Out 60/40 | Regresión logística | Sin selección | 0.9688 | 0.9690 | 0.9688 | 0.9688 | 0.9896 | 0.9584 | 0.9583 | 0.9984 | 0.2158 | 0.1065 | `[[194,6,0,0],[1,195,4,0],[0,3,191,6],[0,0,5,195]]` |
| Mobile Price Classification | Hold-Out 60/40 | SVM lineal | ANOVA F-test | 0.9575 | 0.9585 | 0.9575 | 0.9575 | 0.9858 | 0.9437 | 0.9433 | 0.9980 | 0.1051 | 0.0610 | `[[192,8,0,0],[0,196,4,0],[0,9,182,9],[0,0,4,196]]` |
| Mobile Price Classification | Hold-Out 60/40 | SVM lineal | Sin selección | 0.9575 | 0.9577 | 0.9575 | 0.9575 | 0.9858 | 0.9434 | 0.9433 | 0.9985 | 0.1000 | 0.0560 | `[[192,8,0,0],[4,190,6,0],[0,5,189,6],[0,0,5,195]]` |
| Mobile Price Classification | Hold-Out 60/40 | Árbol de decisión | ANOVA F-test | 0.8150 | 0.8161 | 0.8150 | 0.8155 | 0.9383 | 0.7534 | 0.7533 | 0.8831 | 6.3897 | 0.3700 | `[[180,20,0,0],[17,150,33,0],[0,27,148,25],[0,0,26,174]]` |
| Mobile Price Classification | Hold-Out 60/40 | Árbol de decisión | Sin selección | 0.8125 | 0.8142 | 0.8125 | 0.8131 | 0.9375 | 0.7501 | 0.7500 | 0.8757 | 6.4760 | 0.3750 | `[[171,29,0,0],[19,154,27,0],[0,25,149,26],[0,0,24,176]]` |
| Mobile Price Classification | K-Fold (K=10) | KNN | ANOVA F-test | 0.6600 ± 0.0333 | 0.6627 ± 0.0301 | 0.6612 ± 0.0327 | 0.6600 ± 0.0305 | 0.8868 ± 0.0116 | 0.5472 ± 0.0448 | 0.5461 ± 0.0445 | 0.8785 ± 0.0155 | 1.9220 ± 0.2421 | 0.4433 ± 0.0303 | `[[390,103,7,0],[104,267,126,3],[5,116,280,99],[0,7,110,383]]` |
| Mobile Price Classification | K-Fold (K=10) | KNN | Sin selección | 0.5145 ± 0.0465 | 0.5276 ± 0.0471 | 0.5140 ± 0.0462 | 0.5178 ± 0.0459 | 0.8384 ± 0.0159 | 0.3534 ± 0.0625 | 0.3520 ± 0.0620 | 0.7600 ± 0.0261 | 3.9865 ± 0.5515 | 0.6249 ± 0.0352 | `[[318,142,40,0],[127,198,150,25],[30,144,215,111],[1,35,166,298]]` |
| Mobile Price Classification | K-Fold (K=10) | Red neuronal (MLP) | ANOVA F-test | 0.7745 ± 0.0321 | 0.7758 ± 0.0279 | 0.7756 ± 0.0227 | 0.7624 ± 0.0317 | 0.9250 ± 0.0105 | 0.7066 ± 0.0375 | 0.6991 ± 0.0413 | 0.9530 ± 0.0107 | 0.6643 ± 0.0259 | 0.3688 ± 0.0176 | `[[479,21,0,0],[149,274,76,1],[1,46,332,121],[0,0,36,464]]` |
| Mobile Price Classification | K-Fold (K=10) | Red neuronal (MLP) | Sin selección | 0.7395 ± 0.0215 | 0.7434 ± 0.0209 | 0.7409 ± 0.0182 | 0.7254 ± 0.0211 | 0.9130 ± 0.0067 | 0.6617 ± 0.0259 | 0.6520 ± 0.0271 | 0.9278 ± 0.0065 | 0.7402 ± 0.0293 | 0.4112 ± 0.0182 | `[[477,23,0,0],[159,276,61,4],[1,58,263,178],[0,1,36,463]]` |
| Mobile Price Classification | K-Fold (K=10) | Regresión logística | ANOVA F-test | 0.9665 ± 0.0165 | 0.9671 ± 0.0161 | 0.9665 ± 0.0166 | 0.9665 ± 0.0164 | 0.9888 ± 0.0056 | 0.9554 ± 0.0220 | 0.9552 ± 0.0221 | 0.9988 ± 0.0010 | 0.1903 ± 0.0119 | 0.0943 ± 0.0091 | `[[491,9,0,0],[9,478,13,0],[0,14,471,15],[0,0,7,493]]` |
| Mobile Price Classification | K-Fold (K=10) | Regresión logística | Sin selección | 0.9625 ± 0.0196 | 0.9629 ± 0.0194 | 0.9625 ± 0.0195 | 0.9623 ± 0.0195 | 0.9875 ± 0.0066 | 0.9501 ± 0.0262 | 0.9498 ± 0.0262 | 0.9983 ± 0.0014 | 0.1934 ± 0.0135 | 0.0975 ± 0.0103 | `[[486,14,0,0],[15,471,14,0],[0,10,476,14],[0,0,8,492]]` |
| Mobile Price Classification | K-Fold (K=10) | SVM lineal | ANOVA F-test | 0.9660 ± 0.0171 | 0.9664 ± 0.0170 | 0.9657 ± 0.0174 | 0.9657 ± 0.0173 | 0.9886 ± 0.0058 | 0.9547 ± 0.0228 | 0.9545 ± 0.0229 | 0.9987 ± 0.0011 | 0.0826 ± 0.0203 | 0.0476 ± 0.0163 | `[[487,13,0,0],[12,479,9,0],[0,13,472,15],[0,0,6,494]]` |
| Mobile Price Classification | K-Fold (K=10) | SVM lineal | Sin selección | 0.9570 ± 0.0181 | 0.9582 ± 0.0176 | 0.9571 ± 0.0177 | 0.9571 ± 0.0177 | 0.9856 ± 0.0061 | 0.9428 ± 0.0242 | 0.9425 ± 0.0243 | 0.9984 ± 0.0011 | 0.0951 ± 0.0180 | 0.0568 ± 0.0145 | `[[485,15,0,0],[15,466,19,0],[0,13,472,15],[0,0,9,491]]` |
| Mobile Price Classification | K-Fold (K=10) | Árbol de decisión | ANOVA F-test | 0.8435 ± 0.0113 | 0.8433 ± 0.0110 | 0.8430 ± 0.0123 | 0.8423 ± 0.0115 | 0.9478 ± 0.0039 | 0.7913 ± 0.0152 | 0.7908 ± 0.0152 | 0.8984 ± 0.0111 | 5.4053 ± 0.3908 | 0.3130 ± 0.0226 | `[[456,44,0,0],[53,395,52,0],[0,49,395,56],[0,0,59,441]]` |
| Mobile Price Classification | K-Fold (K=10) | Árbol de decisión | Sin selección | 0.8295 ± 0.0269 | 0.8291 ± 0.0290 | 0.8289 ± 0.0286 | 0.8281 ± 0.0287 | 0.9431 ± 0.0090 | 0.7725 ± 0.0366 | 0.7720 ± 0.0365 | 0.8902 ± 0.0218 | 5.8889 ± 0.9298 | 0.3410 ± 0.0538 | `[[446,54,0,0],[50,400,49,1],[0,55,376,69],[0,0,63,437]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | KNN | ANOVA F-test | 0.6765 | 0.6820 | 0.6765 | 0.6789 | 0.8922 | 0.5689 | 0.5687 | 0.8887 | 1.6691 | 0.4295 | `[[382,115,3,0],[89,281,127,3],[2,112,298,88],[0,7,101,392]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | KNN | Sin selección | 0.5185 | 0.5298 | 0.5185 | 0.5232 | 0.8395 | 0.3584 | 0.3580 | 0.7656 | 3.8599 | 0.6136 | `[[319,142,38,1],[131,196,144,29],[30,147,217,106],[3,42,150,305]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | Red neuronal (MLP) | ANOVA F-test | 0.8110 | 0.8120 | 0.8110 | 0.8052 | 0.9370 | 0.7518 | 0.7480 | 0.9630 | 0.6256 | 0.3456 | `[[482,18,0,0],[105,335,59,1],[0,44,338,118],[0,0,33,467]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | Red neuronal (MLP) | Sin selección | 0.7660 | 0.7680 | 0.7660 | 0.7558 | 0.9220 | 0.6947 | 0.6880 | 0.9419 | 0.6909 | 0.3863 | `[[476,24,0,0],[141,301,55,3],[0,57,289,154],[0,0,34,466]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | Regresión logística | ANOVA F-test | 0.9720 | 0.9720 | 0.9720 | 0.9720 | 0.9907 | 0.9627 | 0.9627 | 0.9990 | 0.1826 | 0.0899 | `[[493,7,0,0],[9,480,11,0],[0,13,476,11],[0,0,5,495]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | Regresión logística | Sin selección | 0.9660 | 0.9659 | 0.9660 | 0.9659 | 0.9887 | 0.9547 | 0.9547 | 0.9985 | 0.1860 | 0.0933 | `[[491,9,0,0],[13,475,12,0],[0,14,473,13],[0,0,7,493]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | SVM lineal | ANOVA F-test | 0.9715 | 0.9715 | 0.9715 | 0.9715 | 0.9905 | 0.9620 | 0.9620 | 0.9991 | 0.0729 | 0.0411 | `[[492,8,0,0],[9,482,9,0],[0,16,474,10],[0,0,5,495]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | SVM lineal | Sin selección | 0.9575 | 0.9574 | 0.9575 | 0.9574 | 0.9858 | 0.9434 | 0.9433 | 0.9983 | 0.0944 | 0.0557 | `[[490,10,0,0],[14,468,18,0],[0,18,465,17],[0,0,8,492]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | Árbol de decisión | ANOVA F-test | 0.8380 | 0.8376 | 0.8380 | 0.8377 | 0.9460 | 0.7841 | 0.7840 | 0.8988 | 5.5953 | 0.3240 | `[[457,43,0,0],[50,393,57,0],[0,44,386,70],[0,0,60,440]]` |
| Mobile Price Classification | Leave-One-Out (LOO) | Árbol de decisión | Sin selección | 0.8260 | 0.8259 | 0.8260 | 0.8259 | 0.9420 | 0.7680 | 0.7680 | 0.8896 | 6.0097 | 0.3480 | `[[446,53,1,0],[50,391,59,0],[0,49,378,73],[0,1,62,437]]` |
| Mobile Price Classification | Random Subsampling (N=30) | KNN | ANOVA F-test | 0.6400 ± 0.0186 | 0.6467 ± 0.0172 | 0.6400 ± 0.0186 | 0.6426 ± 0.0178 | 0.8800 ± 0.0062 | 0.5205 ± 0.0247 | 0.5201 ± 0.0248 | 0.8609 ± 0.0095 | 2.1879 ± 0.2476 | 0.4768 ± 0.0160 | `[[4482,1418,100,0],[1226,3216,1478,80],[77,1508,3249,1166],[0,101,1485,4414]]` |
| Mobile Price Classification | Random Subsampling (N=30) | KNN | Sin selección | 0.5074 ± 0.0175 | 0.5184 ± 0.0170 | 0.5074 ± 0.0175 | 0.5116 ± 0.0170 | 0.8358 ± 0.0058 | 0.3437 ± 0.0233 | 0.3432 ± 0.0233 | 0.7522 ± 0.0110 | 3.8914 ± 0.2771 | 0.6306 ± 0.0148 | `[[3686,1760,517,37],[1577,2358,1699,366],[416,1693,2526,1365],[48,503,1842,3607]]` |
| Mobile Price Classification | Random Subsampling (N=30) | Red neuronal (MLP) | ANOVA F-test | 0.6768 ± 0.0290 | 0.6673 ± 0.0318 | 0.6768 ± 0.0290 | 0.6600 ± 0.0303 | 0.8923 ± 0.0097 | 0.5758 ± 0.0389 | 0.5691 ± 0.0387 | 0.8933 ± 0.0179 | 0.8456 ± 0.0401 | 0.4668 ± 0.0220 | `[[5225,717,56,2],[2170,2479,1199,152],[143,787,3179,1891],[0,32,608,5360]]` |
| Mobile Price Classification | Random Subsampling (N=30) | Red neuronal (MLP) | Sin selección | 0.6244 ± 0.0473 | 0.6097 ± 0.0498 | 0.6244 ± 0.0473 | 0.6021 ± 0.0458 | 0.8748 ± 0.0158 | 0.5074 ± 0.0648 | 0.4992 ± 0.0630 | 0.8506 ± 0.0382 | 0.9313 ± 0.0725 | 0.5122 ± 0.0383 | `[[5128,749,104,19],[2165,2444,1033,358],[278,1122,2205,2395],[16,118,658,5208]]` |
| Mobile Price Classification | Random Subsampling (N=30) | Regresión logística | ANOVA F-test | 0.9605 ± 0.0080 | 0.9607 ± 0.0079 | 0.9605 ± 0.0080 | 0.9605 ± 0.0079 | 0.9868 ± 0.0027 | 0.9475 ± 0.0106 | 0.9474 ± 0.0106 | 0.9981 ± 0.0007 | 0.2199 ± 0.0065 | 0.1109 ± 0.0050 | `[[5870,130,0,0],[159,5688,153,0],[0,212,5621,167],[0,0,126,5874]]` |
| Mobile Price Classification | Random Subsampling (N=30) | Regresión logística | Sin selección | 0.9529 ± 0.0075 | 0.9530 ± 0.0074 | 0.9529 ± 0.0075 | 0.9528 ± 0.0075 | 0.9843 ± 0.0025 | 0.9373 ± 0.0100 | 0.9372 ± 0.0100 | 0.9973 ± 0.0006 | 0.2229 ± 0.0064 | 0.1143 ± 0.0049 | `[[5843,157,0,0],[212,5603,185,0],[0,230,5576,194],[0,0,153,5847]]` |
| Mobile Price Classification | Random Subsampling (N=30) | SVM lineal | ANOVA F-test | 0.9602 ± 0.0083 | 0.9604 ± 0.0082 | 0.9602 ± 0.0083 | 0.9601 ± 0.0083 | 0.9867 ± 0.0028 | 0.9471 ± 0.0110 | 0.9469 ± 0.0110 | 0.9983 ± 0.0006 | 0.1007 ± 0.0111 | 0.0574 ± 0.0081 | `[[5861,139,0,0],[160,5664,176,0],[0,187,5623,190],[0,0,103,5897]]` |
| Mobile Price Classification | Random Subsampling (N=30) | SVM lineal | Sin selección | 0.9524 ± 0.0072 | 0.9526 ± 0.0072 | 0.9524 ± 0.0072 | 0.9523 ± 0.0072 | 0.9841 ± 0.0024 | 0.9367 ± 0.0096 | 0.9366 ± 0.0096 | 0.9976 ± 0.0006 | 0.1154 ± 0.0094 | 0.0669 ± 0.0069 | `[[5835,165,0,0],[227,5561,212,0],[0,211,5597,192],[0,0,135,5865]]` |
| Mobile Price Classification | Random Subsampling (N=30) | Árbol de decisión | ANOVA F-test | 0.8310 ± 0.0144 | 0.8317 ± 0.0140 | 0.8310 ± 0.0144 | 0.8309 ± 0.0143 | 0.9437 ± 0.0048 | 0.7749 ± 0.0191 | 0.7747 ± 0.0192 | 0.8911 ± 0.0096 | 5.8371 ± 0.4976 | 0.3380 ± 0.0288 | `[[5419,581,0,0],[617,4669,711,3],[1,682,4605,712],[0,2,747,5251]]` |
| Mobile Price Classification | Random Subsampling (N=30) | Árbol de decisión | Sin selección | 0.8229 ± 0.0161 | 0.8234 ± 0.0157 | 0.8229 ± 0.0161 | 0.8227 ± 0.0161 | 0.9410 ± 0.0054 | 0.7642 ± 0.0213 | 0.7639 ± 0.0214 | 0.8857 ± 0.0103 | 6.1162 ± 0.5556 | 0.3542 ± 0.0322 | `[[5395,602,3,0],[665,4600,732,3],[3,713,4522,762],[0,3,764,5233]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | KNN | ANOVA F-test | 0.6865 ± 0.0420 | 0.6945 ± 0.0397 | 0.6865 ± 0.0420 | 0.6884 ± 0.0408 | 0.8955 ± 0.0140 | 0.5832 ± 0.0557 | 0.5820 ± 0.0560 | 0.8871 ± 0.0212 | 1.7025 ± 0.3377 | 0.4349 ± 0.0388 | `[[390,104,6,0],[90,295,114,1],[4,107,305,84],[0,5,112,383]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | KNN | Sin selección | 0.5170 ± 0.0324 | 0.5340 ± 0.0330 | 0.5170 ± 0.0324 | 0.5227 ± 0.0323 | 0.8390 ± 0.0108 | 0.3572 ± 0.0434 | 0.3560 ± 0.0432 | 0.7652 ± 0.0218 | 3.9602 ± 0.5418 | 0.6167 ± 0.0284 | `[[311,148,40,1],[128,202,146,24],[28,142,224,106],[4,45,154,297]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | Red neuronal (MLP) | ANOVA F-test | 0.7740 ± 0.0247 | 0.7789 ± 0.0251 | 0.7740 ± 0.0247 | 0.7615 ± 0.0304 | 0.9247 ± 0.0082 | 0.7070 ± 0.0298 | 0.6987 ± 0.0329 | 0.9483 ± 0.0103 | 0.6720 ± 0.0284 | 0.3739 ± 0.0192 | `[[477,23,0,0],[154,260,84,2],[2,40,341,117],[0,0,30,470]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | Red neuronal (MLP) | Sin selección | 0.7350 ± 0.0269 | 0.7352 ± 0.0295 | 0.7350 ± 0.0269 | 0.7198 ± 0.0311 | 0.9117 ± 0.0090 | 0.6553 ± 0.0345 | 0.6467 ± 0.0358 | 0.9270 ± 0.0118 | 0.7416 ± 0.0245 | 0.4123 ± 0.0159 | `[[472,28,0,0],[153,284,59,4],[1,75,251,173],[0,0,37,463]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | Regresión logística | ANOVA F-test | 0.9690 ± 0.0120 | 0.9697 ± 0.0117 | 0.9690 ± 0.0120 | 0.9689 ± 0.0119 | 0.9897 ± 0.0040 | 0.9589 ± 0.0159 | 0.9587 ± 0.0160 | 0.9988 ± 0.0008 | 0.1901 ± 0.0169 | 0.0941 ± 0.0115 | `[[492,8,0,0],[12,480,8,0],[0,15,472,13],[0,0,6,494]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | Regresión logística | Sin selección | 0.9630 ± 0.0075 | 0.9638 ± 0.0072 | 0.9630 ± 0.0075 | 0.9630 ± 0.0074 | 0.9877 ± 0.0025 | 0.9510 ± 0.0100 | 0.9507 ± 0.0100 | 0.9985 ± 0.0008 | 0.1923 ± 0.0167 | 0.0964 ± 0.0113 | `[[492,8,0,0],[14,474,12,0],[0,15,467,18],[0,0,7,493]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | SVM lineal | ANOVA F-test | 0.9715 ± 0.0111 | 0.9721 ± 0.0109 | 0.9715 ± 0.0111 | 0.9715 ± 0.0109 | 0.9905 ± 0.0037 | 0.9622 ± 0.0148 | 0.9620 ± 0.0148 | 0.9989 ± 0.0008 | 0.0794 ± 0.0178 | 0.0451 ± 0.0128 | `[[491,9,0,0],[9,484,7,0],[0,14,473,13],[0,0,5,495]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | SVM lineal | Sin selección | 0.9605 ± 0.0096 | 0.9616 ± 0.0090 | 0.9605 ± 0.0096 | 0.9604 ± 0.0095 | 0.9868 ± 0.0032 | 0.9477 ± 0.0126 | 0.9473 ± 0.0127 | 0.9986 ± 0.0009 | 0.0902 ± 0.0178 | 0.0520 ± 0.0129 | `[[487,13,0,0],[14,471,15,0],[0,13,471,16],[0,0,8,492]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | Árbol de decisión | ANOVA F-test | 0.8415 ± 0.0160 | 0.8442 ± 0.0171 | 0.8415 ± 0.0160 | 0.8415 ± 0.0163 | 0.9472 ± 0.0053 | 0.7896 ± 0.0214 | 0.7887 ± 0.0213 | 0.8958 ± 0.0131 | 5.4744 ± 0.5524 | 0.3170 ± 0.0320 | `[[446,54,0,0],[46,399,55,0],[0,53,391,56],[0,0,53,447]]` |
| Mobile Price Classification | Stratified Cross-Validation (K=10) | Árbol de decisión | Sin selección | 0.8340 ± 0.0197 | 0.8367 ± 0.0196 | 0.8340 ± 0.0197 | 0.8343 ± 0.0199 | 0.9447 ± 0.0066 | 0.7794 ± 0.0261 | 0.7787 ± 0.0263 | 0.8925 ± 0.0148 | 5.7334 ± 0.6801 | 0.3320 ± 0.0394 | `[[443,56,1,0],[38,401,61,0],[0,61,378,61],[0,1,53,446]]` |

## Hallazgos y conclusiones

Se usa **F1 macro medio** como criterio principal para comparar configuraciones porque da el mismo peso a cada clase. En los protocolos de una sola partición se presenta una medición; en Random Subsampling y K-Fold se presenta media ± desviación estándar entre repeticiones/folds. LOO se resume con las predicciones fuera de muestra de todas las observaciones. Las matrices incluidas en la tabla agregan las predicciones de prueba del protocolo; Random Subsampling cuenta cada observación una vez por repetición.

Las filas de Entrenamiento/Prueba usan muestras de prueba pequeñas (75 en Iris y 100 en Mobile), por lo que se leen como una comparación acordada y no como estimación más estable que la validación cruzada. Para decidir qué configuración generaliza mejor, también se informa el mejor resultado entre los protocolos con repeticiones o folds.

### Iris binario

La mayor media de F1 macro entre todos los protocolos corresponde a **Árbol de decisión**, con **ANOVA F-test**, bajo **Entrenamiento/Prueba**: F1 macro 0.9694, accuracy 0.9733, recall macro 0.9600 y ROC-AUC 0.9584. Su matriz agregada es `[[50,0],[2,23]]`.

Entre Random Subsampling, K-Fold, LOO y Stratified Cross-Validation, el mayor resultado es de **KNN**, con **Sin selección**, bajo **Stratified Cross-Validation (K=10)**: F1 macro 0.9487 ± 0.0483, accuracy 0.9533 ± 0.0450, recall macro 0.9550 y ROC-AUC 0.9870. La matriz de confusión agregada de sus predicciones fuera de muestra es `[[95,5],[2,48]]`.

Para esta configuración de validación cruzada, la sensibilidad de la clase positiva `versicolor` es 0.9600 (48 de 50 casos). La diagonal de la matriz muestra las clasificaciones correctas; los falsos negativos son los `versicolor` clasificados como `no versicolor`.

ANOVA aumentó la media de F1 macro en 4 de 30 comparaciones pareadas (misma partición y clasificador). La diferencia media fue -0.0248; el efecto depende del método y el modelo, por lo que la selección se conserva solo cuando los resultados la respaldan.


### Mobile Price Classification

La mayor media de F1 macro entre todos los protocolos corresponde a **Regresión logística**, con **ANOVA F-test**, bajo **Leave-One-Out (LOO)**: F1 macro 0.9720, accuracy 0.9720, recall macro 0.9720 y ROC-AUC 0.9990. Su matriz agregada es `[[493,7,0,0],[9,480,11,0],[0,13,476,11],[0,0,5,495]]`.

Entre Random Subsampling, K-Fold, LOO y Stratified Cross-Validation, el mayor resultado es de **Regresión logística**, con **ANOVA F-test**, bajo **Leave-One-Out (LOO)**: F1 macro 0.9720, accuracy 0.9720, recall macro 0.9720 y ROC-AUC 0.9990. La matriz de confusión agregada de sus predicciones fuera de muestra es `[[493,7,0,0],[9,480,11,0],[0,13,476,11],[0,0,5,495]]`.

Las cuatro clases tienen el mismo tamaño en el conjunto completo. La especificidad y el Brier score se calcularon macro uno-contra-el-resto y con suma de errores cuadráticos por observación, respectivamente.

ANOVA aumentó la media de F1 macro en 26 de 30 comparaciones pareadas (misma partición y clasificador). La diferencia media fue +0.0363; el efecto depende del método y el modelo, por lo que la selección se conserva solo cuando los resultados la respaldan.


Los [resultados agregados](Resultados/resultados.csv), las [métricas por partición](Resultados/resultados_por_particion.csv), las [matrices de confusión](Resultados/matrices_confusion.csv) y las [variables elegidas por ANOVA](Resultados/variables_anova_por_particion.csv) quedan en `Resultados/`. El script [`ejecutar_experimentos.py`](ejecutar_experimentos.py) reproduce el análisis con semilla 42 y las particiones guardadas.
