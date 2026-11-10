# Particiones guardadas

Los `row_id` son índices desde cero y corresponden al orden de las filas en cada CSV de `Datos preparados`; excluye `row_id` de las variables predictoras. En Iris, `target=1` significa versicolor y `target=0` significa no versicolor. En Mobile, `target` conserva `price_range` (0 a 3).

- `holdout_60_40.csv` y `train_test_*.csv`: `partition` indica `train`, `test` o `unused`.
- `random_subsampling_30.csv`: cada `repetition` contiene la asignación train/test de una repetición.
- `kfold_10.csv` y `stratified_cross_validation_10.csv`: `test_fold` identifica el fold de prueba; en cada fold, las demás filas forman el entrenamiento.
- `leave_one_out.csv`: cada fila es una observación de prueba en el fold indicado por `test_fold`; las demás observaciones forman el entrenamiento.
- `resumen_particiones.csv`: tamaños de entrenamiento/prueba, número de folds o repeticiones y semilla.

Mobile usa únicamente `train.csv` para estas particiones porque el `test.csv` original no tiene etiquetas.
