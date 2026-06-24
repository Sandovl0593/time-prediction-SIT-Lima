# Pipeline persistente y reproducible — Plan de implementación (revisado)

Objetivo: dejar el pipeline con artefactos persistentes y reproducibles en disco (logs, JSON, CSV y `.pt`) para alimentar a un documento TEX con resultados reales en la experimentación del proyecto, evitando duplicación innecesaria de salidas. Para funciones existentes solo si están bien hechas se mantienen, pero los faltantes en la estructura del plan sí serán integradas.

## Decisión clave sobre rutas

La generación de rutas debe consolidarse en **un solo CSV maestro** por corrida, en lugar de repetir prácticamente las mismas filas para cada configuración A/B/C y cambiar solo algunas métricas.

### Propuesta

* Generar un archivo base, por ejemplo:

  * `src/outputs/routes/route\_candidates.csv`
* Cada fila representa una ruta candidata o subtramo, con columnas como:

  * `scenario\_id`
  * `target\_km`
  * `km\_tolerance`
  * `curve\_penalty`
  * `accepted\_by\_tolerance`
  * `straightness\_index`
  * `score`
  * `rank\_global`
  * `rank\_within\_scenario`
  * `line`
  * `start\_idx`
  * `end\_idx`
  * `length\_real\_km`
  * `length\_straight\_km`
  * `geometry\_wkt`

### Consecuencia

* Las configuraciones A/B/C no necesitan exportar tres CSV distintos con filas repetidas.
* Las métricas por configuración se obtienen por filtrado o agrupación desde el CSV maestro.
* Las rutas rectas se exportan una sola vez como subconjunto derivado:

  * `src/outputs/routes/straight\_routes.csv`

### Qué se deja de usar

* `export\_top\_segments` deja de ser parte del flujo principal.
* Si hace falta visualización top-k, se obtiene desde el CSV maestro con filtros, no con un exportador separado.

\---

## User Review

> \[!IMPORTANT]
> El pipeline actual genera métricas de rutas casi iguales en varias carpetas, cambiando solo valores como aceptación, score y ranking según `target\_km`, `km\_tolerance` y `curve\_penalty`. Para evitar duplicación, conviene producir un \*\*CSV maestro único\*\* y luego derivar de él:
> - estadísticas por configuración,
> - subconjunto de rutas rectas,
> - rankings globales,
> - comparaciones entre escenarios.

> \[!IMPORTANT]
> Las configuraciones A/B/C del TEX pueden seguir existiendo, pero como \*\*vistas/agrupaciones\*\* del mismo conjunto de rutas, no como archivos independientes con el mismo contenido repetido.

> \[!IMPORTANT]
> Si ya existen CSV limpios o artefactos intermedios, no reescribirlos innecesariamente. Solo regenerar reportes faltantes o inconsistentes.

\---

## Proposed Changes

### 1\) Configuración centralizada

#### \[MODIFY] `src/config.py`

Añadir lo necesario sin duplicar lo existente:

* `ROUTE\_CONFIGS`: diccionario con las configuraciones A/B/C.
* `straightness\_threshold`: umbral para seleccionar rutas rectas o casi rectas.
* `routes\_output\_dir` y `training\_output\_dir`.
* Validaciones en `\_\_post\_init\_\_`:

  * `train\_ratio + val\_ratio + test\_ratio ≈ 1.0`
  * `epochs > 0`, `hidden\_dim >= 2`, `learning\_rate > 0`, `patience > 0`
  * `seed` entero
  * `device` en `{"cpu", "cuda"}`

\---

### 2\) Preprocesamiento — logging y reportes

#### \[MODIFY] `src/utils/others.py`

* Extender `get\_logger` para aceptar `log\_file` y añadir `FileHandler`.

#### \[MODIFY] `src/main\_process/cleaning.py`

* Añadir log a archivo para limpieza.
* Si los CSV limpios ya existen, regenerar únicamente el reporte de calidad a partir de esos CSV.

#### \[MODIFY] `src/main\_process/general\_pipeline.py`

* Registrar logs a archivo durante todo el pipeline.
* Guardar:

  * `stations\_cleaning\_report.json`
  * `stop\_times\_coverage.json`
  * `graph\_summary.json`
* Medir tiempo total de ejecución del pipeline.
* Mantener el `MultiDiGraph` y exportarlo fielmente a CSV.

\---

### 3\) Construcción del grafo — multigrafo fiel

#### Sin cambios mayores

La reconstrucción del `MultiDiGraph` debe conservar:

* `key`
* `route\_id`
* `line`
* `observed`
* `spatial`
* `travel\_time\_s`
* `length\_m`

#### Ajuste importante

La carga hacia PyG debe leer el CSV del grafo sin colapsar aristas paralelas.  
El trainer debe consumir el grafo ya procesado, no volver a construir la lógica de negocio.

\---

### 4\) Rutas — CSV maestro único y derivados

#### \[MODIFY] `src/main\_process/general\_pipeline.py`

* Reemplazar la lógica de exportar un CSV distinto por configuración con una salida maestra:

  * `src/outputs/routes/route\_candidates.csv`
* Incluir en cada fila:

  * parámetros del escenario,
  * métricas geométricas,
  * score,
  * aceptación,
  * ranking,
  * geometría WKT.

#### \[NEW] `src/routes/route\_analysis.py` o módulo equivalente

* Cargar el CSV maestro.
* Generar vistas derivadas:

  * estadísticas por configuración,
  * distribución de longitudes y tiempos,
  * subconjunto de rutas rectas,
  * análisis por bins de curvatura,
  * ranking de mejores a peores rutas.

#### Exportaciones sugeridas

* `src/outputs/routes/route\_candidates.csv`
* `src/outputs/routes/routes\_summary.csv`
* `src/outputs/routes/routes\_distribution.csv`
* `src/outputs/routes/straight\_routes.csv`
* `src/outputs/routes/straight\_routes\_by\_config.json`

\---

### 5\) Training — persistencia de artefactos

#### \[MODIFY] `src/train/trainer.py`

* Crear carpeta por corrida y guardar:

  * `training.log`
  * `history.csv`
  * `final\_metrics.json`
  * `test\_predictions.csv`
  * `best\_model.pt`
* Registrar por época:

  * `epoch`
  * `train\_loss`
  * `val\_loss` o `val\_mse`
  * `val\_mae`
  * `val\_rmse`
  * `val\_r2`
* Usar `val\_ratio` desde `Config`.
* No usar `test` para selección del mejor modelo.

\---

### 6\) Documentación — mapping TEX → artefactos

#### \[NEW] `src/outputs/TEX\_ARTIFACTS\_MAP.md`

Documentar qué archivo alimenta cada tabla o figura del TEX.

Propuesta de mapeo:

|TEX|Archivo fuente|
|-|-|
|`tab:reporte\_limpieza`|`src/data/processed/metrics/stations\_cleaning\_report.json`|
|`tab:reporte\_tiempos`|`src/data/processed/metrics/stop\_times\_coverage.json`|
|`tab:configs\_rutas`|`src/outputs/routes/routes\_summary.csv`|
|`tab:dist\_rutas`|`src/outputs/routes/routes\_distribution.csv`|
|`tab:metricas\_general`|`src/outputs/training/<model>/final\_metrics.json`|
|`tab:metricas\_configs`|`src/outputs/routes/routes\_summary.csv` + métricas de entrenamiento|
|`tab:metricas\_rectas`|Entrenamiento sobre `straight\_routes.csv`|
|`tab:rmse\_por\_penalidad`|Predicciones + análisis por bins|
|`fig:graph\_output`|Visualización exportada del grafo procesado|

\---

## Huecos que todavía conviene cerrar en el repo

1. El flujo de rutas sigue orientado a carpetas separadas por escenario; debe pasar a un CSV maestro.
2. `export\_top\_segments` ya no es necesario si la visualización usa filtros sobre el CSV maestro.
3. La comparación entre A/B/C debe salir por agrupación, no por duplicación de filas.
4. Falta una capa explícita de análisis de rutas para producir estadísticas derivadas.
5. La trazabilidad del training todavía necesita logs y artefactos por corrida.

\---

## Verification Plan

### Preprocesamiento

* Confirmar que se crean:

  * `stations\_cleaning\_report.json`
  * `stop\_times\_coverage.json`
  * `graph\_summary.json`
  * `preprocessing.log`

### Rutas

* Confirmar que existe un solo `route\_candidates.csv`.
* Verificar que `routes\_summary.csv` y `routes\_distribution.csv` se derivan correctamente de ese CSV.
* Verificar que `straight\_routes.csv` contiene solo el subconjunto filtrado por rectitud.

### Training

* Confirmar que se crean:

  * `training.log`
  * `history.csv`
  * `final\_metrics.json`
  * `test\_predictions.csv`
  * `best\_model.pt`

### TEX

* Confirmar que cada tabla del capítulo de resultados apunta a un archivo fuente real y no a valores mock.

