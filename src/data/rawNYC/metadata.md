## Diccionario de datos — rawNYC (generado 2026-05-29)

Resumen: documentación de los CSV presentes en `src/data/rawNYC` usados por `src/data/load_nyc_mta.py`.

#### 1. Estaciones MTA
Archivo: `MTA_Subway_Stations.csv` (CSV, separador `,`)

Campos (cabecera observada en el repo):
- `GTFS Stop ID`: `string`. Identificador de parada (puede incluir letras). Usado como `node_id` al construir el grafo; se recomienda mantenerlo como texto.
- `Station ID`: `string|int`. ID alternativo de estación.
- `Complex ID`: `string|int`. Identificador de complejo (agrupa paradas que constituyen un intercambiador).
- `Division`: `string`. División/operador (p. ej. `IRT`, `BMT`, `IND`, `SIR`).
- `Line`: `string`. Nombre del servicio/linea (se agrupa por esta columna para construir `LineString` por servicio).
- `Stop Name`: `string`. Nombre legible de la parada.
- `Borough`: `string`. Borough/municipio (Manhattan, Brooklyn, Bronx, Queens, SI).
- `CBD`: `boolean-like`. Marca si la estación está en CBD; puede aparecer como `true/false`, `1/0`, `si/no` — `_parse_bool_like` normaliza estos valores.
- `Daytime Routes`: `string`. Servicios diurnos asociados (ej. `1 2 3` o `A C E`).
- `Structure`: `string`. Tipo de estructura (`Subway`, `Elevated`, `Open Cut`, `At Grade`, `Viaduct`, etc.).
- `GTFS Latitude`, `GTFS Longitude`: `float`. Coordenadas en grados decimales (EPSG:4326/WGS84).
- `North Direction Label`, `South Direction Label`: `string`. Etiquetas de sentido.
- `ADA`, `ADA Northbound`, `ADA Southbound`: `int|boolean-like`. Indicadores de accesibilidad.
- `ADA Notes`: `string`. Notas complementarias sobre accesibilidad.
- `Georeference`: `string` (WKT). Ej: `POINT (-73.987495 40.75529)`. Contiene la geometría WKT cuando está presente.

Notas y uso en `load_nyc_mta.py`:
- El loader crea un `GeoDataFrame` a partir de `GTFS Latitude/Longitude` y genera una geometría `Point` (CRS EPSG:4326). Luego reproyecta a EPSG:3857 para crear columnas métricas `_x` y `_y` usadas para cálculos de distancia.
- La columna `Line` se agrupa y se ordena (heurísticamente por PCA sobre coordenadas proyectadas) para construir `LineString` por servicio; si dispone de `stop_sequence` en su GTFS, úselo para ordenar (más fiable que PCA).
- El campo `GTFS Stop ID` debe mantenerse como `string` para evitar problemas de matching; el loader contiene una función que intenta mapear sufijos (`701S` → `701`) mediante regex cuando es necesario.

#### 2. Tiempos por parada (stop_times)
Archivo: `stop_times.csv` (CSV, separador `,`)

Campos observados:
- `trip_uid` (o `trip_id`): `string`. Identificador del viaje/recorrido.
- `stop_id`: `string`. Identificador de parada (debe coincidir con `GTFS Stop ID` del CSV de estaciones, o ser coincidente tras normalizar sufijos alfabéticos).
- `track`: `string` (opcional). Vía/andén.
- `arrival_time` (o `arrival`): `numeric` (epoch seconds) o vacía.
- `departure_time` (o `departure`): `numeric` (epoch seconds) o vacía.
- `last_observed`: `numeric` (epoch seconds) — instante de la última observación.
- `marked_past`: `numeric` (timestamp) o flag auxiliar.

Notas importantes:
- En este repo los tiempos están expresados como segundos epoch (Unix timestamp). `load_nyc_mta.py` intenta convertir columnas de tiempo a numérico con `pd.to_numeric(..., errors='coerce')` — si sus datos usan formato `HH:MM:SS` deben preconvertirse a segundos o adaptar el loader.
- El módulo usa `arrival_time` (o `departure_time`) para ordenar registros por viaje y calcular diferencias entre paradas consecutivas, acumulando tiempos observados por arista.
- `stop_id` puede contener sufijos (ej. `701S`, `127N`); el loader aplica una heurística regex `^(
)[A-Za-z]+$` para mapear al ID base cuando es apropiado.

#### 3. Información de viaje (trips)
Archivo: `trip_times.csv` (CSV, separador `,`)

Campos observados:
- `trip_uid`, `trip_id`: `string`. Identificador del viaje (puede coincidir con el usado en `stop_times`).
- `route_id`: `string`. Identificador de la ruta/servicio asociado al `trip` (ej. `7`, `GS`, `A`). Se usa para etiquetar aristas observadas con `route_id`.
- `direction_id`: `int` (0/1). Sentido del viaje.
- `start_time`: `numeric` (epoch seconds). Inicio estimado del viaje.
- `vehicle_id`: `string`. Identificador del vehículo.
- `last_observed`, `marked_past`: `numeric` (timestamps) — metadatos de observación.
- `num_updates`, `num_schedule_changes`, `num_schedule_rewrites`: `int` — contadores de cambios/actualizaciones.

Uso en `load_nyc_mta.py`:
- Si `trip_times.csv` está presente y contiene `trip_id`/`trip_uid` y `route_id`, el loader construye un mapa `trips_map` para anotar observaciones con `route_id` (útil para agrupar tiempos por servicio).

#### 4. Formatos, calidad y recomendaciones
- Encoding: UTF-8 recomendado.
- Separador: `,` (coma) en los archivos observados en este repo.
- Coordenadas: usar grados decimales en EPSG:4326; el pipeline reproyecta a EPSG:3857 para distancias métricas.
- Tiempos: el loader espera valores numéricos (epoch seconds). Si usa otro formato, convierta previamente o extienda el parser.
- Identificadores: mantenga `GTFS Stop ID` como texto para evitar pérdidas de ceros o casting involuntario.
- Localización de ficheros: se espera que estos CSV se ubiquen en `src/data/rawNYC/`. `load_nyc_mta()` intentará autodetectar `stop_times` y `trips` en el mismo directorio si no se proporcionan explícitamente.

#### 5. Salidas relacionadas (procesado)
- Al ejecutar `load_nyc_mta()` el código guarda artefactos procesados en `src/data/processed/`:
  - `nodes.csv` — atributos de nodos (incluye `geometry_wkt`).
  - `edges.csv` — aristas con atributos (`u`,`v`,`key`, `length_m`, `travel_time_s` cuando existe, `geometry_wkt`, flags `spatial`/`observed`, ...).
  - `lines.csv` — geometrías `LineString` por servicio.