# time-prediction-SIT-Lima

Predicción de tiempos de viaje en la red de Sistema Integrado de Transporte de Lima (SIT) usando modelos basados en grafos y secuencias temporales.

## Preprocesamiento — NYC MTA (load_nyc_mta)

Descripción detallada del preprocesamiento específico para el dataset NYC MTA Subway.

Contexto
- El módulo `src/data/load_nyc_mta.py` procesa datos tipo GTFS del sistema de metro de Nueva York (MTA Subway). La red incluye múltiples líneas, estaciones complejas (intercambios), servicios en ambas direcciones y registros temporales (stop_times) que permiten estimar tiempos observados entre estaciones.

Relación teórica con GeoPandas y NetworkX
- GeoPandas: se usa para representar las estaciones y las trazas de servicio como GeoDataFrames (`Point` y `LineString`). GeoPandas maneja CRS y operaciones geométricas; aquí se carga en EPSG:4326 (WGS84) y se reproyecta a EPSG:3857 para calcular distancias métricas (columnas `_x`, `_y`).
- NetworkX: la topología de la red se modela como un `nx.MultiDiGraph` (grafos dirigidos con aristas paralelas). NetworkX almacena atributos por nodo/arista que luego emplean los encoders GNN (por ejemplo GAT) y el decoder MLP (entrada: `z_src || z_dst || edge_attr` → salida escalar).

Importancia
- Normalizar y proyectar correctamente las geometrías permite calcular longitudes reales por arista (`length_m`) y construir features espaciales consistentes.
- Construir un grafo estructural y añadir tiempos observados (`travel_time_s`) proporciona la señal supervisada necesaria para tareas de regresión de tiempos de viaje.

Etapas y submódulos (resumen funcional)

1) `_parse_bool_like(v)`
	- Helper para interpretar valores booleanos en CSV (acepta `"1"|"true"|"si"|"y"`, `1`, `True`, etc.).

2) `_load_nodes_from_csv(path)`
	- Entrada: CSV de estaciones (mínimo `GTFS Stop ID`, `GTFS Latitude`, `GTFS Longitude`).
	- Salida: `gdf_nodes` (GeoDataFrame, CRS EPSG:4326) con columnas derivadas `_x`, `_y` (EPSG:3857), `is_cbd` y `geometry`.
	- Notas: reproyecta a EPSG:3857 para obtener coordenadas métricas y añade atributos útiles para el grafo.

3) `_build_lines_from_nodes(gdf_nodes)`
	- Agrupa por columna `Line`, ordena estaciones (algoritmo basado en PCA sobre coordenadas proyectadas) y crea `LineString` por servicio.
	- Salida: `gdf_lines` (GeoDataFrame) y `line_orders` (lista ordenada usada para construir aristas espaciales).
	- Advertencia: el ordering por PCA funciona bien para trayectos aproximadamente lineales; si dispone de `stop_sequence` de GTFS, es más fiable.

4) `_build_graph_from_nodes_and_lines(gdf_nodes, line_orders)`
	- Crea `nx.MultiDiGraph`, añade nodos con atributos (`lon`,`lat`,`x`,`y`,`borough`,`is_cbd`,`lines`,`structure`,`geometry`).
	- Añade aristas espaciales bidireccionales entre estaciones consecutivas con atributos: `line`, `length_m`, `geometry` y flag `spatial=True`.
	- Razonamiento: el grafo resultante representa la conectividad física del metro y sirve como base para GNNs.

5) `_process_stop_times(stop_times_path, ...)`
	- Si se proporcionan `stop_times` (o se detectan en la misma carpeta), parsea tiempos de llegada/salida por viaje, empareja par-a-par estaciones consecutivas y calcula tiempos observados promedio por arista (`travel_time_s`).
	- Maneja inconsistencias de `stop_id` (p. ej. sufijos alfabéticos) con una función de mapeo robusta.
	- Añade aristas con `observed=True` y `route_id` cuando aplica; calcula también `length_m` si no existe.

6) `save_processed_graph(G, gdf_nodes, gdf_lines, out_dir)`
	- Serializa `nodes.csv`, `edges.csv`, `lines.csv` en `out_dir` (por defecto `src/data/processed`), exportando geometrías como WKT (`geometry_wkt`) para facilitar recarga.

7) `load_processed_gdfs(processed_dir)`
	- Reconstruye GeoDataFrames y el grafo a partir de los CSVs procesados (útil para evitar reprocesar raw cada vez).

8) `visualize_nodes_edges(...)`
	- Herramienta rápida para inspeccionar mapas de nodos y líneas con Matplotlib/GeoPandas.

9) `load_nyc_mta(path, stop_times_path=None, trips_path=None)`
	- Orquestador: ejecuta las etapas anteriores, intenta autodetectar `stop_times` y `trips` en el mismo directorio, procesa y guarda los CSVs en `src/data/processed`.
	- Nota: imprime resumen (`num_nodes`, `num_edges`) y guarda los artefactos procesados.

Formato esperado de entrada (mínimo)
```
GTFS Stop ID,GTFS Latitude,GTFS Longitude
Line (opcional),Borough (opcional),CBD (opcional),Structure (opcional)
```

Salidas principales
- `src/data/processed/nodes.csv`  — atributos de nodos + `geometry_wkt`
- `src/data/processed/edges.csv`  — aristas (u,v,key,atributos..., `geometry_wkt`)
- `src/data/processed/lines.csv`  — geometrías de servicio por `Line`

Buena prácticas y advertencias
- Verifique encoding y valores nulos en los CSVs originales; convertir identificadores a `str` evita problemas de matching.
- La reproyección a EPSG:3857 es adecuada para distancias a escala metropolitana; para análisis global use reproyección apropiada.
- El ordering por PCA es heurístico; si dispone de `stop_sequence` en GTFS, es preferible usarlo para ordenar paradas por viaje.
- `stop_times` debe contener columnas de trip/stop y tiempos (arrival/departure) en formato numérico o convertible; si no existen, el módulo solo construye la geometría/estructura.

Ejemplo de uso corto
```python
from src.data.load_nyc_mta import load_nyc_mta, load_processed_gdfs

# Procesar raw GTFS-like (genera archivos en src/data/processed)
load_nyc_mta("data/rawNYC/MTA_Subway_Stations.csv")

# Cargar procesados posteriormente
gdf_nodes, gdf_lines = load_processed_gdfs("src/data/processed")
```

## Modelos

Este repositorio considera los siguientes encoders espaciales. Todos los
modelos incluyen un decoder MLP a nivel de arista (entrada: `z_src || z_dst || edge_attr` → salida escalar),
listo para tareas de regresión de tiempo de viaje.

| Modelo | Descripción | Tipo |
|--------|-------------|------|
| `gat` | Graph Attention Network (GAT) encoder + edge-level MLP decoder | Espacial |
| `gatv2` | GATv2 encoder + edge-level MLP decoder | Espacial |
| `graphsage` | GraphSAGE encoder + edge-level MLP decoder | Espacial |

## Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar un modelo
python run.py --model gat --epochs 50
python run.py --model gatv2 --epochs 50
python run.py --model graphsage --epochs 50

# Personalizar hiperparámetros
python run.py --model gat --hidden_dim 128 --num_layers 3 --lr 0.0005
```

## Métricas

- MSE, RMSE, MAE, MAPE (%), R²
