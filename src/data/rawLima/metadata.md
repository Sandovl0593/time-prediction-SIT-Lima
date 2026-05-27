## Diccionario de datos — rawLima (generado 2026-05-02)

#### 1. Estaciones del BRT/Metropolitano
Archivo: stations_BRT_Metrop.json (JSON: array de objetos)

Campos:
- `estacion`: `string`. Nombre de la estación.
- `distrito`: `string`. Distrito ubicado.
- `ubicación`: `string`. Formato `[av1] / [av2]`, donde `av1` es la avenida incidente y `av2`, la avenida o zona de cruce (cercana).
- `activo`: `boolean`. Indica si la estación está operativa.
- `coordenadas`: `string`, par (lat,lon) en grados decimales.
- `rutas_regulares`: `array[string]`. Rutas regulares. Formato `[r]_[i]`, donde `r` puede ser A, B, C o D, e `i`, la posición de la ruta regular.
- `expresos_ns`: `array[string]`. Servicios expreso que pasan la estación de norte a sur. Formato: `[expr]_[i]`, dond `expr` es E1 al E13: Expresos 1 al 13, SE: Superexpreso, SEN: Superexpreso norte o L: Lechucero, e `i`, la posición de la ruta del expreso.
- `expresos_sn`: `array[string]`. Servicios expreso que pasan la estación de sur a norte, mismo formato de `expreso_ns`.



#### 2. Estaciones de buses corredores y alimentadores
Archivos: stations_aliment_Metrop.csv, stations_corredor_azul.csv, stations_corredor_morado.csv y stations_corredor_rojo.csv (CSV separado por ";")

Campos:
- `paradero`: `string`. Nombre del paradero.
- `corredor`: `string`. Código del bus alimnetador o corredor.
- `ida`: `boolean`. Si es parte de la recorrido de ida.
- `vuelta`: `boolean`. Si es parte de la recorrido de vuelta.
- `coordenadas`: `string`, par (lat,lon) en grados decimales.


#### 3. Estaciones de los servicios de trenes de Metro de Lima
Descripción: stations_Linea1.csv, stattions_linea2.csv (CSV separado por ";")

Campos:
- `estacion`: `string`. Nombre de la estación.
- `distrito`: `string`. Distrito ubicado.
- `ubicacion`: `string`. Formato `[av1] / [av2]`, donde `av1` es la avenida incidente y `av2`, la avenida de cruce (cercana).
- `coordenadas`: `string`, par (lat,lon) en grados decimales.
