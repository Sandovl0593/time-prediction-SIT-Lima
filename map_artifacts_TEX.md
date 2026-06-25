# Reporte → LaTeX: Trazabilidad de artefactos

Documento de referencia para reproducir las tablas y figuras del paper.
Cada entrada mapea una etiqueta LaTeX al dato exacto que la alimenta,
el archivo fuente donde vive y la función/comando que lo genera.

---

## Definiciones base

| Símbolo | Definición | Columna en datos |
|---|---|---|
| $r$ | Índice de curvatura: $r = 1 / \text{straightness\_index}$ | derivado de `straightness_index` |
| `straightness_index` | $L_{\text{chord}} / L_{\text{real}} \in (0, 1]$ | columna en `route_candidates.csv`, `straight_routes.csv`, `config_{X}.csv`, `eval_*/predictions.csv` |
| "ruta directa (1-hop)" | Segmento entre dos estaciones consecutivas: geometry_wkt con exactamente 2 puntos coordenados | filtrado en `_build_subset_mask()` |
| `labeled_mask` | Aristas con `travel_time_s` no-NaN en `edges.csv` | aplicado en `_load_processed_graph_as_pyg()` y en `evaluate_on_subset()` |
| `eval_mask` | `test_mask ∩ labeled_mask ∩ in_subset` | construido en `evaluate_on_subset()` |

---

## Sección: Preprocesamiento

### `tab:reporte_limpieza`
| Campo | Fuente |
|---|---|
| Datos de limpieza de estaciones | `src/data/processed/metrics/stations_cleaning_report.json` |
| Generado por | `clean_stations()` en `src/main_process/cleaning.py` |
| Comando | `python run.py --process-nyc` |

### `tab:reporte_tiempos`
| Campo | Fuente |
|---|---|
| Cobertura de stop_times | `src/data/processed/metrics/stop_times_coverage.json` |
| Generado por | `clean_stop_times()` en `src/main_process/cleaning.py` |
| Comando | `python run.py --process-nyc` |

**Otros artefactos de preprocesamiento:**

| Artefacto | Ruta |
|---|---|
| Resumen del grafo | `src/data/processed/metrics/graph_summary.json` |
| Log de preprocesamiento | `src/data/processed/metrics/preprocessing.log` |
| Manifest de fuentes | `src/data/processed/manifests/manifest.json` |

---

## Sección: Rutas

### `tab:configs_rutas` / `tab:dist_rutas`
| Campo | Fuente |
|---|---|
| Resumen por escenario | `src/outputs/routes/routes_summary.csv` |
| Distribución por bin de km | `src/outputs/routes/routes_distribution.csv` |
| CSV maestro (origen) | `src/outputs/routes/route_candidates.csv` |
| Generado por | `run_route_analysis()` en `src/routes/route_analysis.py` |
| Comando | `python run.py -r` |

**Otros artefactos de rutas:**

| Artefacto | Ruta |
|---|---|
| Rutas filtradas por rectitud | `src/outputs/routes/straight_routes.csv` |
| Segmentos por config A/B/C | `src/topsegments/config_A.csv`, `config_B.csv`, `config_C.csv` |
| Resumen de exportación de configs | `src/topsegments/config_export_summary.json` |

---

## Sección: Entrenamiento

Cada corrida de entrenamiento crea:
```
src/outputs/training/<model>/<YYYYMMDD_HHMMSS>/
    training.log
    history.csv               — epoch, train_loss, val_mse, val_mae, val_rmse, val_r2
    final_metrics.json        — mse, rmse, mae, mape, r2, n_test, model, elapsed_seconds, ...
    test_predictions.csv      — edge_idx, pred, target
    best_model.pt
    eval_config_A/
        metrics.json          — subset_name, n_eval, mse, rmse, mae, mape, r2
        predictions.csv       — edge_idx, u, v, pred, target, straightness_index, tol_prox, km_offset, line
    eval_config_B/  (misma estructura)
    eval_config_C/  (misma estructura)
    eval_straight/  (misma estructura)
```

Comando de entrenamiento completo (1 encoder):
```bash
python run.py --model gatv2 --epochs 150 --hidden_dim 64 --dropout 0.3 --lr 1e-3
```
Sin `--eval-subsets`: incluye automáticamente los 4 subsets si sus CSVs existen.
Con `--eval-subsets config_A config_B`: solo esos dos subsets.

---

## Sección: Reporte LaTeX (`tex_data.json`)

**Comando de generación:**
```bash
python run.py --gen-report
```
**Salida:** `src/outputs/reports/tex_data.json`

Generado por `generate_tex_data_report()` en `src/reports/report_builder.py`.
Lee los artefactos del **run más reciente** de cada encoder bajo `src/outputs/training/<model>/`.

---

### `tab:metricas_general`
**Descripción:** Comparación de los 3 encoders sobre el test set completo del grafo.

| tex_data.json key | `encoder_comparison_general` |
|---|---|
| Fuente | `final_metrics.json` de cada encoder |
| Función | `build_encoder_comparison_table()` |
| Columnas | encoder, mse, rmse, mae, mape, r2, n_test |
| n_test | `num_test_edges` del run (≈ `test_ratio` × aristas etiquetadas) |

---

### `tab:metricas_configs`
**Descripción:** GATv2 evaluado sobre test_mask ∩ segmentos directos de config A/B/C.

| tex_data.json key | `config_comparison_gatv2` |
|---|---|
| Fuente | `eval_config_{A/B/C}/metrics.json` del run más reciente de gatv2 |
| Función | `build_config_comparison_table("gatv2")` |
| Columnas | config, km_tol, curve_penalty, n_eval, rmse, mae, mape, r2 |
| Filtro aplicado | `eval_mask = test_mask ∩ labeled_mask ∩ segmentos_1-hop_en_config_{X}` |
| Parámetros de config | A: km_tol=0.2, cp=0.5 · B: km_tol=0.3, cp=0.3 · C: km_tol=0.4, cp=0.2 |

---

### `tab:metricas_rectas`
**Descripción:** Los 3 encoders evaluados sobre test_mask ∩ rutas rectas (straightness_index ≥ threshold).

| tex_data.json key | `encoder_comparison_straight` |
|---|---|
| Fuente | `eval_straight/metrics.json` de cada encoder |
| Función | `build_encoder_comparison_straight()` |
| Columnas | encoder, n_eval, mse, rmse, mae, mape, r2 |
| Filtro aplicado | `eval_mask = test_mask ∩ labeled_mask ∩ segmentos_1-hop_en_straight_routes.csv` |
| Definición de "recto" | `straightness_index ≥ Config.straightness_threshold` (por defecto 0.9, equivale a $r ≤ 1.11$) |

---

### `tab:rmse_por_penalidad`
**Descripción:** RMSE por rango de $r$ para los 3 encoders en config B.

| tex_data.json key | `straightness_breakdown_B` |
|---|---|
| Fuente | `eval_config_B/predictions.csv` de los 3 encoders |
| Función | `build_straightness_breakdown_table("B")` |
| Columnas | r_range, r_min, r_max, n, rmse_graphsage, rmse_gat, rmse_gatv2 |
| Bins de r | ≤1.05 · (1.05, 1.10] · (1.10, 1.20] · >1.20 |
| Requisito | `predictions.csv` debe tener columna `straightness_index` (presente cuando el CSV filtro la tiene) |

---

### `fig:rmse_vs_r`
**Descripción:** RMSE vs $r$ en resolución fina (buckets de 0.02) para los 3 encoders.

| tex_data.json key | `rmse_vs_r_B` |
|---|---|
| Fuente | `eval_config_B/predictions.csv` de los 3 encoders |
| Función | `build_rmse_vs_r_data("B", r_step=0.02)` |
| Columnas | r_center, n, rmse_graphsage, rmse_gat, rmse_gatv2 |

---

## Flujo completo de reproducción

```bash
# 1. Preprocesar datos (una sola vez)
python run.py --process-nyc

# 2. Generar rutas y segmentos (una sola vez)
python run.py -r

# 3. Entrenar los 3 encoders (con evaluación de subsets automática)
python run.py --model graphsage --epochs 150 --hidden_dim 64 --dropout 0.3 --lr 1e-3 --seed 42
python run.py --model gat       --epochs 150 --hidden_dim 64 --dropout 0.3 --lr 1e-3 --seed 42
python run.py --model gatv2     --epochs 150 --hidden_dim 64 --dropout 0.3 --lr 1e-3 --seed 42

# 4. Generar tex_data.json
python run.py --gen-report
```

El archivo `src/outputs/reports/tex_data.json` contiene todos los números
necesarios para rellenar las 4 tablas y la figura del documento LaTeX.

