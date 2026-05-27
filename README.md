# time-prediction-SIT-Lima

Predicción de tiempos de viaje en la red de Sistema Integrado de Transporte de Lima (SIT) usando modelos basados en grafos y secuencias temporales.

## Modelos

Este repositorio considera los siguientes encoders espaciales. Todos los
modelos incluyen un decoder MLP a nivel de arista (entrada: `z_src || z_dst || edge_attr` → salida escalar),
listo para tareas de regresión de tiempo de viaje.

| Modelo | Descripción | Tipo |
|--------|-------------|------|
| `gat` | Graph Attention Network (GAT) encoder + edge-level MLP decoder | Espacial |
| `gatv2` | GATv2 encoder + edge-level MLP decoder | Espacial |
| `graphsage` | GraphSAGE encoder + edge-level MLP decoder | Espacial |

## Estructura

```
├── run.py                   # Entry point unificado
├── src/
│   ├── config.py            # Configuración centralizada
│   ├── data/
│   │   ├── mock_lima_graph.py   # Grafo mock multimodal de Lima
│   │   └── load_nyc_mta.py      # Loader GeoPandas -> NetworkX para NYC MTA
│   ├── models/
│   │   ├── gat_model.py
│   │   ├── gatv2_model.py
│   │   ├── graphsage_model.py
│   │   └── gru_model.py        # (opcional) Híbridos con GRU
│   ├── train/
│   │   └── trainer.py           # Pipeline de entrenamiento
│   └── utils/
│       ├── logging.py
│       ├── metrics.py           # MSE, RMSE, MAE, MAPE, R², estadísticas
│       └── seed.py
```

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
