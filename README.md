# time-prediction-GCN-Lima

Predicción de tiempos de viaje en la red de Sistema Integrado de Transporte de Lima (SIT) usando modelos basados en grafos y secuencias temporales.

## Modelos

| Modelo | Descripción | Tipo |
|--------|-------------|------|
| `gcn` | GCN puro (`torch_geometric.nn.models.GCN`) | Espacial |
| `graphsage` | GraphSAGE con `SAGEConv` + BatchNorm | Espacial |
| `gcn_gru` | GCN encoder + GRU temporal | Espacio-temporal |
| `graphsage_gru` | GraphSAGE encoder + GRU temporal | Espacio-temporal |

## Estructura

```
├── run.py                   # Entry point unificado
├── src/
│   ├── config.py            # Configuración centralizada
│   ├── data/
│   │   └── mock_lima_graph.py   # Grafo mock multimodal de Lima
│   ├── models/
│   │   ├── gcn_model.py
│   │   ├── graphsage_model.py
│   │   └── gru_model.py        # Híbridos GCN+GRU y GraphSAGE+GRU
│   ├── train/
│   │   └── trainer.py           # Pipeline de entrenamiento
│   └── utils/
│       ├── logging.py
│       ├── metrics.py           # MSE, RMSE, MAE, MAPE, R², estadísticas
│       └── seed.py
└── tests/
    └── compare_models.py
```

## Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar un modelo
python run.py --model gcn --epochs 50
python run.py --model graphsage --epochs 50
python run.py --model gcn_gru --epochs 50
python run.py --model graphsage_gru --epochs 50

# Personalizar hiperparámetros
python run.py --model gcn --hidden_dim 128 --num_layers 3 --lr 0.0005
```

## Métricas

- MSE, RMSE, MAE, MAPE (%), R²
