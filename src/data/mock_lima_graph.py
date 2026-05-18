"""Generador de grafo mock multimodal de la red de transporte de Lima.

Enfoque "bosque de caminos":
1. Cada ruta genera su propio camino aleatorio (random walk) dentro del
   rango de Lima, con pasos controlados para evitar clustering espacial.
2. Se obtiene un bosque de caminos desconectados (uno por ruta/modo).
3. Se buscan pares de nodos de DISTINTO modo dentro de un distance threshold
   para crear aristas de transferencia, conectando el bosque.

El grafo incluye:
- Aristas intra-ruta (bidireccionales, estaciones consecutivas)
- Aristas de transferencia (entre nodos de distinto modo, por proximidad)
- Features estáticos (10 dims) y temporales (T × 10 dims)
- Target por arista: tiempo de viaje en minutos.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Definición de la red de transporte de Lima (mock)
# ---------------------------------------------------------------------------

TRANSPORT_MODES = {
    "ferroviario": 0,           # Lineas 1 al 9
    "bus_alta_capacidad": 1,    # BRT Norte y BRT Sur
    "bus_media_capacidad": 2,   # Alimentador A y Alimentador B
    "corredor": 3,              # Corredor Azul, Corredor Rojo y Corredor Morado
}

# Cada línea: (nombre, modo, num_estaciones)
LINES: List[Tuple[str, str, int]] = [
    # 9 líneas ferroviarias
    ("Linea 1", "ferroviario", 8),
    ("Linea 2", "ferroviario", 8),
    ("Linea 3", "ferroviario", 6),
    ("Linea 4", "ferroviario", 6),
    ("Linea 5", "ferroviario", 5),
    ("Linea 6", "ferroviario", 5),
    ("Linea 7", "ferroviario", 5),
    ("Linea 8", "ferroviario", 5),
    ("Linea 9", "ferroviario", 5),
    # Buses de alta capacidad
    ("BRT Norte", "bus_alta_capacidad", 8),
    ("BRT Sur", "bus_alta_capacidad", 8),
    # Buses de media capacidad
    ("Alimentador A", "bus_media_capacidad", 6),
    ("Alimentador B", "bus_media_capacidad", 6),
    # Corredores
    ("Corredor Azul", "corredor", 4),
    ("Corredor Rojo", "corredor", 4),
    ("Corredor Morado", "corredor", 4),
]

# Rangos de latitud/longitud para Lima metropolitana (aprox.)
LIMA_LAT_RANGE = (-12.3, -11.8)
LIMA_LON_RANGE = (-77.2, -76.8)

# Capacidades típicas por modo (pasajeros/vehículo)
MODE_CAPACITY = {
    "ferroviario": 1200,
    "bus_alta_capacidad": 160,
    "bus_media_capacidad": 80,
    "corredor": 50,
}

# Frecuencias típicas (vehículos/hora en hora punta)
MODE_FREQUENCY = {
    "ferroviario": 20,
    "bus_alta_capacidad": 12,
    "bus_media_capacidad": 8,
    "corredor": 15,
}

# Velocidades promedio (km/h) para generar tiempos realistas
MODE_SPEED = {
    "ferroviario": 35.0,
    "bus_alta_capacidad": 22.0,
    "bus_media_capacidad": 18.0,
    "corredor": 15.0,
}

# Defaults para generación de grafo
DEFAULT_TRANSFER_THRESHOLD_KM = 3.5
# Paso entre estaciones en grados (~1-3 km)
STEP_MIN_DEG = 0.05
STEP_MAX_DEG = 0.075


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en km entre dos puntos."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _generate_random_walk_route(
    num_stations: int,
    rng: np.random.Generator
) -> np.ndarray:
    """Genera coordenadas para una ruta como random walk dentro de un rango dado.

    Cada paso tiene una dirección aleatoria con inercia (para que la ruta
    no sea un zigzag puro) y un tamaño de paso controlado para evitar
    nodos demasiado cercanos.

    Args:
        num_stations: Número de estaciones en la ruta.
        rng: Generador de números aleatorios.
        lat_range: Tupla (min_lat, max_lat) donde se debe generar la ruta.
        lon_range: Tupla (min_lon, max_lon) donde se debe generar la ruta.

    Returns:
        Array [num_stations, 2] con coordenadas (lat, lon).
    """
    coords = np.zeros((num_stations, 2))

    # Punto de inicio aleatorio dentro del sub-rango especificado
    coords[0, 0] = rng.uniform(*LIMA_LAT_RANGE)
    coords[0, 1] = rng.uniform(*LIMA_LON_RANGE)

    # Dirección inicial aleatoria
    angle = rng.uniform(0, 2 * math.pi)

    for i in range(1, num_stations):
        # Inercia: la dirección cambia gradualmente (no zigzag)
        angle += rng.normal(0, math.pi / 4)
        step = rng.uniform(STEP_MIN_DEG, STEP_MAX_DEG)

        new_lat = coords[i - 1, 0] + step * math.sin(angle)
        new_lon = coords[i - 1, 1] + step * math.cos(angle)

        # Clamp dentro del sub-rango proporcionado
        new_lat = np.clip(new_lat, *LIMA_LAT_RANGE)
        new_lon = np.clip(new_lon, *LIMA_LON_RANGE)

        coords[i, 0] = new_lat
        coords[i, 1] = new_lon

    return coords


def mock_lima_graph(
    seed: int = 42,
    num_time_steps: int = 12,
    transfer_threshold_km: float = DEFAULT_TRANSFER_THRESHOLD_KM,
) -> Tuple[Data, Dict]:
    """Genera el grafo mock multimodal dirigido de la red de Lima.

    Enfoque bosque-de-caminos:
    1. Cada ruta genera su propio random walk dentro de Lima.
    2. Se obtiene un bosque de caminos desconectados.
    3. Aristas de transferencia conectan nodos de distinto modo por proximidad.

    Args:
        seed: Semilla para reproducibilidad.
        num_time_steps: Número de pasos temporales para features dinámicos.
        transfer_threshold_km: Distancia máxima (km) para aristas de transferencia.

    Returns:
        x.static: Array [num_nodes, 10] con features estáticas por nodo.
            - 0: ferroviario (one-hot)
            - 1: bus_alta_capacidad (one-hot)
            - 2: bus_media_capacidad (one-hot)
            - 3: corredor (one-hot)
            - 4: lat_norm (0-1)
            - 5: lon_norm (0-1)
            - 6: demand (0-1)
            - 7: frequency (0-1)
            - 8: capacity (0-1)
            - 9: transfer_flag (0 o 1)
        edge_index: Tensor [2, num_edges] con índices de aristas (src, dst).
        edge_attr: Array [num_edges, 2] con atributos por arista:
            - 0: distancia en km
            - 1: tipo de arista (0 = intra-ruta, 1 = transferencia)
        node_lines_list: Lista [num_nodes] con el índice de línea de cada nodo.
        metadata: Dict con información descriptiva del grafo.
    """
    rng = np.random.default_rng(seed)

    # =================================================================
    # Paso 1: Generar rutas como random walks independientes
    # =================================================================
    node_features_list = []
    node_types = []
    node_lines_list = []
    node_names = []
    node_coords = []
    node_mode_idx = []  # modo de cada nodo

    # Mapeo: (line_idx, station_idx) -> node_id global
    station_to_node: Dict[Tuple[int, int], int] = {}

    max_freq = max(MODE_FREQUENCY.values())
    max_cap = max(MODE_CAPACITY.values())

    node_id = 0

    # Generar rutas: todas las líneas (una linea por cada servicio)
    for line_idx, (line_name, mode, num_stations) in enumerate(LINES):
        coords = _generate_random_walk_route(num_stations, rng)
        mode_idx = TRANSPORT_MODES[mode]

        for st_idx in range(num_stations):
            station_to_node[(line_idx, st_idx)] = node_id

            # One-hot del tipo de transporte (4 dims)
            type_onehot = [0.0] * 4
            type_onehot[mode_idx] = 1.0

            # Coordenadas normalizadas (2 dims) respecto al rango global de Lima
            lat_norm = (coords[st_idx, 0] - LIMA_LAT_RANGE[0]) / (
                LIMA_LAT_RANGE[1] - LIMA_LAT_RANGE[0]
            )
            lon_norm = (coords[st_idx, 1] - LIMA_LON_RANGE[0]) / (
                LIMA_LON_RANGE[1] - LIMA_LON_RANGE[0]
            )

            # Demanda, frecuencia, capacidad normalizadas
            demand = rng.uniform(0.1, 1.0)
            freq = MODE_FREQUENCY[mode] / max_freq
            cap = MODE_CAPACITY[mode] / max_cap

            # Transfer flag — se actualiza en Paso 3
            transfer_flag = 0.0

            features = type_onehot + [
                lat_norm, lon_norm, demand, freq, cap, transfer_flag
            ]
            node_features_list.append(features)
            node_types.append(mode_idx)
            node_lines_list.append(line_idx)
            node_names.append(f"{line_name} - Est.{st_idx}")
            node_coords.append((coords[st_idx, 0], coords[st_idx, 1]))
            node_mode_idx.append(mode_idx)

            node_id += 1

    num_nodes = node_id
    print(f"Status Mock: {num_nodes} nodos creados en {len(LINES)} rutas")

    # =================================================================
    # Paso 2: Aristas intra-ruta (bidireccionales)
    # =================================================================
    edge_src = []
    edge_dst = []
    edge_distances = []
    edge_types = []      # 0 = intra-ruta, 1 = transferencia
    edge_route_modes = []

    for line_idx, (line_name, mode, num_stations) in enumerate(LINES):
        mode_idx = TRANSPORT_MODES[mode]
        for st_idx in range(num_stations - 1):
            src = station_to_node[(line_idx, st_idx)]
            dst = station_to_node[(line_idx, st_idx + 1)]

            lat1, lon1 = node_coords[src]
            lat2, lon2 = node_coords[dst]
            dist = _haversine_km(lat1, lon1, lat2, lon2)

            # Ida
            edge_src.append(src)
            edge_dst.append(dst)
            edge_distances.append(dist)
            edge_types.append(0)
            edge_route_modes.append(mode_idx)

            # Vuelta
            edge_src.append(dst)
            edge_dst.append(src)
            edge_distances.append(dist)
            edge_types.append(0)
            edge_route_modes.append(mode_idx)

    print("Status Mock: Aristas intra-ruta creadas (bosque de caminos)")

    # =================================================================
    # Paso 3: Aristas de transferencia por distance threshold
    # Solo entre nodos de DISTINTO modo de transporte
    # =================================================================
    existing_pairs = set(zip(edge_src, edge_dst))
    transfer_nodes: set = set()
    num_transfer_edges = 0

    for i in range(num_nodes):
        mode_i = node_mode_idx[i]
        lat_i, lon_i = node_coords[i]
        for j in range(i + 1, num_nodes):
            mode_j = node_mode_idx[j]

            # Solo modos diferentes
            if mode_i == mode_j:
                continue

            lat_j, lon_j = node_coords[j]
            dist = _haversine_km(lat_i, lon_i, lat_j, lon_j)
            if dist > transfer_threshold_km:
                continue

            # Crear arista bidireccional si no existe
            if (i, j) not in existing_pairs:
                edge_src.append(i)
                edge_dst.append(j)
                edge_distances.append(dist)
                edge_types.append(1)
                edge_route_modes.append(mode_i)
                existing_pairs.add((i, j))
                num_transfer_edges += 1

            if (j, i) not in existing_pairs:
                edge_src.append(j)
                edge_dst.append(i)
                edge_distances.append(dist)
                edge_types.append(1)
                edge_route_modes.append(mode_j)
                existing_pairs.add((j, i))
                num_transfer_edges += 1

            transfer_nodes.add(i)
            transfer_nodes.add(j)

    # Actualizar transfer_flag en features
    for nid in transfer_nodes:
        node_features_list[nid][-1] = 1.0

    print(f"Status Mock: {num_transfer_edges} aristas de transferencia, "
          f"{len(transfer_nodes)} nodos de transferencia")

    num_edges = len(edge_src)

    # =================================================================
    # Paso 4: Generar tiempos de viaje (targets)
    # =================================================================
    mode_keys = list(TRANSPORT_MODES.keys())
    travel_times = []
    for i in range(num_edges):
        dist = edge_distances[i]
        e_type = edge_types[i]
        route_mode = edge_route_modes[i]

        mode_key = mode_keys[route_mode]
        speed = MODE_SPEED[mode_key]

        if e_type == 1:
            # Transferencias: tiempo de caminata (3-8 min)
            time_min = rng.uniform(3.0, 8.0)
        else:
            # Tiempo = distancia / velocidad * 60 + ruido + parada
            time_min = (dist / speed) * 60.0
            time_min += rng.uniform(0.5, 2.0)
            time_min += rng.normal(0, 0.3)

        travel_times.append(max(0.5, time_min))

    print("Status Mock: Tiempos de viaje generados")

    x_static = np.array(node_features_list, dtype=np.float32)
    edge_attr = np.array(list(zip(edge_distances, edge_types)), dtype=np.float32)

    # ----- Metadata descriptiva -----
    metadata = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_lines": len(LINES),
        "num_transfer_nodes": len(transfer_nodes),
        "num_transfer_edges": num_transfer_edges,
        "transfer_threshold_km": transfer_threshold_km,
        "node_feature_dim": x_static.shape[1],
        "num_time_steps": num_time_steps,
        "modes": dict(TRANSPORT_MODES),
        "lines": [(name, mode, n) for name, mode, n in LINES],
        "node_names": node_names
    }

    print("Status Mock: Construcción del grafo mock completado")
    return x_static, edge_src, edge_dst, edge_attr, node_lines_list, metadata


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    x_static, edge_src, edge_dst, edge_attr, node_lines_list, metadata = mock_lima_graph()

    print("\nResults Mock:")
    print(f"Nodos: {metadata['num_nodes']}")
    print(f"Aristas: {metadata['num_edges']}")
    print(f"Líneas: {metadata['num_lines']}")
    print(f"Nodos de transferencia: {metadata['num_transfer_nodes']}")
    print(f"Aristas de transferencia: {metadata['num_transfer_edges']}")
    print(f"Threshold: {metadata['transfer_threshold_km']} km")

    # ----- Visualización -----
    lat_all = x_static[:, 4] * (LIMA_LAT_RANGE[1] - LIMA_LAT_RANGE[0]) + LIMA_LAT_RANGE[0]
    lon_all = x_static[:, 5] * (LIMA_LON_RANGE[1] - LIMA_LON_RANGE[0]) + LIMA_LON_RANGE[0]

    num_edges_total = len(edge_src)
    num_nodes_total = x_static.shape[0]

    mode_colors = {
        0: "#E63946",   # línea 1 — rojo
        1: "#457B9D",   # línea 2 — azul
        2: "#2A9D8F",   # línea 3 — verde
        3: "#E9C46A",   # línea 4 — amarillo
        4: "#F4A261",   # línea 5 — naranja
        5: "#8338EC",   # línea 6 — púrpura
        6: "#3A86FF",   # línea 7 — azul brillante
        7: "#FF006E",   # línea 8 — rosa intenso
        8: "#FB5607",   # línea 9 — naranja neón
        9: "#00E676",   # BRT Norte — verde lima
        10: "#FF4081",  # BRT Sur — rosa neón
        11: "#4DB6AC",  # alimentador A — turquesa
        12: "#FF8A65",  # alimentador B — salmón
        13: "#9575CD",  # corredor azul — lavanda
        14: "#FFB74D",  # corredor rojo — mandarina
        15: "#26C6DA",  # corredor verde — cian
    }
    mode_labels = {v: k for k, v in TRANSPORT_MODES.items()}

    edge_line_colors = {
        0: "#555555",   # intra-línea — gris oscuro
        1: "#FF6B6B",   # transferencia — rojo claro
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    # Dibujar todas las aristas
    for i in range(num_edges_total):
        s = edge_src[i]
        d = edge_dst[i]
        e_type = int(edge_attr[i, 1])
        color = edge_line_colors[e_type]
        lw = 1.0 if e_type == 0 else 1.5
        ls = "-" if e_type == 0 else "--"
        ax.annotate(
            "",
            xy=(lon_all[d], lat_all[d]),
            xytext=(lon_all[s], lat_all[s]),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw,
                linestyle=ls, shrinkA=4, shrinkB=4,
            ),
        )

    # Dibujar todos los nodos
    for n in range(num_nodes_total):
        mode_idx = node_lines_list[n]
        is_transfer = x_static[n, 9] > 0.5
        node_size = 80 if is_transfer else 50
        ax.scatter(
            lon_all[n], lat_all[n],
            c=mode_colors[mode_idx], s=node_size, zorder=5,
            edgecolors="black" if is_transfer else "white",
            linewidths=1.5 if is_transfer else 0.8,
        )

    # Leyenda
    # mode_handles = [
    #     mlines.Line2D([], [], color=mode_colors[idx], marker="o",
    #                    linestyle="None", markersize=8, label=mode_labels[idx])
    #     for idx in sorted(mode_colors.keys())
    # ]
    # edge_handles = [
    #     mlines.Line2D([], [], color=edge_line_colors[0], linestyle="-",
    #                    lw=1.5, label="Intra-ruta"),
    #     mlines.Line2D([], [], color=edge_line_colors[1], linestyle="--",
    #                    lw=1.5, label="Transferencia"),
    # ]
    # transfer_handle = mlines.Line2D(
    #     [], [], color="gray", marker="o", linestyle="None",
    #     markersize=10, markeredgecolor="black", markeredgewidth=2.0,
    #     label="Nodo de transferencia",
    # )
    # ax.legend(handles=mode_handles + edge_handles + [transfer_handle],
    #           loc="upper left", fontsize=9)

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Red de transporte de Lima (mock) — Bosque de caminos")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
