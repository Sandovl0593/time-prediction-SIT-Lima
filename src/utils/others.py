"""Logging and seed setup."""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_logger(name: str = "project", log_file: Optional[str] = None) -> logging.Logger:
    """Obtiene (o crea) un logger con StreamHandler y, opcionalmente, FileHandler.

    Args:
        name: Nombre del logger.
        log_file: Ruta al archivo de log. Si se pasa, se añade un FileHandler
            que escribe en ese archivo. El directorio se crea si no existe.
            Si el logger ya tiene un FileHandler para esa ruta, no se duplica.
    """
    logger = logging.getLogger(name)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    if log_file is not None:
        log_path = Path(log_file)
        existing_paths = {
            h.baseFilename
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if str(log_path.resolve()) not in existing_paths:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger
