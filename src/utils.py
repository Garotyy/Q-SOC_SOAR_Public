"""Funciones auxiliares para el MVP SOAR-AI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def cargar_json(ruta: str | Path) -> Any:
    """Carga un archivo JSON en UTF-8."""
    ruta = Path(ruta)
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_json(datos: Any, ruta: str | Path) -> None:
    """Guarda datos JSON con formato estable para reportes y demostraciones."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=2, ensure_ascii=False)


def parsear_timestamp(valor: str | None) -> datetime:
    """Interpreta timestamps ISO-8601 y los normaliza a UTC con zona horaria."""
    if not valor:
        return datetime.min.replace(tzinfo=timezone.utc)

    texto = str(valor).replace("Z", "+00:00")
    try:
        fecha = datetime.fromisoformat(texto)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)

    if fecha.tzinfo is None:
        return fecha.replace(tzinfo=timezone.utc)
    return fecha.astimezone(timezone.utc)


def texto_lista(valores: list[str]) -> str:
    """Devuelve una representación compacta y legible de una lista."""
    return ", ".join(valores) if valores else "N/A"
