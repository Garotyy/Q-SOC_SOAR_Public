"""Clasificador de incidentes basado en reglas para el MVP SOAR-AI."""

from __future__ import annotations

from typing import Any


def clasificar_incidente(incidente: dict[str, Any]) -> str:
    """Clasifica un incidente detectado usando reglas SOC transparentes."""
    if incidente.get("tipo_detector") == "sin_detector_activo":
        return "Unknown Event"

    servicio = str(incidente.get("servicio", "")).lower()
    intentos_fallidos = int(incidente.get("conteo_intentos_fallidos", 0))
    usuarios_atacados = incidente.get("usuarios_atacados", [])
    login_exitoso_posterior = bool(incidente.get("login_exitoso_posterior"))

    if servicio == "ssh" and login_exitoso_posterior and intentos_fallidos < 5:
        return "Suspicious Login Activity"
    if servicio == "ssh" and (intentos_fallidos >= 5 or len(usuarios_atacados) > 1):
        return "SSH Brute Force"
    if servicio == "ssh" and intentos_fallidos > 0:
        return "Unauthorized Access Attempt"
    if servicio == "ssh":
        return "Suspicious Login Activity"
    return "Unknown Event"
