"""Reglas de cálculo de severidad para SOAR-AI."""

from __future__ import annotations

from typing import Any


def calcular_severidad(incidente: dict[str, Any]) -> str:
    """Calcula la severidad usando reglas simples y auditables del MVP."""
    intentos_fallidos = int(incidente.get("conteo_intentos_fallidos", 0))
    usuarios_atacados = incidente.get("usuarios_atacados", [])
    login_exitoso_posterior = bool(incidente.get("login_exitoso_posterior"))

    if (
        intentos_fallidos > 20
        or len(usuarios_atacados) > 1
        or login_exitoso_posterior
    ):
        return "Crítica"
    if intentos_fallidos > 10:
        return "Alta"
    if 5 <= intentos_fallidos <= 10:
        return "Media"
    return "Baja"
