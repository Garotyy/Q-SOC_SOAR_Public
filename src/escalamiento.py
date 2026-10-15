"""Simulación de escalamiento para incidentes SOC críticos."""

from __future__ import annotations

from typing import Any


def evaluar_escalamiento(incidente: dict[str, Any], severidad: str) -> dict[str, Any]:
    """Decide si un incidente debe escalarse dentro del SOC simulado."""
    razones: list[str] = []
    intentos_fallidos = int(incidente.get("conteo_intentos_fallidos", 0))
    usuarios_atacados = incidente.get("usuarios_atacados", [])
    repeticiones_ip = int(incidente.get("repeticiones_ip_origen", 0))

    if severidad in {"Alta", "Crítica"}:
        razones.append(f"severidad calculada {severidad}")
    if incidente.get("login_exitoso_posterior"):
        razones.append("login exitoso posterior al patrón de intentos fallidos")
    if len(usuarios_atacados) > 1:
        razones.append("múltiples usuarios atacados por la misma IP origen")
    if repeticiones_ip >= 5:
        razones.append(f"IP origen aparece repetidamente ({repeticiones_ip} eventos)")
    if intentos_fallidos > 10:
        razones.append(f"volumen elevado de intentos fallidos ({intentos_fallidos})")

    requiere_escalamiento = bool(razones)

    if not requiere_escalamiento:
        return {
            "requiere_escalamiento": False,
            "nivel_escalamiento": "No requerido",
            "prioridad": "P4",
            "urgencia": "Baja",
            "justificacion": "No cumple criterios de escalamiento automático del MVP.",
        }

    if severidad == "Crítica" or incidente.get("login_exitoso_posterior"):
        nivel = "SOC Nivel 2 / Incident Responder"
        prioridad = "P1"
        urgencia = "Inmediata"
    elif severidad == "Alta":
        nivel = "SOC Nivel 2"
        prioridad = "P2"
        urgencia = "Alta"
    else:
        nivel = "SOC Nivel 1"
        prioridad = "P3"
        urgencia = "Media"

    return {
        "requiere_escalamiento": True,
        "nivel_escalamiento": nivel,
        "prioridad": prioridad,
        "urgencia": urgencia,
        "justificacion": "; ".join(razones) + ".",
    }
