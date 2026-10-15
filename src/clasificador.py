"""Clasificador de incidentes basado en reglas para el MVP SOAR-AI."""

from __future__ import annotations

from typing import Any


def clasificar_incidente(incidente: dict[str, Any]) -> str:
    """Clasifica un incidente detectado usando reglas SOC transparentes."""

    tipo_detector = incidente.get("tipo_detector", "")

    # Casos sin detector activo
    if tipo_detector == "sin_detector_activo":
        return "Unknown Event"

    # Exfiltración de datos
    if tipo_detector == "data_exfiltration_aggregate":
        evidencia = " ".join(incidente.get("evidencia", []))
        if "ftp" in evidencia.lower() or "smb" in evidencia.lower() or "icmp" in evidencia.lower():
            return "Data Exfiltration via Covert Channel"
        return "Data Exfiltration"

    # Conexiones de red sospechosas
    if tipo_detector == "network_connection_aggregate":
        evidencia = " ".join(incidente.get("evidencia", []))
        if "inusual" in evidencia.lower() or "ftp" in evidencia.lower() or "smb" in evidencia.lower():
            return "Suspicious Network Connection"
        return "Network Activity Observed"

    # Consultas DNS sospechosas
    if tipo_detector == "dns_query_aggregate":
        evidencia = " ".join(incidente.get("evidencia", []))
        if any(kw in evidencia.lower() for kw in ("malicioso", "c2", "sinkhole", "bloqueada", "tunneling")):
            return "Suspicious DNS Activity"
        return "DNS Query Observed"

    # Actividad de endpoint 
    if tipo_detector == "endpoint_activity_aggregate":
        evidencia = " ".join(incidente.get("evidencia", []))
        tiene_ps   = "powershell" in evidencia.lower()
        tiene_proc = "proceso" in evidencia.lower()
        tiene_file = "archivo" in evidencia.lower()

        if tiene_ps and tiene_proc:
            return "Endpoint Compromise - Execution"
        if tiene_ps:
            return "Suspicious PowerShell Activity"
        if tiene_file and tiene_proc:
            return "Endpoint Compromise - Staging"
        return "Suspicious Endpoint Activity"

    # SSH 
    servicio             = str(incidente.get("servicio", "")).lower()
    intentos_fallidos    = int(incidente.get("conteo_intentos_fallidos", 0))
    usuarios_atacados    = incidente.get("usuarios_atacados", [])
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