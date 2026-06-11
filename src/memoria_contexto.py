"""Funciones auxiliares de contexto histórico para SOAR-AI.

La capa de memoria enriquece los reportes con observaciones previas de la misma
ejecución del flujo de procesamiento. Es intencionalmente analítica: no dispara acciones
reales ni modifica decisiones de detección, MITRE, severidad, playbook o
escalamiento.
"""

from __future__ import annotations

from typing import Any


def construir_contexto_historico(
    reporte: dict[str, Any], reportes_previos: list[dict[str, Any]]
) -> dict[str, Any]:
    """Construye un diccionario compacto de contexto histórico para un reporte.

    La función compara el reporte actual con reportes ya generados en el flujo
    de procesamiento. Esto entrega memoria al agente SOAR académico sobre IPs,
    usuarios, hosts y patrones de SSH Brute Force repetidos sin cambiar el motor
    de reglas.
    """
    ip_origen = reporte.get("ip_origen")
    hostname = reporte.get("hostname")
    usuarios_actuales = set(_normalizar_usuarios(reporte.get("usuario")))

    reportes_relacionados = [
        previo
        for previo in reportes_previos
        if _esta_relacionado(
            reporte_actual=reporte,
            reporte_previo=previo,
            usuarios_actuales=usuarios_actuales,
        )
    ]

    usuarios_previos = set()
    for previo in reportes_previos:
        usuarios_previos.update(_normalizar_usuarios(previo.get("usuario")))

    usuarios_recurrentes = sorted(usuarios_actuales.intersection(usuarios_previos))
    ip_origen_recurrente = any(
        ip_origen and previo.get("ip_origen") == ip_origen for previo in reportes_previos
    )
    hostname_recurrente = any(
        hostname and previo.get("hostname") == hostname for previo in reportes_previos
    )
    incidentes_criticos_previos = sum(
        1 for previo in reportes_relacionados if _es_critico(previo)
    )
    patron_repetido = _existe_patron_repetido_fuerza_bruta(
        reporte, reportes_relacionados
    )

    return {
        "ip_origen_recurrente": ip_origen_recurrente,
        "usuarios_recurrentes": usuarios_recurrentes,
        "hostname_recurrente": hostname_recurrente,
        "incidentes_relacionados_previos": len(reportes_relacionados),
        "incidentes_criticos_previos": incidentes_criticos_previos,
        "patron_repetido": patron_repetido,
        "resumen_contexto": _construir_resumen_contexto(
            ip_origen_recurrente=ip_origen_recurrente,
            usuarios_recurrentes=usuarios_recurrentes,
            hostname_recurrente=hostname_recurrente,
            incidentes_relacionados_previos=len(reportes_relacionados),
            incidentes_criticos_previos=incidentes_criticos_previos,
            patron_repetido=patron_repetido,
        ),
    }


def _esta_relacionado(
    reporte_actual: dict[str, Any],
    reporte_previo: dict[str, Any],
    usuarios_actuales: set[str],
) -> bool:
    """Decide si dos reportes comparten indicadores SOC observables."""
    misma_ip = (
        reporte_actual.get("ip_origen")
        and reporte_actual.get("ip_origen") == reporte_previo.get("ip_origen")
    )
    mismo_host = (
        reporte_actual.get("hostname")
        and reporte_actual.get("hostname") == reporte_previo.get("hostname")
    )
    usuarios_previos = set(_normalizar_usuarios(reporte_previo.get("usuario")))
    usuario_repetido = bool(usuarios_actuales.intersection(usuarios_previos))

    return bool(misma_ip or mismo_host or usuario_repetido)


def _normalizar_usuarios(valor: Any) -> list[str]:
    """Normaliza campos de usuario del reporte para comparaciones limpias."""
    if isinstance(valor, list):
        usuarios = valor
    elif valor in (None, "", "N/A"):
        usuarios = []
    else:
        usuarios = [str(valor)]

    return sorted(
        {
            str(usuario).strip()
            for usuario in usuarios
            if str(usuario).strip() and str(usuario).strip().upper() != "N/A"
        }
    )


def _es_critico(reporte: dict[str, Any]) -> bool:
    severidad = str(reporte.get("severidad", "")).strip().lower()
    return severidad in {"crítica", "critica"}


def _existe_patron_repetido_fuerza_bruta(
    reporte: dict[str, Any], reportes_relacionados: list[dict[str, Any]]
) -> bool:
    """Identifica comportamiento SSH Brute Force repetido en reportes previos."""
    if reporte.get("tipo_incidente") != "SSH Brute Force":
        return False

    return any(
        previo.get("tipo_incidente") == "SSH Brute Force"
        for previo in reportes_relacionados
    )


def _construir_resumen_contexto(
    ip_origen_recurrente: bool,
    usuarios_recurrentes: list[str],
    hostname_recurrente: bool,
    incidentes_relacionados_previos: int,
    incidentes_criticos_previos: int,
    patron_repetido: bool,
) -> str:
    """Crea un resumen legible para analistas y para el prompt de Qwen3."""
    if incidentes_relacionados_previos == 0:
        return "No se identifican incidentes previos relacionados en esta ejecución."

    partes = [
        f"Se identifican {incidentes_relacionados_previos} incidentes previos relacionados."
    ]
    if incidentes_criticos_previos:
        partes.append(f"{incidentes_criticos_previos} fueron críticos.")
    if ip_origen_recurrente:
        partes.append("La IP origen ya había aparecido antes.")
    if usuarios_recurrentes:
        partes.append(
            "Usuarios recurrentes: " + ", ".join(usuarios_recurrentes) + "."
        )
    if hostname_recurrente:
        partes.append("El hostname ya había sido observado previamente.")
    if patron_repetido:
        partes.append("Existe patrón repetido de fuerza bruta SSH.")

    return " ".join(partes)
