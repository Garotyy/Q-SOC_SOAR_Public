"""Capa de detección para alertas SOC simuladas.

El primer caso de uso implementado es la detección de SSH Brute Force. Las
funciones se mantienen lo bastante genéricas para agregar detectores de
phishing, malware, PowerShell o anomalías de red sin cambiar el resto del
flujo de procesamiento.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.utils import parsear_timestamp, texto_lista


SSH_SERVICE_NAMES = {"ssh", "sshd"}
FAILED_RESULTS = {"fallido", "failed", "failure", "denied", "invalid"}
SUCCESS_RESULTS = {"exitoso", "success", "successful", "accepted"}


def detectar_incidentes(alertas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detecta y agrupa actividad sospechosa de autenticación SSH."""
    conteo_por_ip = Counter(
        alerta.get("ip_origen") for alerta in alertas if alerta.get("ip_origen")
    )
    grupos_ssh: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    incidentes: list[dict[str, Any]] = []

    for alerta in alertas:
        if _es_evento_ssh(alerta) and _resultado_autenticacion(alerta) in {
            "fallido",
            "exitoso",
        }:
            llave = (
                alerta.get("ip_origen", "desconocida"),
                alerta.get("ip_destino", "desconocida"),
                alerta.get("hostname", "desconocido"),
            )
            grupos_ssh[llave].append(alerta)
        else:
            incidentes.append(_crear_incidente_no_cubierto(alerta, conteo_por_ip))

    grupos_ordenados = sorted(
        grupos_ssh.items(),
        key=lambda item: min(
            parsear_timestamp(evento.get("timestamp")) for evento in item[1]
        ),
    )

    for indice, ((ip_origen, ip_destino, hostname), eventos) in enumerate(
        grupos_ordenados, start=1
    ):
        eventos_ordenados = sorted(
            eventos, key=lambda evento: parsear_timestamp(evento.get("timestamp"))
        )
        fallidos = [
            evento
            for evento in eventos_ordenados
            if _resultado_autenticacion(evento) == "fallido"
        ]
        exitosos = [
            evento
            for evento in eventos_ordenados
            if _resultado_autenticacion(evento) == "exitoso"
        ]

        if not fallidos and not exitosos:
            continue

        usuarios = sorted(
            {
                evento.get("usuario", "desconocido")
                for evento in fallidos or eventos_ordenados
            }
        )
        login_exitoso_posterior = _hay_login_exitoso_posterior(fallidos, exitosos)
        evidencia = _construir_evidencia(
            ip_origen=ip_origen,
            ip_destino=ip_destino,
            hostname=hostname,
            fallidos=fallidos,
            usuarios=usuarios,
            login_exitoso_posterior=login_exitoso_posterior,
            eventos_exitosos=exitosos,
        )

        incidentes.append(
            {
                "id_alerta": f"INC-SSH-{indice:03d}",
                "timestamp": eventos_ordenados[0].get("timestamp"),
                "ip_origen": ip_origen,
                "ip_destino": ip_destino,
                "hostname": hostname,
                "usuario": usuarios,
                "servicio": "ssh",
                "tipo_detector": "ssh_authentication_aggregate",
                "conteo_intentos_fallidos": len(fallidos),
                "conteo_logins_exitosos": len(exitosos),
                "usuarios_atacados": usuarios,
                "login_exitoso_posterior": login_exitoso_posterior,
                "repeticiones_ip_origen": conteo_por_ip.get(ip_origen, 0),
                "eventos_relacionados": [
                    evento.get("id_alerta", "sin_id") for evento in eventos_ordenados
                ],
                "evidencia": evidencia,
            }
        )

    return sorted(
        incidentes, key=lambda incidente: parsear_timestamp(incidente.get("timestamp"))
    )


def _es_evento_ssh(alerta: dict[str, Any]) -> bool:
    servicio = str(alerta.get("servicio", "")).lower()
    tipo_evento = str(alerta.get("tipo_evento", "")).lower()
    mensaje = str(alerta.get("mensaje", "")).lower()
    return servicio in SSH_SERVICE_NAMES or "ssh" in tipo_evento or " ssh" in mensaje


def _resultado_autenticacion(alerta: dict[str, Any]) -> str:
    resultado = str(alerta.get("resultado", "")).strip().lower()
    mensaje = str(alerta.get("mensaje", "")).lower()

    if resultado in FAILED_RESULTS or "failed password" in mensaje:
        return "fallido"
    if resultado in SUCCESS_RESULTS or "accepted password" in mensaje:
        return "exitoso"
    return "desconocido"


def _hay_login_exitoso_posterior(
    fallidos: list[dict[str, Any]], exitosos: list[dict[str, Any]]
) -> bool:
    if not fallidos or not exitosos:
        return False

    primer_fallido = min(parsear_timestamp(evento.get("timestamp")) for evento in fallidos)
    return any(
        parsear_timestamp(evento.get("timestamp")) > primer_fallido for evento in exitosos
    )


def _construir_evidencia(
    ip_origen: str,
    ip_destino: str,
    hostname: str,
    fallidos: list[dict[str, Any]],
    usuarios: list[str],
    login_exitoso_posterior: bool,
    eventos_exitosos: list[dict[str, Any]],
) -> list[str]:
    evidencia = [
        (
            f"{len(fallidos)} intentos fallidos SSH desde {ip_origen} "
            f"hacia {hostname} ({ip_destino})."
        ),
        f"Usuarios observados: {texto_lista(usuarios)}.",
    ]

    if len(usuarios) > 1:
        evidencia.append("La IP origen probó credenciales contra múltiples usuarios.")
    if login_exitoso_posterior:
        usuarios_exitosos = sorted(
            {evento.get("usuario", "desconocido") for evento in eventos_exitosos}
        )
        evidencia.append(
            "Se observó un login exitoso posterior para: "
            f"{texto_lista(usuarios_exitosos)}."
        )
    return evidencia


def _crear_incidente_no_cubierto(
    alerta: dict[str, Any], conteo_por_ip: Counter
) -> dict[str, Any]:
    ip_origen = alerta.get("ip_origen", "desconocida")
    return {
        "id_alerta": alerta.get("id_alerta", "sin_id"),
        "timestamp": alerta.get("timestamp"),
        "ip_origen": ip_origen,
        "ip_destino": alerta.get("ip_destino", "desconocida"),
        "hostname": alerta.get("hostname", "desconocido"),
        "usuario": alerta.get("usuario", "N/A"),
        "servicio": alerta.get("servicio", "desconocido"),
        "tipo_detector": "sin_detector_activo",
        "conteo_intentos_fallidos": 0,
        "conteo_logins_exitosos": 0,
        "usuarios_atacados": [],
        "login_exitoso_posterior": False,
        "repeticiones_ip_origen": conteo_por_ip.get(ip_origen, 0),
        "eventos_relacionados": [alerta.get("id_alerta", "sin_id")],
        "evidencia": [
            "Evento recibido, pero no coincide con detectores activos del MVP.",
            f"Tipo original: {alerta.get('tipo_evento', 'desconocido')}.",
        ],
    }
