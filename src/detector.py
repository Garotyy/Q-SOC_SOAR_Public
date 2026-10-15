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

EXFIL_TIPOS = {"data_exfiltration"}
NETWORK_TIPOS = {"network_connection"}
DNS_TIPOS = {"dns_query"}
ENDPOINT_TIPOS = {"powershell_activity", "process_creation", "file_access"}

EXFIL_SERVICIOS_SOSPECHOSOS = {"ftp", "smb", "icmp"}
NETWORK_RESULTADOS_SOSPECHOSOS = {"permitido", "detectado", "bloqueado"}


def detectar_incidentes(alertas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detecta y agrupa actividad sospechosa de múltiples tipos."""
    conteo_por_ip = Counter(
        alerta.get("ip_origen") for alerta in alertas if alerta.get("ip_origen")
    )

    grupos_ssh:      dict[tuple, list] = defaultdict(list)
    grupos_exfil:    dict[tuple, list] = defaultdict(list)
    grupos_network:  dict[tuple, list] = defaultdict(list)
    grupos_dns:      dict[tuple, list] = defaultdict(list)
    grupos_endpoint: dict[tuple, list] = defaultdict(list)
    incidentes: list[dict[str, Any]] = []

    for alerta in alertas:
        tipo = str(alerta.get("tipo_evento", "")).lower()

        if _es_evento_ssh(alerta) and _resultado_autenticacion(alerta) in {"fallido", "exitoso"}:
            llave = (
                alerta.get("ip_origen", "desconocida"),
                alerta.get("ip_destino", "desconocida"),
                alerta.get("hostname", "desconocido"),
            )
            grupos_ssh[llave].append(alerta)

        elif tipo in EXFIL_TIPOS:
            llave = (
                alerta.get("ip_origen", "desconocida"),
                alerta.get("hostname", "desconocido"),
                alerta.get("usuario", "desconocido"),
            )
            grupos_exfil[llave].append(alerta)

        elif tipo in NETWORK_TIPOS:
            llave = (
                alerta.get("ip_origen", "desconocida"),
                alerta.get("hostname", "desconocido"),
            )
            grupos_network[llave].append(alerta)

        elif tipo in DNS_TIPOS:
            llave = (
                alerta.get("ip_origen", "desconocida"),
                alerta.get("hostname", "desconocido"),
            )
            grupos_dns[llave].append(alerta)

        elif tipo in ENDPOINT_TIPOS:
            llave = (
                alerta.get("hostname", "desconocido"),
                alerta.get("usuario", "desconocido"),
            )
            grupos_endpoint[llave].append(alerta)

        else:
            incidentes.append(_crear_incidente_no_cubierto(alerta, conteo_por_ip))

    # SSH
    grupos_ordenados_ssh = sorted(
        grupos_ssh.items(),
        key=lambda item: min(
            parsear_timestamp(e.get("timestamp")) for e in item[1]
        ),
    )
    for indice, ((ip_origen, ip_destino, hostname), eventos) in enumerate(
        grupos_ordenados_ssh, start=1
    ):
        eventos_ord = sorted(eventos, key=lambda e: parsear_timestamp(e.get("timestamp")))
        fallidos = [e for e in eventos_ord if _resultado_autenticacion(e) == "fallido"]
        exitosos = [e for e in eventos_ord if _resultado_autenticacion(e) == "exitoso"]
        if not fallidos and not exitosos:
            continue
        usuarios = sorted({e.get("usuario", "desconocido") for e in fallidos or eventos_ord})
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
        incidentes.append({
            "id_alerta": f"INC-SSH-{indice:03d}",
            "timestamp": eventos_ord[0].get("timestamp"),
            "ip_origen": ip_origen,
            "ip_destino": ip_destino,
            "hostname": hostname,
            "usuario": usuarios,
            "servicio": "ssh",
            "tipo_detector": "ssh_authentication_aggregate",
            "tipo_evento": "ssh_authentication",
            "conteo_intentos_fallidos": len(fallidos),
            "conteo_logins_exitosos": len(exitosos),
            "usuarios_atacados": usuarios,
            "login_exitoso_posterior": login_exitoso_posterior,
            "repeticiones_ip_origen": conteo_por_ip.get(ip_origen, 0),
            "eventos_relacionados": [e.get("id_alerta", "sin_id") for e in eventos_ord],
            "evidencia": evidencia,
        })

    # EXFILTRACIÓN
    for indice, ((ip_origen, hostname, usuario), eventos) in enumerate(
        sorted(grupos_exfil.items(),
               key=lambda i: min(parsear_timestamp(e.get("timestamp")) for e in i[1])),
        start=1,
    ):
        eventos_ord = sorted(eventos, key=lambda e: parsear_timestamp(e.get("timestamp")))
        servicios   = sorted({e.get("servicio", "desconocido") for e in eventos_ord})
        destinos    = sorted({e.get("ip_destino", "desconocida") for e in eventos_ord})
        sospechoso  = any(e.get("servicio", "").lower() in EXFIL_SERVICIOS_SOSPECHOSOS
                          for e in eventos_ord)

        evidencia = [
            f"{len(eventos_ord)} evento(s) de exfiltración desde {ip_origen} ({hostname}).",
            f"Usuario involucrado: {usuario}.",
            f"Protocolos observados: {texto_lista(servicios)}.",
            f"Destinos externos: {texto_lista(destinos)}.",
        ]
        if sospechoso:
            evidencia.append(
                "Se detectaron protocolos inusuales (FTP/SMB/ICMP) asociados a exfiltración."
            )

        incidentes.append({
            "id_alerta": f"INC-EXFIL-{indice:03d}",
            "timestamp": eventos_ord[0].get("timestamp"),
            "ip_origen": ip_origen,
            "ip_destino": destinos[0] if destinos else "desconocida",
            "hostname": hostname,
            "usuario": usuario,
            "servicio": servicios[0] if servicios else "desconocido",
            "tipo_detector": "data_exfiltration_aggregate",
            "tipo_evento": "data_exfiltration",
            "conteo_intentos_fallidos": 0,
            "conteo_logins_exitosos": 0,
            "usuarios_atacados": [usuario],
            "login_exitoso_posterior": False,
            "repeticiones_ip_origen": conteo_por_ip.get(ip_origen, 0),
            "eventos_relacionados": [e.get("id_alerta", "sin_id") for e in eventos_ord],
            "evidencia": evidencia,
        })

    # CONEXIONES DE RED
    for indice, ((ip_origen, hostname), eventos) in enumerate(
        sorted(grupos_network.items(),
               key=lambda i: min(parsear_timestamp(e.get("timestamp")) for e in i[1])),
        start=1,
    ):
        eventos_ord = sorted(eventos, key=lambda e: parsear_timestamp(e.get("timestamp")))
        servicios   = sorted({e.get("servicio", "desconocido") for e in eventos_ord})
        destinos    = sorted({e.get("ip_destino", "desconocida") for e in eventos_ord})
        usuarios    = sorted({e.get("usuario", "desconocido") for e in eventos_ord})
        resultados  = sorted({e.get("resultado", "desconocido") for e in eventos_ord})

        evidencia = [
            f"{len(eventos_ord)} conexión(es) de red detectadas desde {ip_origen} ({hostname}).",
            f"Servicios: {texto_lista(servicios)}.",
            f"Destinos: {texto_lista(destinos)}.",
            f"Resultados observados: {texto_lista(resultados)}.",
        ]
        if any(s in {"ftp", "smb"} for s in servicios):
            evidencia.append("Uso de protocolo inusual (FTP/SMB) detectado en conexión de red.")

        incidentes.append({
            "id_alerta": f"INC-NET-{indice:03d}",
            "timestamp": eventos_ord[0].get("timestamp"),
            "ip_origen": ip_origen,
            "ip_destino": destinos[0] if destinos else "desconocida",
            "hostname": hostname,
            "usuario": usuarios,
            "servicio": servicios[0] if servicios else "desconocido",
            "tipo_detector": "network_connection_aggregate",
            "tipo_evento": "network_connection",
            "conteo_intentos_fallidos": 0,
            "conteo_logins_exitosos": 0,
            "usuarios_atacados": [],
            "login_exitoso_posterior": False,
            "repeticiones_ip_origen": conteo_por_ip.get(ip_origen, 0),
            "eventos_relacionados": [e.get("id_alerta", "sin_id") for e in eventos_ord],
            "evidencia": evidencia,
        })

    # DNS
    for indice, ((ip_origen, hostname), eventos) in enumerate(
        sorted(grupos_dns.items(),
               key=lambda i: min(parsear_timestamp(e.get("timestamp")) for e in i[1])),
        start=1,
    ):
        eventos_ord = sorted(eventos, key=lambda e: parsear_timestamp(e.get("timestamp")))
        destinos    = sorted({e.get("ip_destino", "desconocida") for e in eventos_ord})
        usuarios    = sorted({e.get("usuario", "desconocido") for e in eventos_ord})
        bloqueados  = [e for e in eventos_ord if e.get("resultado", "").lower() == "bloqueado"]
        sospechoso  = any(
            kw in e.get("mensaje", "").lower()
            for e in eventos_ord
            for kw in ("malicious", "tunneling", "c2", "onion", "exfil", "sinkhole")
        )

        evidencia = [
            f"{len(eventos_ord)} consulta(s) DNS desde {ip_origen} ({hostname}).",
            f"Destinos de resolución: {texto_lista(destinos)}.",
        ]
        if bloqueados:
            evidencia.append(f"{len(bloqueados)} consulta(s) bloqueadas por sinkhole o firewall.")
        if sospechoso:
            evidencia.append("Consultas asociadas a dominios maliciosos o canales C2 detectados.")

        incidentes.append({
            "id_alerta": f"INC-DNS-{indice:03d}",
            "timestamp": eventos_ord[0].get("timestamp"),
            "ip_origen": ip_origen,
            "ip_destino": destinos[0] if destinos else "desconocida",
            "hostname": hostname,
            "usuario": usuarios,
            "servicio": "dns",
            "tipo_detector": "dns_query_aggregate",
            "tipo_evento": "dns_query",
            "conteo_intentos_fallidos": 0,
            "conteo_logins_exitosos": 0,
            "usuarios_atacados": [],
            "login_exitoso_posterior": False,
            "repeticiones_ip_origen": conteo_por_ip.get(ip_origen, 0),
            "eventos_relacionados": [e.get("id_alerta", "sin_id") for e in eventos_ord],
            "evidencia": evidencia,
        })

    # ENDPOINT
    for indice, ((hostname, usuario), eventos) in enumerate(
        sorted(grupos_endpoint.items(),
               key=lambda i: min(parsear_timestamp(e.get("timestamp")) for e in i[1])),
        start=1,
    ):
        eventos_ord  = sorted(eventos, key=lambda e: parsear_timestamp(e.get("timestamp")))
        tipos_ep     = sorted({e.get("tipo_evento", "desconocido") for e in eventos_ord})
        ip_origenes  = sorted({e.get("ip_origen", "desconocida") for e in eventos_ord})
        tiene_ps     = any(e.get("tipo_evento") == "powershell_activity" for e in eventos_ord)
        tiene_proc   = any(e.get("tipo_evento") == "process_creation"    for e in eventos_ord)
        tiene_file   = any(e.get("tipo_evento") == "file_access"         for e in eventos_ord)

        evidencia = [
            f"{len(eventos_ord)} evento(s) de endpoint en {hostname} por usuario {usuario}.",
            f"Tipos de actividad: {texto_lista(tipos_ep)}.",
        ]
        if tiene_ps:
            evidencia.append("Actividad PowerShell detectada: posible ejecución de payload.")
        if tiene_proc:
            evidencia.append("Creación de procesos sospechosos detectada.")
        if tiene_file:
            evidencia.append("Acceso a archivos sensibles o creación de staging detectada.")
        if tiene_ps and tiene_proc:
            evidencia.append(
                "Combinación PowerShell + proceso sugiere ejecución remota o persistencia."
            )

        incidentes.append({
            "id_alerta": f"INC-ENDPOINT-{indice:03d}",
            "timestamp": eventos_ord[0].get("timestamp"),
            "ip_origen": ip_origenes[0] if ip_origenes else "desconocida",
            "ip_destino": hostname,
            "hostname": hostname,
            "usuario": usuario,
            "servicio": "os",
            "tipo_detector": "endpoint_activity_aggregate",
            "tipo_evento": "endpoint_activity",
            "conteo_intentos_fallidos": 0,
            "conteo_logins_exitosos": 0,
            "usuarios_atacados": [usuario],
            "login_exitoso_posterior": False,
            "repeticiones_ip_origen": conteo_por_ip.get(ip_origenes[0] if ip_origenes else "", 0),
            "eventos_relacionados": [e.get("id_alerta", "sin_id") for e in eventos_ord],
            "evidencia": evidencia,
        })

    return sorted(
        incidentes, key=lambda inc: parsear_timestamp(inc.get("timestamp"))
    )


# helpers SSH 

def _es_evento_ssh(alerta: dict[str, Any]) -> bool:
    servicio    = str(alerta.get("servicio", "")).lower()
    tipo_evento = str(alerta.get("tipo_evento", "")).lower()
    mensaje     = str(alerta.get("mensaje", "")).lower()
    return servicio in SSH_SERVICE_NAMES or "ssh" in tipo_evento or " ssh" in mensaje


def _resultado_autenticacion(alerta: dict[str, Any]) -> str:
    resultado = str(alerta.get("resultado", "")).strip().lower()
    mensaje   = str(alerta.get("mensaje", "")).lower()
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
    primer_fallido = min(parsear_timestamp(e.get("timestamp")) for e in fallidos)
    return any(parsear_timestamp(e.get("timestamp")) > primer_fallido for e in exitosos)


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
        f"{len(fallidos)} intentos fallidos SSH desde {ip_origen} hacia {hostname} ({ip_destino}).",
        f"Usuarios observados: {texto_lista(usuarios)}.",
    ]
    if len(usuarios) > 1:
        evidencia.append("La IP origen probó credenciales contra múltiples usuarios.")
    if login_exitoso_posterior:
        usuarios_exitosos = sorted(
            {e.get("usuario", "desconocido") for e in eventos_exitosos}
        )
        evidencia.append(
            f"Se observó un login exitoso posterior para: {texto_lista(usuarios_exitosos)}."
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
        "tipo_evento": alerta.get("tipo_evento", "desconocido"),
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