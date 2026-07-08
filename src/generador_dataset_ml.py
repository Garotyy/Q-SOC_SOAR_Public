"""Generador de eventos simulados para la capa ML del MVP SOAR-AI.

El dataset se mantiene separado de las alertas del pipeline principal para no
alterar la deteccion, la clasificacion por reglas ni los reportes existentes.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.utils import DATA_DIR


RANDOM_STATE = 42
TOTAL_EVENTOS = 2000
RUTA_DATASET_ML = DATA_DIR / "eventos_ml_simulados.csv"

CLASES_EVENTO = ("normal", "fallido", "sospechoso")
USUARIOS = (
    "admin",
    "root",
    "backup",
    "oracle",
    "devops",
    "soporte",
    "analyst",
    "ubuntu",
    "deploy",
    "postgres",
    "test",
)
USUARIOS_PRIVILEGIADOS = {"admin", "root", "backup", "oracle", "devops", "deploy", "postgres"}
HOSTS = ("srv-web-01", "srv-db-01", "srv-auth-01", "srv-app-01", "ws-analyst-07")
HOSTS_CRITICOS = {"srv-web-01", "srv-db-01", "srv-auth-01", "srv-app-01"}
SERVICIOS_SENSIBLES = {"ssh", "smb", "ftp", "powershell", "os"}

TIPOS_POR_CLASE = {
    "normal": (
        "network_connection",
        "dns_query",
        "file_access",
        "process_creation",
        "ssh_authentication",
    ),
    "fallido": (
        "ssh_authentication",
        "network_connection",
        "dns_query",
        "process_creation",
    ),
    "sospechoso": (
        "ssh_authentication",
        "data_exfiltration",
        "powershell_activity",
        "process_creation",
        "file_access",
        "dns_query",
        "network_connection",
    ),
}

SERVICIOS_POR_TIPO = {
    "ssh_authentication": ("ssh",),
    "data_exfiltration": ("ftp", "smb", "https", "icmp"),
    "network_connection": ("http", "https", "ftp", "smb"),
    "dns_query": ("dns",),
    "powershell_activity": ("powershell",),
    "process_creation": ("os",),
    "file_access": ("os",),
}

IP_EXTERNAS_DOCUMENTACION = (
    "203.0.113.45",
    "203.0.113.88",
    "203.0.113.200",
    "198.51.100.77",
    "198.51.100.88",
    "198.51.100.130",
    "192.0.2.14",
    "192.0.2.44",
    "192.0.2.55",
)
IP_PRIVADAS = (
    "10.10.20.15",
    "10.10.20.20",
    "10.10.30.21",
    "10.10.40.12",
    "10.10.50.23",
    "172.16.10.20",
    "192.168.10.44",
)
IP_DESTINO_LAB = (
    "10.10.20.15",
    "10.10.20.20",
    "10.10.30.21",
    "10.10.40.12",
    "10.10.50.23",
    "8.8.8.8",
)

COLUMNAS_DATASET = (
    "id_evento",
    "timestamp",
    "tipo_evento",
    "servicio",
    "usuario",
    "hostname",
    "ip_origen",
    "ip_destino",
    "hora_evento",
    "es_horario_laboral",
    "ip_origen_es_privada",
    "usuario_privilegiado",
    "host_critico",
    "conteo_eventos_ip_ventana",
    "conteo_usuarios_distintos_ip",
    "conteo_hosts_distintos_ip",
    "conteo_fallidos_ventana",
    "servicio_sensible",
    "severidad_inicial",
    "resultado",
    "clase_evento_ml",
)


def generar_dataset_ml(
    total_eventos: int = TOTAL_EVENTOS,
    random_state: int = RANDOM_STATE,
    ruta_salida: str | Path = RUTA_DATASET_ML,
) -> list[dict[str, Any]]:
    """Genera y guarda un dataset deterministico para entrenar ML supervisado."""
    rng = random.Random(random_state)
    inicio = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)
    eventos = []

    for indice in range(1, total_eventos + 1):
        clase = _seleccionar_clase(rng)
        evento = _crear_evento(indice, clase, inicio, rng)
        eventos.append(evento)

    _guardar_csv(eventos, ruta_salida)
    return eventos


def _seleccionar_clase(rng: random.Random) -> str:
    """Mantiene una distribucion realista y no perfectamente balanceada."""
    return rng.choices(CLASES_EVENTO, weights=(0.58, 0.24, 0.18), k=1)[0]


def _crear_evento(
    indice: int,
    clase: str,
    inicio: datetime,
    rng: random.Random,
) -> dict[str, Any]:
    tipo_evento = rng.choice(TIPOS_POR_CLASE[clase])
    servicio = rng.choice(SERVICIOS_POR_TIPO[tipo_evento])
    usuario = _seleccionar_usuario(clase, rng)
    hostname = _seleccionar_hostname(clase, rng)
    hora_evento = _seleccionar_hora(clase, rng)
    timestamp = inicio + timedelta(minutes=indice * 3 + rng.randint(0, 2))
    timestamp = timestamp.replace(hour=hora_evento, minute=rng.randint(0, 59))

    ip_origen_es_privada = _seleccionar_ip_privada(clase, rng)
    ip_origen = rng.choice(IP_PRIVADAS if ip_origen_es_privada else IP_EXTERNAS_DOCUMENTACION)
    ip_destino = rng.choice(IP_DESTINO_LAB)

    usuario_privilegiado = usuario in USUARIOS_PRIVILEGIADOS
    host_critico = hostname in HOSTS_CRITICOS
    servicio_sensible = servicio in SERVICIOS_SENSIBLES
    es_horario_laboral = 8 <= hora_evento <= 18

    conteos = _generar_conteos(clase, rng)
    severidad_inicial = _seleccionar_severidad(
        clase=clase,
        servicio_sensible=servicio_sensible,
        host_critico=host_critico,
        rng=rng,
    )
    resultado = _seleccionar_resultado(clase, tipo_evento, rng)

    return {
        "id_evento": f"EVT-ML-{indice:04d}",
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "tipo_evento": tipo_evento,
        "servicio": servicio,
        "usuario": usuario,
        "hostname": hostname,
        "ip_origen": ip_origen,
        "ip_destino": ip_destino,
        "hora_evento": hora_evento,
        "es_horario_laboral": int(es_horario_laboral),
        "ip_origen_es_privada": int(ip_origen_es_privada),
        "usuario_privilegiado": int(usuario_privilegiado),
        "host_critico": int(host_critico),
        "conteo_eventos_ip_ventana": conteos["eventos_ip"],
        "conteo_usuarios_distintos_ip": conteos["usuarios_ip"],
        "conteo_hosts_distintos_ip": conteos["hosts_ip"],
        "conteo_fallidos_ventana": conteos["fallidos"],
        "servicio_sensible": int(servicio_sensible),
        "severidad_inicial": severidad_inicial,
        "resultado": resultado,
        "clase_evento_ml": clase,
    }


def _seleccionar_usuario(clase: str, rng: random.Random) -> str:
    if clase == "normal":
        return rng.choices(USUARIOS, weights=(2, 1, 1, 1, 2, 8, 10, 8, 3, 1, 3), k=1)[0]
    if clase == "fallido":
        return rng.choices(USUARIOS, weights=(5, 4, 2, 1, 4, 5, 6, 6, 3, 1, 3), k=1)[0]
    return rng.choices(USUARIOS, weights=(7, 7, 4, 4, 6, 2, 2, 2, 5, 3, 1), k=1)[0]


def _seleccionar_hostname(clase: str, rng: random.Random) -> str:
    if clase == "normal":
        return rng.choices(HOSTS, weights=(4, 2, 3, 4, 9), k=1)[0]
    if clase == "fallido":
        return rng.choices(HOSTS, weights=(6, 4, 6, 4, 3), k=1)[0]
    return rng.choices(HOSTS, weights=(7, 7, 8, 7, 1), k=1)[0]


def _seleccionar_hora(clase: str, rng: random.Random) -> int:
    if clase == "normal":
        return rng.choices(range(24), weights=_pesos_horarios(8, 18, 8, 1), k=1)[0]
    if clase == "fallido":
        return rng.choices(range(24), weights=_pesos_horarios(8, 18, 4, 3), k=1)[0]
    return rng.choices(range(24), weights=_pesos_horarios(8, 18, 2, 6), k=1)[0]


def _pesos_horarios(inicio_laboral: int, fin_laboral: int, peso_laboral: int, peso_fuera: int) -> list[int]:
    return [
        peso_laboral if inicio_laboral <= hora <= fin_laboral else peso_fuera
        for hora in range(24)
    ]


def _seleccionar_ip_privada(clase: str, rng: random.Random) -> bool:
    if clase == "normal":
        return rng.random() < 0.88
    if clase == "fallido":
        return rng.random() < 0.55
    return rng.random() < 0.28


def _generar_conteos(clase: str, rng: random.Random) -> dict[str, int]:
    if clase == "normal":
        return {
            "eventos_ip": rng.randint(1, 6),
            "usuarios_ip": rng.randint(1, 2),
            "hosts_ip": rng.randint(1, 2),
            "fallidos": rng.choices((0, 1, 2), weights=(12, 3, 1), k=1)[0],
        }
    if clase == "fallido":
        return {
            "eventos_ip": rng.randint(3, 14),
            "usuarios_ip": rng.randint(1, 4),
            "hosts_ip": rng.randint(1, 3),
            "fallidos": rng.randint(1, 9),
        }
    return {
        "eventos_ip": rng.randint(8, 35),
        "usuarios_ip": rng.randint(2, 9),
        "hosts_ip": rng.randint(2, 7),
        "fallidos": rng.randint(3, 24),
    }


def _seleccionar_severidad(
    clase: str,
    servicio_sensible: bool,
    host_critico: bool,
    rng: random.Random,
) -> str:
    if clase == "normal":
        return rng.choices(("Baja", "Media"), weights=(12, 2), k=1)[0]
    if clase == "fallido":
        return rng.choices(("Baja", "Media", "Alta"), weights=(4, 8, 2), k=1)[0]

    if servicio_sensible and host_critico:
        return rng.choices(("Media", "Alta", "Critica"), weights=(1, 6, 6), k=1)[0]
    return rng.choices(("Media", "Alta", "Critica"), weights=(3, 6, 3), k=1)[0]


def _seleccionar_resultado(clase: str, tipo_evento: str, rng: random.Random) -> str:
    """Genera trazabilidad operacional sin convertir resultado en feature ML."""
    if clase == "normal":
        return rng.choices(("permitido", "observado", "fallido"), weights=(9, 5, 1), k=1)[0]
    if clase == "fallido":
        return rng.choices(("fallido", "bloqueado", "observado"), weights=(9, 2, 1), k=1)[0]
    if tipo_evento == "ssh_authentication":
        return rng.choices(("fallido", "exitoso", "detectado"), weights=(7, 2, 2), k=1)[0]
    return rng.choices(("detectado", "bloqueado", "observado", "fallido"), weights=(7, 3, 2, 2), k=1)[0]


def _guardar_csv(eventos: list[dict[str, Any]], ruta_salida: str | Path) -> None:
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_DATASET)
        escritor.writeheader()
        escritor.writerows(eventos)


if __name__ == "__main__":
    eventos_generados = generar_dataset_ml()
    print(f"Dataset ML generado: {RUTA_DATASET_ML}")
    print(f"Eventos generados: {len(eventos_generados)}")
