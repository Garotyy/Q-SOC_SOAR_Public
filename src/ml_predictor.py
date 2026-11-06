"""Predicción ML aislada para enriquecer reportes SOAR-AI.

El predictor carga el pipeline Random Forest previamente entrenado y devuelve
un bloque JSON auditable. No reemplaza reglas, MITRE, severidad, playbooks ni
análisis IA.
"""

from __future__ import annotations

import ipaddress
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.generador_dataset_ml import (
    HOSTS_CRITICOS,
    SERVICIOS_SENSIBLES,
    USUARIOS_PRIVILEGIADOS,
)
from src.utils import parsear_timestamp
from src.telemetria_red import tiene_telemetria_red

try:
    import joblib
    import pandas as pd
    from src.random_forest import (
        COLUMNAS_CATEGORICAS,
        COLUMNAS_NUMERICAS,
        RUTA_MODELO,
    )
except Exception as import_error:
    joblib = None
    pd = None
    IMPORT_ERROR = import_error
    COLUMNAS_CATEGORICAS = [
        "usuario",
        "servicio",
    ]
    COLUMNAS_NUMERICAS = [
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
    ]
    RUTA_MODELO = Path(__file__).resolve().parent.parent / "data" / "modelo_random_forest.joblib"
else:
    IMPORT_ERROR = None


MODELO_ML = "RandomForestClassifier (CICIDS2017 Kali)"
FEATURES_USADAS = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS
REDES_PRIVADAS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def predecir_incidente_ml(
    incidente: dict[str, Any],
    severidad_calculada: str | None = None,
    ruta_modelo: str | Path = RUTA_MODELO,
    ruta_modelo_red: str | Path | None = None,
) -> dict[str, Any]:
    """Predice la clase ML de un incidente sin interrumpir el pipeline."""
    if tiene_telemetria_red(incidente):
        from src.ml_predictor_red import predecir_incidente_red
        if ruta_modelo_red is not None:
            return predecir_incidente_red(incidente, ruta_modelo_red)
        return predecir_incidente_red(incidente)
    try:
        pipeline = _obtener_pipeline_entrenado(str(Path(ruta_modelo)))
        features = _construir_features_incidente(incidente, severidad_calculada)
        datos_prediccion = pd.DataFrame([features], columns=FEATURES_USADAS)

        clase_predicha = str(pipeline.predict(datos_prediccion)[0])
        probabilidades = _calcular_probabilidades(pipeline, datos_prediccion)

        return {
            "modelo": MODELO_ML,
            "representacion": "contextual_homologada",
            "clase_predicha": clase_predicha,
            "confianza": probabilidades.get(clase_predicha, 0.0),
            "probabilidades": probabilidades,
            "features_usadas": FEATURES_USADAS,
            "estado": "ok",
        }
    except Exception as error:
        return {
            "estado": "error",
            "detalle": str(error),
            "modelo": MODELO_ML,
        }


@lru_cache(maxsize=1)
def _obtener_pipeline_entrenado(ruta_modelo: str) -> Any:
    """Carga una sola vez el pipeline Random Forest ya entrenado."""
    if IMPORT_ERROR is not None or pd is None or joblib is None:
        raise RuntimeError(f"No se pudo cargar la capa ML: {IMPORT_ERROR}")

    ruta = Path(ruta_modelo)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el modelo Random Forest requerido: {ruta}. "
            "Ejecuta primero: python -m src.random_forest"
        )

    pipeline = joblib.load(ruta)
    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        raise TypeError("El archivo cargado no contiene un pipeline ML válido.")
    return pipeline


def _construir_features_incidente(
    incidente: dict[str, Any], severidad_calculada: str | None
) -> dict[str, Any]:
    """Deriva las features esperadas por Random Forest desde un incidente SOAR."""
    timestamp = parsear_timestamp(incidente.get("timestamp"))
    hora_evento = int(timestamp.hour)
    usuario = _normalizar_usuario(incidente.get("usuario"))
    usuarios = _normalizar_usuarios(incidente.get("usuario"))
    usuarios_atacados = _normalizar_usuarios(incidente.get("usuarios_atacados"))
    usuarios_distintos = usuarios_atacados or usuarios
    servicio = str(incidente.get("servicio") or "desconocido").lower()
    hostname = str(incidente.get("hostname") or "desconocido")
    ip_origen = str(incidente.get("ip_origen") or "")
    eventos_relacionados = incidente.get("eventos_relacionados") or []

    return {
        "tipo_evento": str(incidente.get("tipo_evento") or "desconocido"),
        "servicio": servicio,
        "usuario": usuario,
        "hostname": hostname,
        "severidad_inicial": _normalizar_severidad(severidad_calculada),
        "hora_evento": hora_evento,
        "es_horario_laboral": int(8 <= hora_evento <= 18),
        "ip_origen_es_privada": int(_es_ip_privada_laboratorio(ip_origen)),
        "usuario_privilegiado": int(usuario in USUARIOS_PRIVILEGIADOS),
        "host_critico": int(hostname in HOSTS_CRITICOS),
        "conteo_eventos_ip_ventana": _conteo_eventos_ip(
            incidente, eventos_relacionados
        ),
        "conteo_usuarios_distintos_ip": max(1, len(set(usuarios_distintos))),
        "conteo_hosts_distintos_ip": 1,
        "conteo_fallidos_ventana": int(
            incidente.get("conteo_intentos_fallidos", 0) or 0
        ),
        "servicio_sensible": int(servicio in SERVICIOS_SENSIBLES),
    }


def _calcular_probabilidades(pipeline: Any, datos_prediccion: Any) -> dict[str, float]:
    clases_modelo = [str(clase) for clase in pipeline.named_steps["modelo"].classes_]
    valores = pipeline.predict_proba(datos_prediccion)[0]
    return {
        clase: round(float(probabilidad), 4)
        for clase, probabilidad in zip(clases_modelo, valores)
    }


def _normalizar_usuario(valor: Any) -> str:
    usuarios = _normalizar_usuarios(valor)
    if not usuarios:
        return "desconocido"
    return usuarios[0]


def _normalizar_usuarios(valor: Any) -> list[str]:
    if isinstance(valor, list):
        candidatos = valor
    elif valor in (None, "", "N/A"):
        candidatos = []
    else:
        candidatos = [valor]

    return sorted(
        {
            str(usuario).strip()
            for usuario in candidatos
            if str(usuario).strip() and str(usuario).strip().upper() != "N/A"
        }
    )


def _normalizar_severidad(severidad: str | None) -> str:
    if not severidad:
        return "Baja"

    texto = str(severidad).strip()
    texto_ascii = (
        unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    )
    normalizado = texto_ascii.lower()
    equivalencias = {
        "critica": "Critica",
        "cratica": "Critica",
        "alta": "Alta",
        "media": "Media",
        "baja": "Baja",
    }
    return equivalencias.get(normalizado, texto)


def _es_ip_privada_laboratorio(ip_origen: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_origen)
    except ValueError:
        return False
    return any(ip in red for red in REDES_PRIVADAS)


def _conteo_eventos_ip(incidente: dict[str, Any], eventos_relacionados: list[Any]) -> int:
    repeticiones = incidente.get("repeticiones_ip_origen")
    if repeticiones not in (None, ""):
        return int(repeticiones)
    return max(1, len(eventos_relacionados))
