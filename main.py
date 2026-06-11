"""Punto de entrada del MVP académico SOAR-AI."""

from __future__ import annotations

from typing import Any

from src.clasificador import clasificar_incidente
from src.detector import detectar_incidentes
from src.escalamiento import evaluar_escalamiento
from src.memoria_contexto import construir_contexto_historico
from src.mitre_mapper import mapear_mitre
from src.playbooks import recomendar_playbook
from src.qwen_agent_placeholder import analizar_con_qwen3 as analizar_con_qwen3_placeholder
from src.report_generator import construir_reporte_incidente, imprimir_reporte
from src.severity_engine import calcular_severidad
from src.utils import DATA_DIR, cargar_json, guardar_json

try:
    from src.qwen_agent import analizar_con_qwen3 as analizar_con_qwen3_real
except Exception:
    analizar_con_qwen3_real = None


RUTA_ALERTAS = DATA_DIR / "alertas_simuladas.json"
RUTA_REPORTES = DATA_DIR / "reportes_generados.json"


def _intentar_analisis_qwen3_real(reporte: dict) -> dict[str, Any]:
    """Intenta usar Qwen3/Ollama real sin interrumpir el MVP."""
    if analizar_con_qwen3_real is None:
        return {
            "error": "qwen3_no_disponible",
            "detalle": "No se pudo importar la integración real de Qwen3.",
        }

    try:
        respuesta = analizar_con_qwen3_real(reporte)
    except Exception as error:
        return {
            "error": "qwen3_no_disponible",
            "detalle": str(error),
        }

    if not respuesta:
        return {
            "error": "respuesta_vacia",
            "respuesta_original": "",
        }
    if isinstance(respuesta, str):
        return {
            "error": "respuesta_string_no_valida",
            "respuesta_original": respuesta,
        }
    if not isinstance(respuesta, dict):
        return {
            "error": "respuesta_no_json",
            "respuesta_original": str(respuesta),
        }
    return respuesta


def ejecutar_pipeline(imprimir: bool = True) -> list[dict]:
    """Ejecuta el flujo completo de análisis SOAR-AI."""
    alertas = cargar_json(RUTA_ALERTAS)
    incidentes = detectar_incidentes(alertas)
    reportes = []

    for incidente in incidentes:
        tipo_incidente = clasificar_incidente(incidente)
        severidad = calcular_severidad(incidente)
        mitre = mapear_mitre(tipo_incidente)
        recomendaciones = recomendar_playbook(tipo_incidente, severidad, incidente)
        escalamiento = evaluar_escalamiento(incidente, severidad)
        analisis_qwen3_placeholder = analizar_con_qwen3_placeholder(incidente)

        reporte = construir_reporte_incidente(
            incidente=incidente,
            tipo_incidente=tipo_incidente,
            severidad=severidad,
            mitre=mitre,
            recomendaciones=recomendaciones,
            escalamiento=escalamiento,
            analisis_qwen3_placeholder=analisis_qwen3_placeholder,
        )

        # Enriquecimiento de memoria: compara el reporte actual con los reportes
        # previos ya generados para que Qwen3 reciba contexto histórico.
        reporte["contexto_historico"] = construir_contexto_historico(
            reporte, reportes
        )
        reporte["analisis_qwen3_real"] = _intentar_analisis_qwen3_real(reporte)
        reportes.append(reporte)

    guardar_json(reportes, RUTA_REPORTES)

    if imprimir:
        imprimir_reporte(reportes)
        print(f"\nReporte JSON guardado en: {RUTA_REPORTES}")

    return reportes


if __name__ == "__main__":
    ejecutar_pipeline()
