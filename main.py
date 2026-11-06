"""Punto de entrada del MVP académico SOAR-AI."""

from __future__ import annotations

import sys
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from src.telemetria_red import anexar_telemetria, mediciones_json
from src.politica_red import (decidir_revision, integrar_escalamiento_red,
                              severidad_flujo, playbook_flujo)

try:
    from src.ml_predictor import predecir_incidente_ml
except Exception:
    predecir_incidente_ml = None

try:
    from src.qwen_agent import analizar_con_qwen3 as analizar_con_qwen3_real
except Exception:
    analizar_con_qwen3_real = None


RUTA_ALERTAS = DATA_DIR / "alertas_simuladas.json"
RUTA_REPORTES = DATA_DIR / "reportes_generados.json"


def _intentar_prediccion_ml(incidente: dict, severidad: str, ruta_modelo_red=None) -> dict[str, Any]:
    """Intenta enriquecer el reporte con ML sin interrumpir el MVP."""
    if predecir_incidente_ml is None:
        return {
            "estado": "error",
            "detalle": "No se pudo importar la capa ML de prediccion.",
            "modelo": "RandomForestClassifier",
        }

    try:
        if ruta_modelo_red is not None:
            return predecir_incidente_ml(incidente, severidad_calculada=severidad,
                                        ruta_modelo_red=ruta_modelo_red)
        return predecir_incidente_ml(incidente, severidad_calculada=severidad)
    except Exception as error:
        return {
            "estado": "error",
            "detalle": str(error),
            "modelo": "RandomForestClassifier",
        }


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


def ejecutar_pipeline(imprimir: bool = True, *, alertas: list[dict] | None = None,
                      ruta_reportes=None, usar_qwen: bool = True, ruta_modelo_red=None) -> list[dict]:
    """Ejecuta el flujo completo de análisis SOAR-AI con arquitectura Híbrida."""
    alertas = cargar_json(RUTA_ALERTAS) if alertas is None else alertas
    ruta_reportes = RUTA_REPORTES if ruta_reportes is None else ruta_reportes
    incidentes = detectar_incidentes(alertas)
    anexar_telemetria(incidentes, alertas)
    reportes = []

    print(f"\nIniciando procesamiento de {len(incidentes)} incidentes...")

    for incidente in incidentes:
        es_flujo_red = incidente.get("tipo_detector") == "network_flow_native"
        tipo_incidente = clasificar_incidente(incidente)
        severidad = severidad_flujo(incidente) if es_flujo_red else calcular_severidad(incidente)
        mitre = mapear_mitre(tipo_incidente)
        recomendaciones = recomendar_playbook(tipo_incidente, severidad, incidente)
        escalamiento = evaluar_escalamiento(incidente, severidad)
        analisis_qwen3_placeholder = {} if es_flujo_red else analizar_con_qwen3_placeholder(incidente)

        # 1. Ejecutamos el cerebro matemático (Rápido)
        prediccion_ml = _intentar_prediccion_ml(incidente, severidad, ruta_modelo_red)
        decision = decidir_revision(prediccion_ml, severidad, escalamiento)
        if (es_flujo_red or prediccion_ml.get("representacion") == "red_original_12"
                or prediccion_ml.get("estado") != "ok"):
            escalamiento = integrar_escalamiento_red(escalamiento, prediccion_ml, severidad)
        if es_flujo_red:
            recomendaciones = playbook_flujo()

        reporte = construir_reporte_incidente(
            incidente=incidente,
            tipo_incidente=tipo_incidente,
            severidad=severidad,
            mitre=mitre,
            recomendaciones=recomendaciones,
            escalamiento=escalamiento,
            analisis_qwen3_placeholder=analisis_qwen3_placeholder,
        )

        reporte["prediccion_ml"] = prediccion_ml
        if es_flujo_red:
            reporte["unidad_analisis"] = "flujo de red"
            reporte["telemetria_red"] = mediciones_json(incidente.get("telemetria_red", {}))
            reporte["procedencia_red"] = {k:incidente.get(k) for k in
                                          ["archivo_captura", "fila_captura_cero_based"]}
        reporte["decision_analisis"] = decision
        reporte["contexto_historico"] = construir_contexto_historico(reporte, reportes)

        # El análisis generativo respeta las reglas independientes y puede
        # dejarse bajo demanda durante la validación de lotes.
        if decision["habilitar_analisis_avanzado"]:
            if usar_qwen:
                print("Análisis avanzado solicitado por ML o reglas de escalamiento...")
                reporte["analisis_qwen3_real"] = _intentar_analisis_qwen3_real(reporte)
            else:
                reporte["analisis_qwen3_real"] = {
                    "estado": "pendiente_bajo_demanda",
                    "motivo": "Análisis avanzado disponible bajo demanda para revisión.",
                }
        else:
            reporte["analisis_qwen3_real"] = {
                "estado": "omitido",
                "motivo": "Clasificación normal sin criterios independientes de revisión.",
            }

        reportes.append(reporte)

    guardar_json(reportes, ruta_reportes)

    if imprimir:
        imprimir_reporte(reportes)
        print(f"\nReporte JSON final guardado en: {ruta_reportes}")

    return reportes


if __name__ == "__main__":
    ejecutar_pipeline()
