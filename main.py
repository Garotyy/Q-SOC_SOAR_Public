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
    from src.ml_predictor import predecir_incidente_ml
except Exception:
    predecir_incidente_ml = None

try:
    from src.qwen_agent import analizar_con_qwen3 as analizar_con_qwen3_real
except Exception:
    analizar_con_qwen3_real = None


RUTA_ALERTAS = DATA_DIR / "alertas_simuladas.json"
RUTA_REPORTES = DATA_DIR / "reportes_generados.json"


def _intentar_prediccion_ml(incidente: dict, severidad: str) -> dict[str, Any]:
    """Intenta enriquecer el reporte con ML sin interrumpir el MVP."""
    if predecir_incidente_ml is None:
        return {
            "estado": "error",
            "detalle": "No se pudo importar la capa ML de prediccion.",
            "modelo": "DecisionTreeClassifier",
        }

    try:
        return predecir_incidente_ml(incidente, severidad_calculada=severidad)
    except Exception as error:
        return {
            "estado": "error",
            "detalle": str(error),
            "modelo": "DecisionTreeClassifier",
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


def ejecutar_pipeline(imprimir: bool = True) -> list[dict]:
    """Ejecuta el flujo completo de análisis SOAR-AI con arquitectura Híbrida."""
    alertas = cargar_json(RUTA_ALERTAS)
    incidentes = detectar_incidentes(alertas)
    reportes = []

    print(f"\nIniciando procesamiento de {len(incidentes)} incidentes...")

    for incidente in incidentes:
        tipo_incidente = clasificar_incidente(incidente)
        severidad = calcular_severidad(incidente)
        mitre = mapear_mitre(tipo_incidente)
        recomendaciones = recomendar_playbook(tipo_incidente, severidad, incidente)
        escalamiento = evaluar_escalamiento(incidente, severidad)
        analisis_qwen3_placeholder = analizar_con_qwen3_placeholder(incidente)

        # 1. Ejecutamos el cerebro matemático (Rápido)
        prediccion_ml = _intentar_prediccion_ml(incidente, severidad)

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
        reporte["contexto_historico"] = construir_contexto_historico(reporte, reportes)

# =================================================================
        # LÓGICA TWO-STAGE (Con diseño Fail-Secure ante fallos de infraestructura)
        # =================================================================
        clase_ml = prediccion_ml.get("clase_predicha", "").lower() 
        estado_ml = prediccion_ml.get("estado", "ok") # Revisamos si el ML reportó error

        # Activamos Qwen si:
        # 1. El ML explícitamente dice que es peligroso (sospechoso/fallido)
        # 2. La severidad por reglas es alta/crítica
        # 3. ¡NUEVO! La capa de ML falló (estado == "error"), por lo que requerimos respaldo de la IA
        if estado_ml == "error" or clase_ml in ["sospechoso", "fallido"] or severidad in ["alta", "critica", "high", "critical"]:
            
            if estado_ml == "error":
                print(f"⚠️ Capa ML falló. Activando Qwen como respaldo de seguridad (Fail-Secure)...")
            else:
                print(f"🧠 ML detectó riesgo ({clase_ml}). Despertando a Qwen para análisis profundo...")
                
            reporte["analisis_qwen3_real"] = _intentar_analisis_qwen3_real(reporte)
            
        else:
            # Aquí solo entra si el ML funcionó correctamente Y determinó que es NORMAL
            print(f"⚡ ML determinó que es NORMAL. Omitiendo Qwen para ahorrar recursos.")
            reporte["analisis_qwen3_real"] = {
                "estado": "omitido",
                "motivo": "Filtrado en Etapa 1 (Machine Learning clasificó como NORMAL o Severidad Baja)",
                "accion_sugerida": "PERMITIR"
            }

        reportes.append(reporte)

    guardar_json(reportes, RUTA_REPORTES)

    if imprimir:
        imprimir_reporte(reportes)
        print(f"\nReporte JSON final guardado en: {RUTA_REPORTES}")

    return reportes


if __name__ == "__main__":
    ejecutar_pipeline()
