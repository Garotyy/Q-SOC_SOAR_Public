"""Generación de reportes estructurados para SOAR-AI."""

from __future__ import annotations

import json
from typing import Any


def construir_reporte_incidente(
    incidente: dict[str, Any],
    tipo_incidente: str,
    severidad: str,
    mitre: dict[str, str],
    recomendaciones: list[str],
    escalamiento: dict[str, Any],
    analisis_qwen3_placeholder: dict[str, Any],
) -> dict[str, Any]:
    """Construye el registro final de reporte requerido por el MVP."""
    return {
        "id_alerta": incidente.get("id_alerta"),
        "timestamp": incidente.get("timestamp"),
        "ip_origen": incidente.get("ip_origen"),
        "ip_destino": incidente.get("ip_destino"),
        "hostname": incidente.get("hostname"),
        "usuario": incidente.get("usuario"),
        "tipo_incidente": tipo_incidente,
        "severidad": severidad,
        "táctica_mitre": mitre.get("tactica", "N/A"),
        "técnica_mitre": mitre.get("tecnica", "N/A"),
        "id_técnica": mitre.get("id_tecnica", "N/A"),
        "subtécnica_mitre": mitre.get("subtecnica", "N/A"),
        "id_subtécnica": mitre.get("id_subtecnica", "N/A"),
        "evidencia": incidente.get("evidencia", []),
        "recomendación_playbook": recomendaciones,
        "requiere_escalamiento": escalamiento.get("requiere_escalamiento", False),
        "nivel_escalamiento": escalamiento.get("nivel_escalamiento", "No requerido"),
        "prioridad": escalamiento.get("prioridad", "P4"),
        "urgencia": escalamiento.get("urgencia", "Baja"),
        "justificación": escalamiento.get("justificacion", "N/A"),
        "eventos_relacionados": incidente.get("eventos_relacionados", []),
        "contexto_historico": {},
        "analisis_qwen3_placeholder": analisis_qwen3_placeholder,
        "analisis_qwen3_real": None,
    }


def imprimir_reporte(reportes: list[dict[str, Any]]) -> None:
    """Imprime un reporte legible en consola."""
    print("\n=== REPORTE SOAR-AI MVP ===")
    print(f"Incidentes generados: {len(reportes)}")

    for reporte in reportes:
        usuarios = reporte.get("usuario")
        if isinstance(usuarios, list):
            usuarios = ", ".join(usuarios)

        print("\n" + "-" * 72)
        print(
            f"{reporte.get('id_alerta')} | {reporte.get('tipo_incidente')} | "
            f"Severidad: {reporte.get('severidad')}"
        )
        print(
            f"Origen: {reporte.get('ip_origen')} -> "
            f"Destino: {reporte.get('hostname')} ({reporte.get('ip_destino')})"
        )
        print(f"Usuario(s): {usuarios}")
        print(
            "MITRE ATT&CK: "
            f"{reporte.get('táctica_mitre')} / {reporte.get('técnica_mitre')} "
            f"({reporte.get('id_técnica')})"
        )
        print("Evidencia:")
        for evidencia in reporte.get("evidencia", []):
            print(f"  - {evidencia}")
        print("Playbook recomendado:")
        for accion in reporte.get("recomendación_playbook", []):
            print(f"  - {accion}")
        print(
            "Escalamiento: "
            f"{reporte.get('requiere_escalamiento')} | "
            f"{reporte.get('nivel_escalamiento')} | "
            f"{reporte.get('prioridad')} | {reporte.get('urgencia')}"
        )
        print(f"Justificación: {reporte.get('justificación')}")
        print("Contexto histórico:")
        _imprimir_contexto_historico(reporte.get("contexto_historico"))
        print("Análisis IA Qwen3:")
        _imprimir_analisis_qwen3(reporte.get("analisis_qwen3_real"))


def _imprimir_contexto_historico(contexto: Any) -> None:
    """Imprime el contexto histórico en un formato compacto para SOC."""
    if not isinstance(contexto, dict) or not contexto:
        print("  - No disponible en el reporte.")
        return

    print(f"  - {contexto.get('resumen_contexto', 'No disponible en el reporte')}")
    print(
        "  - Relacionados previos: "
        f"{contexto.get('incidentes_relacionados_previos', 0)} | "
        f"Críticos previos: {contexto.get('incidentes_criticos_previos', 0)} | "
        f"Patrón repetido: {contexto.get('patron_repetido', False)}"
    )


def _imprimir_analisis_qwen3(analisis: Any) -> None:
    """Imprime de forma legible la respuesta JSON estructurada de Qwen3."""
    if not analisis:
        print("  - No disponible; se conserva el placeholder académico.")
        return

    if not isinstance(analisis, dict):
        print(f"  - {_resumir_texto(analisis)}")
        return

    if analisis.get("error"):
        print(f"  - Error: {analisis.get('error')}")
        respuesta_original = analisis.get("respuesta_original")
        if respuesta_original:
            print(f"  - Respuesta original: {_resumir_texto(respuesta_original)}")
        return

    campos_texto = (
        ("Resumen ejecutivo", "resumen_ejecutivo"),
        ("Análisis de riesgo", "analisis_riesgo"),
        ("Explicación MITRE ATT&CK", "explicacion_mitre"),
        ("Uso del contexto histórico", "uso_contexto_historico"),
        ("Siguiente acción sugerida", "siguiente_accion_sugerida"),
    )
    for etiqueta, clave in campos_texto:
        print(f"  - {etiqueta}: {analisis.get(clave, 'No disponible en el reporte')}")

    _imprimir_lista_analisis("Hechos observados", analisis.get("hechos_observados", []))
    _imprimir_lista_analisis(
        "Inferencias razonables", analisis.get("inferencias_razonables", [])
    )
    _imprimir_lista_analisis("Recomendación SOC", analisis.get("recomendacion_soc", []))
    _imprimir_lista_analisis(
        "Posibles falsos positivos", analisis.get("posibles_falsos_positivos", [])
    )


def _imprimir_lista_analisis(etiqueta: str, valores: Any) -> None:
    """Imprime una lista del análisis IA con fallback legible."""
    print(f"  - {etiqueta}:")
    if not isinstance(valores, list) or not valores:
        print("    * No disponible en el reporte")
        return

    for item in valores:
        print(f"    * {item}")


def _resumir_texto(valor: Any, limite: int = 450) -> str:
    """Mantiene compacta en consola una salida LLM potencialmente larga."""
    if isinstance(valor, dict):
        texto = json.dumps(valor, ensure_ascii=False)
    else:
        texto = str(valor)

    texto = " ".join(texto.split())
    if len(texto) <= limite:
        return texto
    return texto[: limite - 3] + "..."
