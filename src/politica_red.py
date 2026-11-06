"""Política de revisión y análisis asistido, sin acciones de contención."""
import unicodedata


def normalizar(texto):
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().strip().lower()


def decidir_revision(prediccion, severidad, escalamiento):
    razones = []
    clase = normalizar(prediccion.get("clase_predicha", ""))
    if prediccion.get("estado") != "ok" or clase not in {"normal", "fallido", "sospechoso"}:
        razones.append("predicción ML no disponible o inválida")
    elif clase in {"fallido", "sospechoso"}:
        razones.append("clasificación ML que requiere revisión")
    if normalizar(severidad) in {"alta", "critica", "high", "critical"}:
        razones.append("severidad alta o crítica por reglas")
    if escalamiento.get("requiere_escalamiento"):
        razones.append("criterio independiente de escalamiento")
    return {"requiere_revision": bool(razones), "habilitar_analisis_avanzado": bool(razones),
            "razones": razones, "tipo_accion": "revisión recomendada; sin ejecución de contención"}


def integrar_escalamiento_red(escalamiento, prediccion, severidad):
    resultado = dict(escalamiento)
    decision = decidir_revision(prediccion, severidad, escalamiento)
    if decision["requiere_revision"] and not resultado.get("requiere_escalamiento"):
        resultado.update({"requiere_escalamiento": True, "nivel_escalamiento": "SOC Nivel 1",
                          "prioridad": "P3", "urgencia": "Media",
                          "justificacion": "; ".join(decision["razones"])+"."})
    return resultado


def severidad_flujo(incidente):
    return {"baja":"Baja", "media":"Media", "alta":"Alta", "critica":"Crítica",
            "low":"Baja", "medium":"Media", "high":"Alta", "critical":"Crítica"}.get(
        normalizar(incidente.get("severidad_observada", "")), "No determinada")


def playbook_flujo():
    return ["Revisar las mediciones del flujo y corroborar la alerta con otras fuentes.",
            "Obtener IP, host y contexto temporal cuando estén disponibles.",
            "Asignar técnica MITRE y acciones específicas solo cuando exista evidencia suficiente.",
            "Documentar la decisión del analista; no se ejecutan bloqueos automáticamente."]
