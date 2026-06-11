"""Playbooks reutilizables de respuesta simulada.

Todas las acciones son recomendaciones para un prototipo académico SOC/SOAR.
Este módulo no ejecuta contención, bloqueos, cambios de credenciales ni acciones
de red.
"""

from __future__ import annotations

from typing import Any


BASE_PLAYBOOKS = {
    "SSH Brute Force": [
        "Simular bloqueo temporal de la IP origen en controles perimetrales.",
        "Revisar logs de autenticación SSH en el host afectado.",
        "Verificar si existieron accesos exitosos posteriores al patrón de fallos.",
        "Validar si los usuarios atacados usan contraseñas débiles o reutilizadas.",
        "Forzar cambio de contraseña para cuentas comprometidas o de alto riesgo.",
        "Habilitar o reforzar MFA para cuentas administrativas.",
        "Aumentar monitoreo sobre la IP origen, host destino y usuarios atacados.",
    ],
    "Unauthorized Access Attempt": [
        "Revisar logs de autenticación para confirmar si el intento fue aislado.",
        "Correlacionar la IP origen con alertas recientes del laboratorio.",
        "Aumentar monitoreo temporal sobre el usuario afectado.",
        "Recomendar MFA si el usuario no lo tiene habilitado.",
    ],
    "Suspicious Login Activity": [
        "Verificar si el login exitoso fue legítimo con el dueño de la cuenta.",
        "Revisar comandos ejecutados después del acceso.",
        "Buscar persistencia, cambios de llaves SSH o creación de usuarios.",
        "Forzar cambio de contraseña si no se puede validar legitimidad.",
        "Notificar a un analista SOC para revisión manual.",
    ],
    "Unknown Event": [
        "Registrar el evento para análisis posterior.",
        "Revisar si existe un detector aplicable o si se requiere crear uno nuevo.",
        "Mantener el evento en cola de triage de baja prioridad.",
    ],
}


def recomendar_playbook(
    tipo_incidente: str, severidad: str, incidente: dict[str, Any]
) -> list[str]:
    """Devuelve acciones recomendadas de respuesta simulada."""
    acciones = list(BASE_PLAYBOOKS.get(tipo_incidente, BASE_PLAYBOOKS["Unknown Event"]))

    if severidad in {"Alta", "Crítica"}:
        acciones.append("Notificar analista SOC y documentar decisión de escalamiento.")
    if incidente.get("login_exitoso_posterior"):
        acciones.append("Priorizar revisión de accesos exitosos y posible compromiso.")
    if len(incidente.get("usuarios_atacados", [])) > 1:
        acciones.append("Evaluar campaña de password spraying contra múltiples cuentas.")

    return acciones
