"""Capa de mapeo MITRE ATT&CK."""

from __future__ import annotations


MITRE_MAPPING = {
    "SSH Brute Force": {
        "tactica": "Credential Access",
        "tecnica": "Brute Force",
        "id_tecnica": "T1110",
        "subtecnica": "Password Guessing",
        "id_subtecnica": "T1110.001",
    },
    "Unauthorized Access Attempt": {
        "tactica": "Credential Access",
        "tecnica": "Brute Force",
        "id_tecnica": "T1110",
        "subtecnica": "Password Guessing",
        "id_subtecnica": "T1110.001",
    },
    "Suspicious Login Activity": {
        "tactica": "Initial Access",
        "tecnica": "Valid Accounts",
        "id_tecnica": "T1078",
        "subtecnica": "N/A",
        "id_subtecnica": "N/A",
    },
    "Unknown Event": {
        "tactica": "N/A",
        "tecnica": "N/A",
        "id_tecnica": "N/A",
        "subtecnica": "N/A",
        "id_subtecnica": "N/A",
    },
}


def mapear_mitre(tipo_incidente: str) -> dict[str, str]:
    """Devuelve el mapeo MITRE y conserva un punto de extensión simple."""
    return MITRE_MAPPING.get(tipo_incidente, MITRE_MAPPING["Unknown Event"])
