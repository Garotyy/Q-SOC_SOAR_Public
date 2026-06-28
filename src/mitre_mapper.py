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
    "Data Exfiltration": {
        "tactica": "Exfiltration",
        "tecnica": "Exfiltration Over C2 Channel",
        "id_tecnica": "T1041",
        "subtecnica": "N/A",
        "id_subtecnica": "N/A",
    },
    "Data Exfiltration via Covert Channel": {
        "tactica": "Exfiltration",
        "tecnica": "Exfiltration Over Alternative Protocol",
        "id_tecnica": "T1048",
        "subtecnica": "Exfiltration Over Unencrypted Non-C2 Protocol",
        "id_subtecnica": "T1048.003",
    },
    "Suspicious Network Connection": {
        "tactica": "Command and Control",
        "tecnica": "Application Layer Protocol",
        "id_tecnica": "T1071",
        "subtecnica": "N/A",
        "id_subtecnica": "N/A",
    },
    "Network Activity Observed": {
        "tactica": "Discovery",
        "tecnica": "Network Service Discovery",
        "id_tecnica": "T1046",
        "subtecnica": "N/A",
        "id_subtecnica": "N/A",
    },
    "Suspicious DNS Activity": {
        "tactica": "Command and Control",
        "tecnica": "Application Layer Protocol",
        "id_tecnica": "T1071",
        "subtecnica": "DNS",
        "id_subtecnica": "T1071.004",
    },
    "DNS Query Observed": {
        "tactica": "Discovery",
        "tecnica": "Network Service Discovery",
        "id_tecnica": "T1046",
        "subtecnica": "N/A",
        "id_subtecnica": "N/A",
    },
    "Endpoint Compromise - Execution": {
        "tactica": "Execution",
        "tecnica": "Command and Scripting Interpreter",
        "id_tecnica": "T1059",
        "subtecnica": "PowerShell",
        "id_subtecnica": "T1059.001",
    },
    "Suspicious PowerShell Activity": {
        "tactica": "Execution",
        "tecnica": "Command and Scripting Interpreter",
        "id_tecnica": "T1059",
        "subtecnica": "PowerShell",
        "id_subtecnica": "T1059.001",
    },
    "Endpoint Compromise - Staging": {
        "tactica": "Collection",
        "tecnica": "Data Staged",
        "id_tecnica": "T1074",
        "subtecnica": "Local Data Staging",
        "id_subtecnica": "T1074.001",
    },
    "Suspicious Endpoint Activity": {
        "tactica": "Defense Evasion",
        "tecnica": "Masquerading",
        "id_tecnica": "T1036",
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