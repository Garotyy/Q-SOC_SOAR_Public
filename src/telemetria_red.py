"""Contrato de las doce mediciones de red; sin etiquetas de entrenamiento."""
from pathlib import Path

VARIABLES_RED = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Mean",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count",
]
CLASES_RED = ["fallido", "normal", "sospechoso"]
RUTA_MODELO_RED = Path(__file__).resolve().parents[1]/"data/modelo_random_forest_red.joblib"
FORMATO_MODELO_RED = "qsoc.rf_red.v1"


def tiene_telemetria_red(incidente):
    return ("telemetria_red" in incidente or "flujos_red" in incidente
            or any(c in incidente for c in VARIABLES_RED))


def extraer_flujos(incidente):
    if "flujos_red" in incidente:
        flujos = incidente["flujos_red"]
        if not isinstance(flujos, list) or not flujos or not all(isinstance(f, dict) for f in flujos):
            raise ValueError("flujos_red debe contener al menos un flujo con sus mediciones.")
        return flujos
    flujo = incidente.get("telemetria_red", incidente)
    if not isinstance(flujo, dict):
        raise ValueError("telemetria_red debe ser un objeto de mediciones.")
    return [flujo]


def matriz_red(datos):
    """Mismo float32 y recorte de duración que en la comparación verificada.

    No completa ausentes con ceros ni utiliza Label, IP, fecha o usuario.
    Devuelve una máscara para dirigir entradas inválidas a revisión.
    """
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(datos).reindex(columns=VARIABLES_RED)
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Flow Duration"] = df["Flow Duration"].clip(lower=0)
    puertos = df["Destination Port"].to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        valores = df.to_numpy(dtype=np.float32, na_value=np.nan)
    validas = np.isfinite(valores).all(axis=1)
    validas &= np.isfinite(puertos) & (puertos >= 0) & (puertos <= 65535) & (puertos == np.floor(puertos))
    return valores, validas


def anexar_telemetria(incidentes, alertas):
    """Conserva mediciones por flujo al agrupar alertas; no las promedia."""
    por_id = {}
    for alerta in alertas:
        identificador = alerta.get("id_alerta")
        if identificador is not None and tiene_telemetria_red(alerta):
            por_id.setdefault(identificador, []).append(alerta)
    for incidente in incidentes:
        if tiene_telemetria_red(incidente):
            continue
        flujos = []
        try:
            for identificador in incidente.get("eventos_relacionados", []):
                for alerta in por_id.get(identificador, []):
                    flujos.extend(extraer_flujos(alerta))
        except ValueError as error:
            incidente["flujos_red"] = []
            incidente["error_telemetria_red"] = str(error)
            continue
        if flujos:
            incidente["flujos_red"] = flujos
    return incidentes


def crear_alerta_flujo(mediciones, indice, archivo, metadatos=None):
    import pandas as pd
    meta = metadatos or {}
    port = pd.to_numeric(pd.Series([mediciones.get("Destination Port")]), errors="coerce").iloc[0]
    servicio = {22:"ssh", 21:"ftp", 80:"http", 443:"https", 53:"dns", 445:"smb"}.get(port, "red")
    alerta = {
        "id_alerta": f"FLOW-{int(indice):09d}", "tipo_evento": "network_flow",
        "telemetria_red": mediciones_json(mediciones),
        "archivo_captura": str(archivo), "fila_captura_cero_based": int(indice),
        "servicio": servicio,
        "timestamp": meta.get("timestamp"), "ip_origen": meta.get("ip_origen"),
        "ip_destino": meta.get("ip_destino"), "hostname": meta.get("hostname"),
        "usuario": meta.get("usuario"),
    }
    if meta.get("severidad"):
        alerta["severidad_observada"] = meta["severidad"]
    return alerta


def mediciones_json(mediciones):
    import math
    import numbers
    resultado = {}
    for campo in VARIABLES_RED:
        valor = mediciones.get(campo)
        if isinstance(valor, numbers.Real):
            valor = float(valor) if math.isfinite(float(valor)) else None
        elif valor is not None and not isinstance(valor, str):
            valor = None
        resultado[campo] = valor
    return resultado


def incidente_flujo(alerta, conteo_por_ip):
    incidente = dict(alerta)
    incidente.update({
        "telemetria_red": alerta.get("telemetria_red", {c:alerta.get(c) for c in VARIABLES_RED}),
        "tipo_detector": "network_flow_native", "eventos_relacionados": [alerta.get("id_alerta")],
        "usuarios_atacados": [], "login_exitoso_posterior": False,
        "repeticiones_ip_origen": int(alerta.get("conteo_mismo_origen_lote") or
                                     (conteo_por_ip.get(alerta.get("ip_origen"), 0) if alerta.get("ip_origen") else 0)),
        "evidencia": ["Flujo de red recibido con mediciones de telemetría.",
                       f"Procedencia: {alerta.get('archivo_captura')}; fila {alerta.get('fila_captura_cero_based')}.",
                       "La clase ML no demuestra por sí sola una técnica MITRE ni fallos de autenticación."],
    })
    return incidente
