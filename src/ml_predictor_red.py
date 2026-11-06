"""Inferencia del RF seleccionado, con validación estricta del contrato."""
from functools import lru_cache
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.telemetria_red import (VARIABLES_RED, CLASES_RED, FORMATO_MODELO_RED,
                               RUTA_MODELO_RED, extraer_flujos, matriz_red)


@lru_cache(maxsize=2)
def _leer_modelo(ruta, mtime_ns, size):
    bundle = joblib.load(ruta)
    if not isinstance(bundle, dict) or bundle.get("formato") != FORMATO_MODELO_RED:
        raise ValueError("El archivo no contiene un modelo RF de red compatible.")
    if bundle.get("features") != VARIABLES_RED or bundle.get("clases") != CLASES_RED:
        raise ValueError("El esquema de mediciones o clases del modelo no coincide.")
    modelo = bundle["modelo"]
    if getattr(modelo, "n_features_in_", None) != 12 or list(modelo.classes_) != [0, 1, 2]:
        raise ValueError("El estimador guardado no corresponde a las doce entradas y tres clases.")
    h = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        for bloque in iter(lambda:f.read(1024*1024), b""): h.update(bloque)
    return modelo, h.hexdigest(), bundle


def cargar_modelo_red(ruta=RUTA_MODELO_RED):
    ruta = Path(ruta).resolve()
    estado = ruta.stat()
    return _leer_modelo(str(ruta), estado.st_mtime_ns, estado.st_size)


def predecir_tabla_red(datos, ruta_modelo=RUTA_MODELO_RED):
    modelo, version, _ = cargar_modelo_red(ruta_modelo)
    valores, validas = matriz_red(datos)
    n = len(valores)
    salida = pd.DataFrame({"estado": ["error"]*n, "clase_predicha": [""]*n,
                           "confianza": np.zeros(n),
                           "detalle": ["Mediciones ausentes, no numéricas, no finitas o puerto inválido."]*n})
    for clase in CLASES_RED: salida["p_"+clase] = np.nan
    if validas.any():
        probabilidades = modelo.predict_proba(valores[validas])
        pred = np.asarray(modelo.predict(valores[validas]), dtype=int)
        if (probabilidades.shape != (int(validas.sum()), 3) or not np.isfinite(probabilidades).all()
                or (probabilidades < 0).any() or (probabilidades > 1).any()
                or not np.allclose(probabilidades.sum(axis=1), 1) or not np.isin(pred, [0,1,2]).all()):
            raise ValueError("El modelo devolvió predicciones o probabilidades incompatibles.")
        salida.loc[validas, "estado"] = "ok"
        salida.loc[validas, "detalle"] = ""
        salida.loc[validas, "clase_predicha"] = np.asarray(CLASES_RED)[pred]
        salida.loc[validas, "confianza"] = probabilidades[np.arange(len(pred)), pred]
        for i, clase in enumerate(CLASES_RED): salida.loc[validas, "p_"+clase] = probabilidades[:, i]
    salida.attrs["version_modelo"] = version
    return salida


def predecir_incidente_red(incidente, ruta_modelo=RUTA_MODELO_RED):
    base = {"modelo": "RandomForestClassifier (12 variables originales de red)",
            "representacion": "red_original_12", "features_usadas": VARIABLES_RED}
    try:
        flujos = extraer_flujos(incidente)
        salida = predecir_tabla_red(flujos, ruta_modelo)
        if not salida["estado"].eq("ok").all():
            raise ValueError("Uno o más flujos carecen de mediciones válidas; requieren revisión.")
        prioridad = {"normal": 0, "fallido": 1, "sospechoso": 2}
        orden = salida["clase_predicha"].map(prioridad)
        elegida = salida.loc[orden.idxmax()]
        return {**base, "estado": "ok", "clase_predicha": elegida["clase_predicha"],
                "confianza": float(elegida["confianza"]),
                "probabilidades": {c:float(elegida["p_"+c]) for c in CLASES_RED},
                "version_modelo": salida.attrs["version_modelo"], "n_flujos": len(flujos),
                "metodo_agregacion": "Clase de mayor prioridad; probabilidades del flujo seleccionado."}
    except Exception as error:
        return {**base, "estado": "error", "detalle": str(error)}
