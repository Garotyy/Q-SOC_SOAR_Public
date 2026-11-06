"""Entrada CSV y validación RF -> política de revisión -> reportes Q-SOC."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_predictor_red import predecir_tabla_red, cargar_modelo_red
from src.telemetria_red import VARIABLES_RED, CLASES_RED, RUTA_MODELO_RED, crear_alerta_flujo
from src.politica_red import normalizar

RAIZ = Path(__file__).resolve().parents[1]
ALIASES = {"ip_origen": ["ip_origen", "Source IP"], "ip_destino": ["ip_destino", "Destination IP"],
           "timestamp": ["timestamp", "Timestamp"], "hostname": ["hostname"], "usuario": ["usuario"],
           "severidad": ["severidad_observada", "severidad"]}


def guardar_json(ruta, datos):
    Path(ruta).write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def huella(ruta):
    h = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        for bloque in iter(lambda:f.read(1024*1024), b""): h.update(bloque)
    return h.hexdigest()


def extraer_metadatos(datos):
    meta = pd.DataFrame(index=range(len(datos)))
    for campo, aliases in ALIASES.items():
        fuente = next((c for c in aliases if c in datos.columns), None)
        valores = datos[fuente].reset_index(drop=True) if fuente else pd.Series([None]*len(datos))
        meta[campo] = valores.where(valores.notna(), None).map(
            lambda v: None if v is None or str(v).strip().lower() in {"", "nan", "n/a", "none"} else str(v))
    ip = meta["ip_origen"]
    conteos = ip.value_counts(dropna=True)
    meta["conteo_mismo_origen_lote"] = ip.map(conteos).fillna(0).astype(int)
    return meta


def politica_lote(pred, meta):
    ml = pred["estado"].ne("ok") | pred["clase_predicha"].ne("normal")
    severas = meta["severidad"].map(normalizar).isin(["alta", "critica", "high", "critical"])
    repetidas = meta["conteo_mismo_origen_lote"].ge(5)
    return (ml | severas | repetidas).to_numpy(dtype=bool)


def ejecutar_tabla(datos, *, archivo, salida, ruta_modelo=RUTA_MODELO_RED,
                   procedencia=None, max_reportes=90, validacion=None):
    """Clasifica todo el lote; el informe detallado conserva una muestra explícita.

    Label se ignora. Las etiquetas reales solo pueden recibirse como evaluación
    externa a la inferencia mediante el parámetro validacion.
    """
    salida = Path(salida)
    if salida.exists():
        raise ValueError("La carpeta de salida ya existe; utiliza una nueva para conservar las evidencias.")
    if len(datos) == 0 or max_reportes < 1:
        raise ValueError("Se requiere un lote no vacío y al menos un reporte de muestra.")
    datos = datos.reset_index(drop=True)
    pred = predecir_tabla_red(datos, ruta_modelo)
    meta = extraer_metadatos(datos)
    revisar = politica_lote(pred, meta)
    pred["revision_sugerida"] = revisar
    pred["fila_lote_cero_based"] = np.arange(len(pred))
    if procedencia is None:
        fuentes = pd.DataFrame({"archivo_captura": str(archivo), "fila_captura_cero_based": np.arange(len(pred))})
    else:
        fuentes = procedencia[["archivo_captura", "fila_captura_cero_based"]].reset_index(drop=True)
        if len(fuentes) != len(datos) or fuentes.isna().any().any():
            raise ValueError("La procedencia no coincide con los flujos de entrada.")
    # Estratos según PREDICCIÓN, nunca según la etiqueta real ni el acierto.
    elegidas = []
    cuota = max(1, max_reportes//4)
    grupos = pred["clase_predicha"].where(pred["estado"].eq("ok"), "error")
    for clase in ["normal", "fallido", "sospechoso", "error"]:
        elegidas.extend(pred.index[grupos.eq(clase)][:cuota].tolist())
    conjunto_elegidas = set(elegidas)
    restantes = [i for i in pred.index if i not in conjunto_elegidas]
    elegidas = (elegidas+restantes[:max(0,max_reportes-len(elegidas))])[:max_reportes]
    alertas = []
    for i in elegidas:
        alerta = crear_alerta_flujo(datos.iloc[i].to_dict(), i, fuentes.iloc[i]["archivo_captura"], meta.iloc[i].to_dict())
        alerta["fila_captura_cero_based"] = int(fuentes.iloc[i]["fila_captura_cero_based"])
        alerta["conteo_mismo_origen_lote"] = int(meta.iloc[i]["conteo_mismo_origen_lote"])
        alertas.append(alerta)
    salida.mkdir(parents=True, exist_ok=False)
    from main import ejecutar_pipeline
    reportes = ejecutar_pipeline(imprimir=False, alertas=alertas, ruta_reportes=salida/"reportes_red.json",
                                  usar_qwen=False, ruta_modelo_red=ruta_modelo)
    por_id = {r["id_alerta"]:r for r in reportes}
    for i in elegidas:
        r = por_id[f"FLOW-{i:09d}"]
        if r["prediccion_ml"].get("estado") != pred.loc[i,"estado"]:
            raise RuntimeError("El predictor integrado difiere del predictor por lotes.")
        if pred.loc[i,"estado"] == "ok" and r["prediccion_ml"]["clase_predicha"] != pred.loc[i,"clase_predicha"]:
            raise RuntimeError("La clase cambió al ingresar al flujo Q-SOC.")
        if bool(r["requiere_escalamiento"]) != bool(revisar[i]):
            raise RuntimeError("La política del lote difiere de los reportes integrados.")
    resumen = {
        "estado": "completado", "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "modo": "validacion_integracion" if validacion is not None else "clasificacion_csv",
        "n_flujos": len(pred), "n_validos": int(pred["estado"].eq("ok").sum()),
        "n_errores_entrada": int(pred["estado"].ne("ok").sum()),
        "n_revision_sugerida": int(revisar.sum()), "n_filtrados": int((~revisar).sum()),
        "clases_predichas": {str(k):int(v) for k,v in pred.loc[pred["estado"].eq("ok"),"clase_predicha"].value_counts().items()},
        "n_reportes_muestra": len(reportes), "reportes_integrados_verificados": True,
        "llm_consultado": False, "modelo": str(Path(ruta_modelo).resolve()), "modelo_sha256": pred.attrs["version_modelo"],
        "archivo_origen": str(archivo), "unidad": "flujos; no incidentes SOC consolidados",
        "alcance": "Clasificación de todo el lote y comprobación de política/reportes sobre una muestra explícita. "
                   "Las recomendaciones no ejecutan contención. No mide tiempos ni carga de trabajo humana.",
        "identidad_ip_disponible": bool(meta["ip_origen"].notna().any()),
    }
    if validacion is not None:
        from src.comparar_modelos import calcular_metricas, CLASES
        if not pred["estado"].eq("ok").all():
            raise ValueError("Hay errores de entrada en la reproducción de la prueba.")
        codigos = {c:i for i,c in enumerate(CLASES)}
        reales = np.asarray(validacion["clases_reales"])
        if len(reales) != len(pred) or not np.isin(reales, CLASES).all():
            raise ValueError("Las etiquetas de evaluación no coinciden con el lote.")
        m = calcular_metricas(np.array([codigos[c] for c in reales]), pred["clase_predicha"].map(codigos).to_numpy())
        if m["matriz_confusion"]["valores"] != validacion["matriz_esperada"]:
            raise RuntimeError("El modelo persistido no reproduce la matriz de referencia.")
        resumen["evaluacion_clasificador"] = m
        resumen["referencia_evaluacion_sha256"] = validacion["referencia_sha256"]
        pred["clase_real_evaluacion"] = reales
    pd.concat([fuentes, pred], axis=1).to_csv(salida/"predicciones_red.csv.gz", index=False, compression="gzip")
    pd.concat([fuentes.loc[revisar].reset_index(drop=True), pred.loc[revisar].reset_index(drop=True)], axis=1).to_csv(
        salida/"cola_revision_red.csv.gz", index=False, compression="gzip")
    _, _, bundle = cargar_modelo_red(ruta_modelo)
    guardar_json(salida/"ficha_modelo.json", {k:v for k,v in bundle.items() if k!="modelo"})
    guardar_json(salida/"resumen_integracion_red.json", resumen)
    print(f"Flujos: {len(pred)}; revisión sugerida: {int(revisar.sum())}; filtrados: {int((~revisar).sum())}")
    print(f"Reportes de muestra verificados: {len(reportes)}; análisis avanzado bajo demanda.")
    print(f"Resultados guardados en: {salida.resolve()}")
    return salida


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--modelo", type=Path, default=RUTA_MODELO_RED)
    p.add_argument("--salida", type=Path, default=None)
    p.add_argument("--max-reportes", type=int, default=90)
    a = p.parse_args(argv)
    encabezado = pd.read_csv(a.csv, skipinitialspace=True, nrows=0).columns
    faltan = set(VARIABLES_RED)-set(encabezado)
    if faltan: p.error("Faltan mediciones: "+", ".join(sorted(faltan)))
    # No se carga Label ni otros campos no utilizados por la inferencia.
    columnas = VARIABLES_RED+[c for aliases in ALIASES.values() for c in aliases if c in encabezado]
    datos = pd.read_csv(a.csv, skipinitialspace=True, usecols=list(dict.fromkeys(columnas)))
    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return ejecutar_tabla(datos, archivo=a.csv.name, salida=a.salida or RAIZ/"outputs/integracion_red"/fecha,
                          ruta_modelo=a.modelo, max_reportes=a.max_reportes)


if __name__ == "__main__":
    main()
