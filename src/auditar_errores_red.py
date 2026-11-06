"""Audita predicciones ya guardadas; no entrena ni cambia el modelo de Q-SOC."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import comparar_modelos as comun
from src.comparar_representaciones import VARIABLES_RED

MODELOS = ("random_forest_alternativo_red", "xgboost_red")
ID = "fila_csv_homologado_cero_based"
ARCHIVO = "archivo_captura"
FILA_RAW = "fila_captura_cero_based"


def categorizar(real: pd.Series, pred: pd.Series) -> np.ndarray:
    resultado = np.full(len(real), "acierto", dtype=object)
    resultado[(real == "normal") & (pred != "normal")] = "falso_positivo"
    resultado[(real != "normal") & (pred == "normal")] = "falso_negativo"
    resultado[(real != "normal") & (pred != "normal") & (real != pred)] = "confusion_entre_ataques"
    return resultado


def validar_predicciones(pred: pd.DataFrame, protocolo: dict, resultados: dict) -> None:
    necesarias = {"fila_csv_cero_based", "clase_real", *MODELOS}
    if not necesarias.issubset(pred.columns) or pred.isna().any().any():
        raise ValueError("Las predicciones están incompletas.")
    filas = pred["fila_csv_cero_based"].to_numpy(dtype=np.int64)
    if len(filas) != protocolo["n_test"] or len(np.unique(filas)) != len(filas):
        raise ValueError("Tamaño incorrecto o índices de prueba repetidos.")
    if hashlib.sha256(filas.astype("<i8").tobytes()).hexdigest() != protocolo["particion"]["sha256_indices_test"]:
        raise ValueError("Los índices de prueba no coinciden con el experimento.")
    codigos = {c: i for i, c in enumerate(comun.CLASES)}
    for columna in ["clase_real", *MODELOS]:
        if not pred[columna].isin(codigos).all():
            raise ValueError(f"Clase desconocida en {columna}.")
    real = pred["clase_real"].map(codigos).to_numpy()
    for nombre in MODELOS:
        medidas = comun.calcular_metricas(real, pred[nombre].map(codigos).to_numpy())
        previo = resultados[nombre]["prueba"]
        if medidas["matriz_confusion"]["valores"] != previo["matriz_confusion"]["valores"]:
            raise ValueError(f"Las predicciones de {nombre} no reproducen su matriz de confusión.")


def cargar_variables(data_dir: Path, procedencia: pd.DataFrame, filas_test: pd.DataFrame,
                    archivos: list[dict], variables: list[str]) -> tuple[pd.DataFrame, dict]:
    """Extrae test y comprueba solapamiento de vectores con train por bloques.

    Usa hashes de las entradas convertidas a float32 como en el experimento.
    Un vector repetido no demuestra por sí mismo que sea el mismo evento.
    """
    piezas, firmas_train, firmas_test = [], [], []
    for entrada in archivos:
        nombre = entrada["archivo"]
        if Path(nombre).name != nombre:
            raise ValueError("Nombre de captura inválido en el protocolo.")
        ruta = data_dir / nombre
        if comun._sha256(ruta) != entrada["sha256"]:
            raise ValueError(f"Cambió la captura original {nombre}.")
        validas = procedencia.loc[procedencia[ARCHIVO] == nombre, FILA_RAW].to_numpy()
        prueba = filas_test.loc[filas_test[ARCHIVO] == nombre, FILA_RAW].to_numpy()
        print(f"Auditando {nombre}...", flush=True)
        n_vistas = 0
        for bloque in pd.read_csv(ruta, skipinitialspace=True, usecols=variables+["Label"], chunksize=100000):
            seleccionado = bloque.loc[bloque.index.isin(validas)].copy()
            if seleccionado.empty:
                continue
            n_vistas += len(seleccionado)
            red = seleccionado[variables].apply(pd.to_numeric, errors="raise")
            red["Flow Duration"] = red["Flow Duration"].clip(lower=0)
            valores = red.to_numpy(dtype=np.float32)
            if not np.isfinite(valores).all():
                raise ValueError(f"Valores no finitos en {nombre}.")
            firmas = pd.util.hash_pandas_object(pd.DataFrame(valores, columns=variables), index=False).to_numpy()
            es_test = seleccionado.index.isin(prueba)
            firmas_train.append(firmas[~es_test])
            firmas_test.append(firmas[es_test])
            parte = red.loc[es_test].copy()
            parte[ARCHIVO] = nombre
            parte[FILA_RAW] = parte.index.to_numpy(dtype=np.int64)
            parte["etiqueta_original"] = seleccionado.loc[es_test, "Label"].astype(str).str.strip()
            piezas.append(parte.reset_index(drop=True))
        if n_vistas != len(validas):
            raise ValueError(f"La procedencia no coincide con las filas de {nombre}.")
    red_test = pd.concat(piezas, ignore_index=True)
    if red_test.duplicated([ARCHIVO, FILA_RAW]).any() or len(red_test) != len(filas_test):
        raise ValueError("La reconstrucción de filas de prueba no es unívoca.")
    train, test = np.concatenate(firmas_train), np.concatenate(firmas_test)
    conocidos = np.isin(test, np.unique(train))
    diagnostico = {
        "registros_train": len(train), "registros_test": len(test),
        "vectores_distintos_train": len(np.unique(train)), "vectores_distintos_test": len(np.unique(test)),
        "registros_test_con_vector_presente_en_train": int(conocidos.sum()),
        "proporcion_test_con_vector_presente_en_train": float(conocidos.mean()),
        "metodo": "Hash de 64 bits de las 12 entradas float32, sin etiqueta.",
        "interpretacion": "Coincidir en las variables no demuestra identidad de eventos ni fuga por sí solo. "
                          "Es un diagnóstico de similitud entre los subconjuntos; no prueba generalización temporal.",
    }
    return red_test, diagnostico


def tabla_grupos(df: pd.DataFrame, columnas: list[str], error: str) -> list[dict]:
    x = df[columnas].copy()
    x["n_test"] = 1
    for categoria in ["falso_positivo", "falso_negativo", "confusion_entre_ataques"]:
        x[categoria] = (df[error] == categoria).astype(int)
    agrupado = x.groupby(columnas, dropna=False, observed=True).sum(numeric_only=True).reset_index()
    agrupado["errores_total"] = agrupado[["falso_positivo", "falso_negativo", "confusion_entre_ataques"]].sum(axis=1)
    agrupado["fraccion_error_grupo"] = agrupado["errores_total"] / agrupado["n_test"]
    return agrupado.sort_values(["errores_total", "n_test"], ascending=False).to_dict(orient="records")


def resumir(df: pd.DataFrame, variables: list[str]) -> dict:
    modelos = {}
    for nombre in MODELOS:
        error = "error_" + nombre
        distribuciones = []
        for (clase, tipo), grupo in df.groupby(["clase_real", error], observed=True):
            cuantiles = grupo[variables].quantile([0.25, 0.5, 0.75])
            distribuciones.append({
                "clase_real": clase, "resultado": tipo, "n": len(grupo),
                "variables": {c: {"p25": float(cuantiles.loc[.25, c]),
                                  "mediana": float(cuantiles.loc[.5, c]),
                                  "p75": float(cuantiles.loc[.75, c])} for c in variables},
            })
        modelos[nombre] = {
            "conteos": df[error].value_counts().to_dict(),
            "por_etiqueta_original": tabla_grupos(df, ["etiqueta_original"], error),
            "por_captura_y_etiqueta": tabla_grupos(df, [ARCHIVO, "etiqueta_original"], error),
            "puertos_con_mas_errores": tabla_grupos(df, ["Destination Port"], error)[:20],
            "distribuciones_descriptivas": distribuciones,
        }
    real = df["clase_real"] != "normal"
    rf, xgb = df[MODELOS[0]] != "normal", df[MODELOS[1]] != "normal"
    comparacion = {
        "ataques_omitidos_por_ambos": int((real & ~rf & ~xgb).sum()),
        "ataques_omitidos_solo_RF": int((real & ~rf & xgb).sum()),
        "ataques_omitidos_solo_XGBoost": int((real & rf & ~xgb).sum()),
        "falsos_positivos_de_ambos": int((~real & rf & xgb).sum()),
        "falsos_positivos_solo_RF": int((~real & rf & ~xgb).sum()),
        "falsos_positivos_solo_XGBoost": int((~real & ~rf & xgb).sum()),
    }
    return {"modelos": modelos, "solapamiento_errores": comparacion}


def ejecutar(args: argparse.Namespace) -> Path:
    experimento = args.experimento.resolve()
    ruta_json = experimento / "resultados_representacion.json"
    datos = json.loads(ruta_json.read_text(encoding="utf-8-sig"))
    protocolo = datos["protocolo"]
    if protocolo.get("estado") != "completado" or protocolo.get("modo") != "comparacion_representaciones":
        if not args.verificacion_tecnica:
            raise ValueError("Se necesita un experimento de representación real y completado.")
    if protocolo["features_originales"] != VARIABLES_RED:
        raise ValueError("Las variables no corresponden a esta versión del auditor.")
    ruta_pred = experimento / "predicciones_variables_red.csv.gz"
    ruta_proc = experimento / "procedencia_muestras.csv.gz"
    pred = pd.read_csv(ruta_pred)
    validar_predicciones(pred, protocolo, datos["resultados"])
    proc = pd.read_csv(ruta_proc)
    requeridas = {ID, ARCHIVO, FILA_RAW, "clase"}
    if not requeridas.issubset(proc.columns) or proc[list(requeridas)].isna().any().any():
        raise ValueError("El archivo de procedencia está incompleto.")
    if len(proc) != protocolo["n_train"]+protocolo["n_test"] or proc[ID].duplicated().any():
        raise ValueError("El tamaño o los índices de procedencia son incorrectos.")
    if not np.array_equal(np.sort(proc[ID].to_numpy()), np.arange(len(proc))):
        raise ValueError("La procedencia no contiene todos los índices de la cohorte.")
    if proc.duplicated([ARCHIVO, FILA_RAW]).any():
        raise ValueError("Hay posiciones de capturas repetidas en la procedencia.")
    test = pred.rename(columns={"fila_csv_cero_based": ID}).merge(
        proc, on=ID, how="left", validate="one_to_one", sort=False)
    if test["clase"].isna().any() or not test["clase_real"].equals(test["clase"]):
        raise ValueError("Las clases de predicciones y procedencia no coinciden.")
    red, solapamiento = cargar_variables(args.data_dir, proc, test, protocolo["cohorte"]["archivos"], VARIABLES_RED)
    df = test.merge(red, on=[ARCHIVO, FILA_RAW], how="left", validate="one_to_one", sort=False)
    etiqueta = df["etiqueta_original"].str.upper()
    clase_original = np.where(etiqueta == "BENIGN", "normal",
                              np.where(etiqueta.str.contains("PATATOR|PORT"), "fallido", "sospechoso"))
    if not np.array_equal(clase_original, df["clase_real"].to_numpy()) or df[VARIABLES_RED].isna().any().any():
        raise ValueError("La etiqueta original o las variables no coinciden con el experimento.")
    if solapamiento["registros_train"] != protocolo["n_train"] or solapamiento["registros_test"] != protocolo["n_test"]:
        raise ValueError("El solapamiento no se calculó sobre la misma cohorte.")
    for nombre in MODELOS:
        df["error_"+nombre] = categorizar(df["clase_real"], df[nombre])
    analisis = resumir(df, VARIABLES_RED)
    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida = args.salida.resolve()/fecha
    salida.mkdir(parents=True, exist_ok=False)
    errores = df.loc[(df["error_"+MODELOS[0]] != "acierto") | (df["error_"+MODELOS[1]] != "acierto")]
    errores.to_csv(salida/"casos_mal_clasificados.csv.gz", index=False, compression="gzip")
    comun._guardar_json(salida/"auditoria_errores_red.json", {
        "protocolo": {"fecha_utc": fecha, "modo": "verificacion_tecnica" if args.verificacion_tecnica else "auditoria_descriptiva",
                      "experimento": str(experimento), "resultados_sha256": comun._sha256(ruta_json),
                      "predicciones_sha256": comun._sha256(ruta_pred), "procedencia_sha256": comun._sha256(ruta_proc),
                      "auditor_sha256": comun._sha256(Path(__file__)), "n_test": len(df)},
        "solapamiento_vectores_train_test": solapamiento, **analisis,
        "limites": "Auditoría descriptiva de una prueba ya examinada. Las asociaciones no demuestran causas. "
                   "No se ajustan umbrales ni se elige un ensamble usando test. No incluye probabilidades "
                   "porque no se guardaron en el experimento original; no permite evaluar confianza o calibración. "
                   "Las clases agrupadas no sustituyen la etiqueta original de cada ataque.",
    })
    for nombre, r in analisis["modelos"].items():
        print(nombre + ": " + json.dumps(r["conteos"], ensure_ascii=False), flush=True)
    print(f"Resultados de auditoría guardados en: {salida}", flush=True)
    print("Se analizaron predicciones guardadas. No se entrenaron modelos ni se consultó Qwen.")
    return salida


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experimento", required=True, type=Path)
    p.add_argument("--data-dir", type=Path, default=comun.RAIZ/"data")
    p.add_argument("--salida", type=Path, default=comun.RAIZ/"outputs/auditoria_errores_red")
    p.add_argument("--verificacion-tecnica", action="store_true")
    return ejecutar(p.parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"ERROR: {e}") from e
