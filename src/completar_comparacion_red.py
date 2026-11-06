"""Completa la comparación con el árbol base sobre las 12 variables de red.

Conserva cohorte, partición y configuración; reutiliza predicciones guardadas
de RF y XGBoost. No cambia el modelo integrado ni consulta a Qwen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from src import comparar_modelos as comun
from src import comparar_representaciones as representacion

MODELOS_PREVIOS = ("random_forest_alternativo_red", "xgboost_red")


def sha_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()


def firmas_float32(X: pd.DataFrame) -> np.ndarray:
    partes = []
    for inicio in range(0, len(X), 100000):
        valores = X.iloc[inicio:inicio+100000].to_numpy(dtype=np.float32)
        if not np.isfinite(valores).all():
            raise ValueError("Las entradas float32 contienen valores no finitos.")
        partes.append(pd.util.hash_pandas_object(pd.DataFrame(valores, columns=X.columns), index=False).to_numpy())
    return np.concatenate(partes)


def evaluar_subconjuntos(real: np.ndarray, pred: np.ndarray, vistos: np.ndarray) -> dict:
    resultados = {}
    for nombre, mascara in [("vector_presente_en_train", vistos), ("vector_ausente_en_train", ~vistos)]:
        cantidad = int(mascara.sum())
        resultados[nombre] = {
            "n": cantidad,
            "soporte_por_clase": {c: int((real[mascara] == i).sum()) for i, c in enumerate(comun.CLASES)},
            "metricas": comun.calcular_metricas(real[mascara], pred[mascara]) if cantidad else None,
        }
    return resultados


def validar_predicciones_previas(guardadas: pd.DataFrame, test: np.ndarray,
                                reales: np.ndarray, previo: dict) -> dict[str, np.ndarray]:
    columnas = {"fila_csv_cero_based", "clase_real", *MODELOS_PREVIOS}
    if not columnas.issubset(guardadas.columns) or guardadas[list(columnas)].isna().any().any():
        raise ValueError("Las predicciones previas están incompletas.")
    if not np.array_equal(guardadas["fila_csv_cero_based"].to_numpy(), test):
        raise ValueError("El orden de las predicciones previas no coincide con los índices de prueba.")
    codigos = {c: i for i, c in enumerate(comun.CLASES)}
    if not guardadas["clase_real"].isin(codigos).all():
        raise ValueError("Hay etiquetas reales desconocidas en las predicciones previas.")
    if not np.array_equal(guardadas["clase_real"].map(codigos).to_numpy(), reales):
        raise ValueError("Las etiquetas de las predicciones previas no coinciden con la cohorte.")
    predicciones = {}
    for nombre in MODELOS_PREVIOS:
        if not guardadas[nombre].isin(codigos).all():
            raise ValueError(f"Hay predicciones desconocidas en {nombre}.")
        pred = guardadas[nombre].map(codigos).to_numpy(dtype=int)
        matriz = comun.calcular_metricas(reales, pred)["matriz_confusion"]["valores"]
        if matriz != previo["resultados"][nombre]["prueba"]["matriz_confusion"]["valores"]:
            raise ValueError(f"Las predicciones de {nombre} no reproducen su matriz guardada.")
        predicciones[nombre] = pred
    return predicciones


def ejecutar(args: argparse.Namespace) -> Path:
    ruta_previa = args.experimento.resolve()/"resultados_representacion.json"
    ruta_pred = args.experimento.resolve()/"predicciones_variables_red.csv.gz"
    previo = json.loads(ruta_previa.read_text(encoding="utf-8-sig"))
    primera = json.loads(args.referencia_inicial.read_text(encoding="utf-8-sig"))
    p = previo["protocolo"]
    if not args.verificacion_tecnica:
        if p.get("estado") != "completado" or p.get("modo") != "comparacion_representaciones":
            raise ValueError("Se requiere una comparación real de representaciones completada.")
        if primera["protocolo"].get("modo") != "experimento_dataset_homologado":
            raise ValueError("La referencia inicial no corresponde a la comparación homologada.")
        if p["entorno_referencia"]["scikit_learn"] != sklearn.__version__:
            raise ValueError("Cambió scikit-learn respecto del entorno de referencia.")
    if p["referencia_sha256"] != comun._sha256(args.referencia_inicial):
        raise ValueError("La referencia inicial no coincide con la utilizada por RF y XGBoost.")
    if p["features_originales"] != representacion.VARIABLES_RED:
        raise ValueError("Las variables originales no coinciden con esta entrega.")
    if p["dataset_homologado_sha256"] != comun._sha256(args.dataset):
        raise ValueError("Cambió el CSV homologado utilizado para verificar la cohorte.")
    for nombre, esperado in primera["protocolo"].get("codigo_sha256", {}).items():
        ruta = comun.RAIZ/nombre
        if ruta.is_file() and comun._sha256(ruta) != esperado:
            raise ValueError(f"Cambió {nombre} respecto de la comparación inicial.")
    if p.get("codigo_experimento_sha256") != comun._sha256(Path(representacion.__file__)):
        raise ValueError("Cambió el módulo que reconstruye las variables originales.")
    for entrada in p["cohorte"]["archivos"]:
        nombre = entrada["archivo"]
        if Path(nombre).name != nombre or comun._sha256(args.data_dir/nombre) != entrada["sha256"]:
            raise ValueError(f"Cambió la captura original {nombre}.")
    modelo = comun.construir_modelos(args.hilos, False)["arbol_decision"]
    parametros = modelo.get_params(deep=False)
    parametros_previos = primera["resultados"]["arbol_decision"]["parametros_efectivos"]
    if comun._json_seguro(parametros) != parametros_previos:
        raise ValueError("La configuración del árbol no coincide con su comparación inicial.")
    Xhom, yhom, _ = comun.cargar_datos(args.dataset)
    Xhom[comun.OBJETIVO] = yhom
    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida = args.salida.resolve()/fecha
    salida.mkdir(parents=True, exist_ok=False)
    X, y, cohorte = representacion.reconstruir_cohorte(args.data_dir, Xhom, salida)
    del Xhom, yhom
    train, test = train_test_split(np.arange(len(X)), test_size=p["particion"]["test_size"],
                                  stratify=y, random_state=p["particion"]["semilla"])
    for nombre, indices in [("train", train), ("test", test)]:
        if sha_indices(indices) != p["particion"]["sha256_indices_"+nombre]:
            raise ValueError("La partición no coincide con RF y XGBoost.")
    encoder = LabelEncoder().fit(list(comun.CLASES))
    yt, yv = encoder.transform(y.iloc[train]), encoder.transform(y.iloc[test])
    guardadas = pd.read_csv(ruta_pred)
    predicciones = validar_predicciones_previas(guardadas, test, yv, previo)
    firmas = firmas_float32(X)
    vistos = np.isin(firmas[test], np.unique(firmas[train]))
    Ztrain, Ztest = X.iloc[train].to_numpy(dtype=np.float32), X.iloc[test].to_numpy(dtype=np.float32)
    print("Entrenando árbol de decisión base con variables originales de red...", flush=True)
    with threadpool_limits(limits=args.hilos):
        inicio = perf_counter()
        modelo.fit(Ztrain, yt)
        tiempo_ajuste = perf_counter()-inicio
        pred_train = np.asarray(modelo.predict(Ztrain), dtype=int)
        modelo.predict(Ztest[:min(1024, len(test))])
        inicio = perf_counter()
        pred_arbol = np.asarray(modelo.predict(Ztest), dtype=int)
        tiempo_pred = perf_counter()-inicio
    resultados = {
        "arbol_decision_red": {
            "parametros_efectivos": parametros,
            "entrenamiento": comun.calcular_metricas(yt, pred_train),
            "prueba": comun.calcular_metricas(yv, pred_arbol),
            "tiempo_ajuste_s": tiempo_ajuste,
            "tiempo_prediccion_lote_prueba_s": tiempo_pred,
        },
        **{n: previo["resultados"][n] for n in MODELOS_PREVIOS},
    }
    predicciones = {"arbol_decision_red": pred_arbol, **predicciones}
    filas = []
    tabla_pred = pd.DataFrame({"fila_csv_cero_based": test, "clase_real": encoder.inverse_transform(yv),
                               "vector_presente_en_train": vistos})
    for nombre, r in resultados.items():
        r["subconjuntos_prueba"] = evaluar_subconjuntos(yv, predicciones[nombre], vistos)
        tabla_pred[nombre] = encoder.inverse_transform(predicciones[nombre])
        m = r["prueba"]
        filas.append({"modelo": nombre, "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
                      "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
                      **m["filtrado_binario"], "tiempo_ajuste_s": r["tiempo_ajuste_s"],
                      "tiempo_prediccion_lote_prueba_s": r["tiempo_prediccion_lote_prueba_s"]})
        f = m["filtrado_binario"]
        print(f"{nombre}: F1 macro={m['f1_macro']:.4f}; FP={f['falsos_positivos']}; FN={f['falsos_negativos']}", flush=True)
    protocolo = {
        "estado": "completado", "modo": "verificacion_tecnica" if args.verificacion_tecnica else "comparacion_tres_modelos_red",
        "fecha_utc": fecha, "cohorte": cohorte, "features_originales": representacion.VARIABLES_RED,
        "particion": p["particion"], "referencia_inicial_sha256": comun._sha256(args.referencia_inicial),
        "resultados_RF_XGBoost_sha256": comun._sha256(ruta_previa),
        "predicciones_RF_XGBoost_sha256": comun._sha256(ruta_pred),
        "codigo_sha256": comun._sha256(Path(__file__)), "scikit_learn": sklearn.__version__,
        "modelos_entrenados_en_esta_ejecucion": ["arbol_decision_red"],
        "modelos_reutilizados": list(MODELOS_PREVIOS),
        "n_test_vector_presente_en_train": int(vistos.sum()),
        "n_test_vector_ausente_en_train": int((~vistos).sum()),
        "limites": "Comparación exploratoria de configuraciones prefijadas sobre un test ya examinado. "
                   "Separar métricas según coincidencia de vectores no es una validación externa ni elimina "
                   "otras dependencias entre flujos. Los hashes de 64 bits usan las entradas float32 sin etiqueta; "
                   "una coincidencia no prueba identidad de eventos. El F1 macro usa las tres clases; "
                   "revisar sus soportes en cada subconjunto. Los tiempos provienen de una ejecución de cada "
                   "modelo; excluyen lectura, reconstrucción, Qwen y revisión humana.",
        "seleccion_automatica": None,
    }
    tabla_pred.to_csv(salida/"predicciones_tres_modelos.csv.gz", index=False, compression="gzip")
    pd.DataFrame(filas).to_csv(salida/"comparacion_tres_modelos.csv", index=False, encoding="utf-8-sig")
    comun._guardar_json(salida/"resultados_tres_modelos_red.json", {"protocolo": protocolo, "resultados": resultados})
    print(f"Resultados guardados en: {salida}", flush=True)
    print("Comparación completada. Se conservó el modelo integrado y no se consultó Qwen.")
    return salida


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experimento", type=Path, required=True)
    p.add_argument("--referencia-inicial", type=Path, default=Path.home()/"Downloads/resultados_completos_qsoc.json")
    p.add_argument("--dataset", type=Path, default=comun.rf_original.RUTA_DATASET_KALI)
    p.add_argument("--data-dir", type=Path, default=comun.RAIZ/"data")
    p.add_argument("--salida", type=Path, default=comun.RAIZ/"outputs/comparacion_final_red")
    p.add_argument("--hilos", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--verificacion-tecnica", action="store_true")
    args = p.parse_args(argv)
    if args.hilos < 1:
        p.error("--hilos debe ser positivo")
    return ejecutar(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError, ImportError) as e:
        raise SystemExit(f"ERROR: {e}") from e
