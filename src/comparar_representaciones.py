"""Prueba controlada de variables originales frente a homologadas en Q-SOC.

Reconstruye la cohorte del CSV homologado a partir de las capturas originales,
verifica su identidad y compara RF alternativo y XGBoost con 12 variables de
red. No sustituye el modelo que procesa incidentes ni modifica reportes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from src import comparar_modelos as comun
from src import random_forest as rf


VARIABLES_RED = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Mean",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "SYN Flag Count", "ACK Flag Count",
]
ARCHIVOS = {
    "monday-workinghours.pcap_iscx.csv",
    "tuesday-workinghours.pcap_iscx.csv",
    "wednesday-workinghours.pcap_iscx.csv",
}


def homologar_para_verificar(df: pd.DataFrame) -> pd.DataFrame:
    """Replica solo las entradas y etiquetas del homologador recibido.

    Se usa para comprobar identidad, nunca como entrada de los modelos de red.
    """
    port = pd.to_numeric(df["Destination Port"], errors="raise")
    servicio = pd.Series("otros", index=df.index)
    for puertos, nombre in [([22], "ssh"), ([80, 443], "http"),
                            ([21], "ftp"), ([139, 445], "smb")]:
        servicio.loc[port.isin(puertos)] = nombre
    label = df["Label"].astype(str).str.strip().str.upper()
    y = pd.Series("sospechoso", index=df.index)
    y.loc[label.eq("BENIGN")] = "normal"
    y.loc[~label.eq("BENIGN") & (label.str.contains("PATATOR") | label.str.contains("PORT"))] = "fallido"
    duracion = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0).clip(lower=0)
    columnas_bwd = [c for c in df.columns if ("bwd" in c.lower() or "backward" in c.lower())
                    and "packet" in c.lower() and "/" in c.lower()]
    bwd = (pd.to_numeric(df[columnas_bwd[0]], errors="coerce").fillna(0).clip(lower=0)
           if columnas_bwd else pd.Series(0, index=df.index))
    log = np.log1p(duracion)
    log = log.where(np.isfinite(log), 0).astype(np.int64)
    bwd = bwd.where(np.isfinite(bwd), 0).clip(upper=100).astype(np.int64)
    return pd.DataFrame({
        "usuario": "network_system", "servicio": servicio,
        "hora_evento": 12, "es_horario_laboral": 1, "ip_origen_es_privada": 1,
        "usuario_privilegiado": 0, "conteo_eventos_ip_ventana": log,
        "conteo_usuarios_distintos_ip": 1, "conteo_hosts_distintos_ip": 1,
        "conteo_fallidos_ventana": bwd, "host_critico": port.isin([22, 21, 445]).astype(int),
        "servicio_sensible": servicio.isin(["ssh", "ftp", "smb"]).astype(int),
        comun.OBJETIVO: y,
    }, index=df.index)


def firmas_comparables(df: pd.DataFrame) -> np.ndarray:
    # El dtype explícito evita diferencias de hash entre inferencia int/float
    # o string de pandas al leer archivos distintos en una misma ejecución.
    canon = pd.DataFrame(index=df.index)
    for c in comun.FEATURES + [comun.OBJETIVO]:
        canon[c] = (pd.to_numeric(df[c], errors="raise").astype(np.float64)
                    if c in rf.COLUMNAS_NUMERICAS else df[c].astype(str).astype(object))
    return pd.util.hash_pandas_object(canon, index=False).to_numpy()


def reconstruir_cohorte(data_dir: Path, referencia: pd.DataFrame,
                       salida: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    rutas = [p for p in data_dir.glob("*pcap_ISCX.csv") if p.name.casefold() in ARCHIVOS]
    faltantes = ARCHIVOS - {p.name.casefold() for p in rutas}
    if faltantes:
        raise FileNotFoundError("Faltan capturas originales: " + ", ".join(sorted(faltantes)))
    esperado = firmas_comparables(referencia)
    partes, detalle = {}, []
    for ruta in rutas:
        matrices, etiquetas, firmas, filas = [], [], [], []
        n_original = n_limpio = 0
        print(f"Leyendo y verificando {ruta.name}...", flush=True)
        for bloque in pd.read_csv(ruta, skipinitialspace=True, chunksize=100000):
            n_original += len(bloque)
            # Misma limpieza de cohorte que el homologador: todas las columnas,
            # no solo las 12 seleccionadas para el modelo.
            limpio = bloque.replace([np.inf, -np.inf], np.nan).dropna()
            if limpio.empty:
                continue
            ausentes = set(VARIABLES_RED + ["Label"]) - set(limpio.columns)
            if ausentes:
                raise ValueError(f"Faltan columnas originales en {ruta.name}: {sorted(ausentes)}")
            homologado = homologar_para_verificar(limpio)
            red = limpio[VARIABLES_RED].apply(pd.to_numeric, errors="raise")
            # Conserva el tratamiento no negativo de la duración usado antes.
            red["Flow Duration"] = red["Flow Duration"].clip(lower=0)
            z = red.to_numpy(dtype=np.float64)
            if not np.isfinite(z).all():
                raise ValueError(f"Hay entradas no finitas en {ruta.name}")
            matrices.append(z)
            etiquetas.append(homologado[comun.OBJETIVO].to_numpy())
            firmas.append(firmas_comparables(homologado))
            filas.append(limpio.index.to_numpy(dtype=np.int64))
            n_limpio += len(limpio)
        if not matrices:
            raise ValueError(f"No quedaron registros válidos en {ruta.name}")
        partes[ruta.name] = {"X": np.concatenate(matrices), "y": np.concatenate(etiquetas),
                             "firma": np.concatenate(firmas), "fila": np.concatenate(filas)}
        detalle.append({"archivo": ruta.name, "sha256": comun._sha256(ruta),
                        "registros_iniciales": n_original, "registros_validos": n_limpio})
    n_reconstruido = sum(len(p["y"]) for p in partes.values())
    if n_reconstruido != len(referencia):
        comun._guardar_json(salida / "cohorte_no_coincide.json", {
            "referencia": len(referencia), "reconstruida": n_reconstruido, "archivos": detalle})
        raise ValueError("La cohorte reconstruida tiene otro tamaño. Se detuvo la prueba sin entrenar.")
    ordenes = []
    for orden in itertools.permutations(partes):
        observado = np.concatenate([partes[n]["firma"] for n in orden])
        if np.array_equal(esperado, observado):
            ordenes.append(orden)
    if not ordenes:
        comun._guardar_json(salida / "cohorte_no_coincide.json", {
            "total": len(referencia), "ordenes_ensayados": 6, "archivos": detalle})
        raise ValueError("Las entradas/etiquetas reconstruidas no coinciden fila por fila con el "
                         "CSV homologado. Se detuvo la prueba para evitar una comparación inválida.")
    orden = ordenes[0]
    X = pd.DataFrame(np.concatenate([partes[n]["X"] for n in orden]), columns=VARIABLES_RED)
    y = pd.Series(np.concatenate([partes[n]["y"] for n in orden]), name=comun.OBJETIVO)
    metadatos = pd.DataFrame({
        "fila_csv_homologado_cero_based": np.arange(len(X)),
        "archivo_captura": np.concatenate([np.repeat(n, len(partes[n]["y"])) for n in orden]),
        "fila_captura_cero_based": np.concatenate([partes[n]["fila"] for n in orden]),
        "clase": y,
    })
    metadatos.to_csv(salida / "procedencia_muestras.csv.gz", index=False, compression="gzip")
    return X, y, {"verificada_fila_por_fila": True, "metodo": "hash de 64 bits de 12 entradas y etiqueta",
                  "orden_capturas": orden, "ordenes_coincidentes": len(ordenes), "archivos": detalle}


def ejecutar(args: argparse.Namespace) -> Path:
    referencia = json.loads(args.referencia.read_text(encoding="utf-8-sig"))
    previo = referencia["protocolo"]
    if previo.get("estado") != "completado" or previo.get("modo") != "experimento_dataset_homologado":
        if not args.verificacion_tecnica:
            raise ValueError("La referencia no corresponde a un experimento real completado.")
    if comun._sha256(args.dataset) != previo["dataset"]["sha256"]:
        raise ValueError("Cambió el CSV homologado respecto de la comparación anterior.")
    # Asegura que la referencia histórica y los modelos nuevos representan
    # las mismas fábricas de estimadores, no una versión posterior del código.
    for nombre in ["src/random_forest.py", "src/modelo_arbol_decision.py",
                   "src/procesar_kali_dataset.py", "src/ml_predictor.py"]:
        esperado = previo.get("codigo_sha256", {}).get(nombre)
        if esperado and comun._sha256(comun.RAIZ / nombre) != esperado:
            raise ValueError(f"Cambió {nombre} respecto del experimento de referencia.")
    Xhom, yhom, _ = comun.cargar_datos(args.dataset)
    hom = Xhom.copy()
    hom[comun.OBJETIVO] = yhom
    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida = args.salida.resolve() / fecha
    salida.mkdir(parents=True, exist_ok=False)
    X, y, cohorte = reconstruir_cohorte(args.data_dir, hom, salida)
    del Xhom, hom
    train, test = train_test_split(np.arange(len(X)), test_size=0.25,
                                  stratify=y, random_state=rf.RANDOM_STATE)
    import hashlib
    for nombre, indices in [("train", train), ("test", test)]:
        real = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
        if real != previo["particion"][f"sha256_indices_{nombre}"]:
            raise ValueError("La partición no coincide con la comparación anterior.")
    encoder = LabelEncoder().fit(list(comun.CLASES))
    yt, yv = encoder.transform(y.iloc[train]), encoder.transform(y.iloc[test])
    Ztrain = X.iloc[train].to_numpy(dtype=np.float32)
    Ztest = X.iloc[test].to_numpy(dtype=np.float32)
    if not np.isfinite(Ztrain).all() or not np.isfinite(Ztest).all():
        raise ValueError("La conversión a float32 produjo valores fuera de rango.")
    modelos_todos = comun.construir_modelos(min(args.hilos, os.cpu_count() or 1), True)
    nombres = ["random_forest_alternativo", "xgboost"]
    protocolo = {
        "estado": "iniciado", "modo": "verificacion_tecnica" if args.verificacion_tecnica else "comparacion_representaciones",
        "fecha_utc": fecha,
        "referencia_sha256": comun._sha256(args.referencia), "cohorte": cohorte,
        "dataset_homologado_sha256": previo["dataset"]["sha256"],
        "codigo_experimento_sha256": comun._sha256(Path(__file__)),
        "particion": previo["particion"], "entorno_referencia": previo["entorno"],
        "features_originales": VARIABLES_RED, "n_train": len(train), "n_test": len(test),
        "parametros": {n: modelos_todos[n].get_params(deep=False) for n in nombres},
        "alcance": "Misma cohorte y partición; cambia la representación de entrada. "
                   "Los resultados sobre flujos no validan el adaptador de incidentes. "
                   "Prueba exploratoria, sin selección de parámetros ni umbrales sobre test.",
    }
    comun._guardar_json(salida / "protocolo_representacion.json", protocolo)
    resultados = {}
    predicciones = pd.DataFrame({"fila_csv_cero_based": test, "clase_real": encoder.inverse_transform(yv)})
    pesos = compute_sample_weight("balanced", yt)
    for nombre in nombres:
        modelo = modelos_todos[nombre]
        anterior = referencia["resultados"][nombre]
        # Se conservan las versiones: un cambio de biblioteca puede confundir
        # la atribución de diferencias a la representación.
        import xgboost
        import sklearn
        if not args.verificacion_tecnica:
            for biblioteca, version in [("scikit_learn", sklearn.__version__), ("xgboost", xgboost.__version__)]:
                if previo["entorno"][biblioteca] != version:
                    raise ValueError(f"Cambió la versión de {biblioteca}; mantén el entorno de referencia.")
        print(f"Entrenando {nombre} con variables de red...", flush=True)
        with threadpool_limits(limits=args.hilos):
            inicio = perf_counter()
            if nombre == "xgboost":
                modelo.fit(Ztrain, yt, sample_weight=pesos, verbose=False)
            else:
                modelo.fit(Ztrain, yt)
            ajuste = perf_counter() - inicio
            pred_train = np.asarray(modelo.predict(Ztrain), dtype=int)
            modelo.predict(Ztest[:min(1024, len(test))])
            inicio = perf_counter()
            pred = np.asarray(modelo.predict(Ztest), dtype=int)
            tiempo = perf_counter() - inicio
        clave = nombre + "_red"
        resultados[nombre + "_homologado"] = anterior
        resultados[clave] = {
            "parametros_efectivos": modelo.get_params(deep=False),
            "entrenamiento": comun.calcular_metricas(yt, pred_train),
            "prueba": comun.calcular_metricas(yv, pred),
            "tiempo_ajuste_s": ajuste, "tiempo_prediccion_lote_prueba_s": tiempo,
        }
        predicciones[clave] = encoder.inverse_transform(pred)
        m = resultados[clave]["prueba"]
        f = m["filtrado_binario"]
        print(f"  F1 macro={m['f1_macro']:.4f}; FP={f['falsos_positivos']}; FN={f['falsos_negativos']}", flush=True)
    predicciones.to_csv(salida / "predicciones_variables_red.csv.gz", index=False, compression="gzip")
    filas = []
    for nombre, r in resultados.items():
        m = r["prueba"]
        filas.append({"configuracion": nombre, "f1_macro_train": r["entrenamiento"]["f1_macro"],
                      "f1_macro_test": m["f1_macro"], "accuracy_test": m["accuracy"],
                      **m["filtrado_binario"], "tiempo_ajuste_s": r["tiempo_ajuste_s"]})
    pd.DataFrame(filas).to_csv(salida / "comparacion_representaciones.csv", index=False, encoding="utf-8-sig")
    protocolo["estado"] = "completado"
    if comun._sha256(args.dataset) != previo["dataset"]["sha256"]:
        raise ValueError("El CSV homologado cambió durante la ejecución.")
    protocolo["columnas_constantes_originales_train"] = [c for c in VARIABLES_RED if X[c].iloc[train].nunique() == 1]
    comun._guardar_json(salida / "resultados_representacion.json", {"protocolo": protocolo, "resultados": resultados})
    comun._guardar_json(salida / "protocolo_representacion.json", protocolo)
    print(f"\nResultados guardados en: {salida}", flush=True)
    print("Prueba sobre flujos terminada. No se modificó el modelo productivo ni se ejecutó Qwen.")
    return salida


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--referencia", type=Path, default=Path.home()/"Downloads/resultados_completos_qsoc.json")
    p.add_argument("--dataset", type=Path, default=rf.RUTA_DATASET_KALI)
    p.add_argument("--data-dir", type=Path, default=comun.RAIZ/"data")
    p.add_argument("--salida", type=Path, default=comun.RAIZ/"outputs/comparacion_representaciones")
    p.add_argument("--hilos", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--verificacion-tecnica", action="store_true")
    args = p.parse_args(argv)
    if args.hilos < 1:
        p.error("--hilos debe ser positivo")
    return ejecutar(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
