"""Entrena, persiste y valida el RF de red seleccionado dentro de Q-SOC."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src import comparar_modelos as comun
from src import comparar_representaciones as rep
from src.completar_comparacion_red import sha_indices
from src.entrada_red import ejecutar_tabla, huella, guardar_json
from src.ml_predictor_red import cargar_modelo_red, predecir_tabla_red
from src.telemetria_red import VARIABLES_RED, CLASES_RED, FORMATO_MODELO_RED, RUTA_MODELO_RED, matriz_red


def ejecutar(args):
    referencia = args.experimento.resolve()/"resultados_tres_modelos_red.json"
    datos = json.loads(referencia.read_text(encoding="utf-8-sig"))
    p = datos["protocolo"]
    inicial = json.loads(args.referencia_inicial.read_text(encoding="utf-8-sig"))
    if not args.verificacion_tecnica:
        if p.get("estado") != "completado" or p.get("modo") != "comparacion_tres_modelos_red":
            raise ValueError("Se requiere la comparación de tres modelos completada.")
        if p["scikit_learn"] != sklearn.__version__:
            raise ValueError("Mantén la versión de scikit-learn utilizada durante la comparación.")
    if huella(args.referencia_inicial) != p["referencia_inicial_sha256"]:
        raise ValueError("La referencia inicial no coincide con el experimento seleccionado.")
    if huella(args.dataset) != inicial["protocolo"]["dataset"]["sha256"]:
        raise ValueError("Cambió el dataset homologado que identifica la cohorte.")
    if p["features_originales"] != VARIABLES_RED or rep.VARIABLES_RED != VARIABLES_RED:
        raise ValueError("La selección de mediciones cambió.")
    for entrada in p["cohorte"]["archivos"]:
        nombre = entrada["archivo"]
        if Path(nombre).name != nombre or huella(args.data_dir/nombre) != entrada["sha256"]:
            raise ValueError(f"La captura original cambió: {nombre}")
    elegido = datos["resultados"]["random_forest_alternativo_red"]
    parametros = dict(elegido["parametros_efectivos"])
    perfil = {"n_estimators":100, "max_depth":12, "min_samples_leaf":5,
              "min_samples_split":15, "class_weight":"balanced", "random_state":42}
    if any(parametros.get(k) != v for k,v in perfil.items()):
        raise ValueError("La configuración no corresponde al Random Forest seleccionado.")
    Xhom, yhom, _ = comun.cargar_datos(args.dataset)
    Xhom[comun.OBJETIVO] = yhom
    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    preparacion = args.salida.resolve()/(fecha+"_preparacion")
    preparacion.mkdir(parents=True, exist_ok=False)
    X, y, cohorte = rep.reconstruir_cohorte(args.data_dir, Xhom, preparacion)
    del Xhom, yhom
    train, test = train_test_split(np.arange(len(X)), test_size=p["particion"]["test_size"],
                                  stratify=y, random_state=p["particion"]["semilla"])
    for nombre, indices in [("train",train), ("test",test)]:
        if sha_indices(indices) != p["particion"]["sha256_indices_"+nombre]:
            raise ValueError("Cambió la partición de entrenamiento/prueba.")
    valores, validas = matriz_red(X)
    if not validas.all():
        raise ValueError("Las mediciones no cumplen el contrato del predictor de red.")
    ruta_modelo = args.modelo.resolve()
    if ruta_modelo.exists():
        _, _, bundle = cargar_modelo_red(ruta_modelo)
        if bundle.get("referencia_sha256") != huella(referencia):
            raise FileExistsError("Ya existe otro modelo de red. Usa --modelo con una ruta nueva para conservarlo.")
        print("Reutilizando el RF de red persistido para esta misma referencia...", flush=True)
    else:
        parametros["n_jobs"] = args.hilos
        modelo = RandomForestClassifier(**parametros)
        codigos = {c:i for i,c in enumerate(CLASES_RED)}
        print("Entrenando y guardando el RF seleccionado con las doce variables originales...", flush=True)
        modelo.fit(valores[train], y.iloc[train].map(codigos).to_numpy(dtype=int))
        bundle = {"formato": FORMATO_MODELO_RED, "features": VARIABLES_RED, "clases": CLASES_RED,
                  "modelo": modelo, "referencia_sha256": huella(referencia),
                  "parametros": modelo.get_params(deep=False), "fecha_utc": fecha,
                  "scikit_learn": sklearn.__version__, "particion": p["particion"],
                  "cohorte": cohorte, "evaluacion_referencia": elegido["prueba"],
                  "modo": "verificacion_tecnica" if args.verificacion_tecnica else "RF_red_seleccionado",
                  "alcance": "Modelo entrenado solo con train; evaluación reproducida en el mismo test. "
                             "No es una validación externa ni un modelo de incidentes sin telemetría."}
        # Primero se verifica el artefacto serializado en una ubicación temporal.
        ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ruta_modelo.parent) as tmp:
            candidato = Path(tmp)/"candidato.joblib"
            joblib.dump(bundle, candidato, compress=3)
            _verificar_inferencia(candidato, X.iloc[test], test, y.iloc[test], elegido, args.experimento)
            if ruta_modelo.exists():
                raise FileExistsError("La ruta del modelo se ocupó durante el entrenamiento.")
            # 'xb' impide sobrescribir un modelo existente incluso si hay otra ejecución.
            with candidato.open("rb") as fuente, ruta_modelo.open("xb") as destino:
                import shutil
                shutil.copyfileobj(fuente, destino)
    pred = _verificar_inferencia(ruta_modelo, X.iloc[test], test, y.iloc[test], elegido, args.experimento)
    pred.to_csv(preparacion/"comprobacion_predictor.csv.gz", index=False, compression="gzip")
    procedencia = pd.read_csv(preparacion/"procedencia_muestras.csv.gz").set_index("fila_csv_homologado_cero_based").loc[test]
    salida = ejecutar_tabla(X.iloc[test].reset_index(drop=True), archivo="cohorte de prueba CICIDS2017",
                            salida=args.salida.resolve()/fecha, ruta_modelo=ruta_modelo,
                            procedencia=procedencia, max_reportes=args.max_reportes,
                            validacion={"clases_reales": y.iloc[test].to_numpy(),
                                        "matriz_esperada": elegido["prueba"]["matriz_confusion"]["valores"],
                                        "referencia_sha256":huella(referencia)})
    guardar_json(ruta_modelo.with_suffix(".json"), {k:v for k,v in bundle.items() if k!="modelo"})
    print(f"Modelo de red disponible en: {ruta_modelo}", flush=True)
    print("Validación completada sobre el test histórico; el dashboard anterior conserva sus fuentes.")
    return salida


def _verificar_inferencia(ruta_modelo, Xtest, test, ytest, elegido, experimento):
    pred = predecir_tabla_red(Xtest, ruta_modelo)
    if not pred["estado"].eq("ok").all():
        raise RuntimeError("El predictor persistido rechazó mediciones de la cohorte verificada.")
    anteriores = pd.read_csv(Path(experimento)/"predicciones_tres_modelos.csv.gz",
                             usecols=["fila_csv_cero_based", "random_forest_alternativo_red"])
    if not np.array_equal(anteriores["fila_csv_cero_based"].to_numpy(), test):
        raise ValueError("Las predicciones de referencia tienen otros índices.")
    if not np.array_equal(anteriores["random_forest_alternativo_red"].to_numpy(), pred["clase_predicha"].to_numpy()):
        raise RuntimeError("El modelo persistido no reproduce las predicciones del RF seleccionado.")
    codigos = {c:i for i,c in enumerate(CLASES_RED)}
    metricas = comun.calcular_metricas(ytest.map(codigos).to_numpy(), pred["clase_predicha"].map(codigos).to_numpy())
    if metricas["matriz_confusion"]["valores"] != elegido["prueba"]["matriz_confusion"]["valores"]:
        raise RuntimeError("La matriz del predictor integrado difiere de la referencia.")
    return pred


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experimento", required=True, type=Path)
    p.add_argument("--referencia-inicial", type=Path, default=Path.home()/"Downloads/resultados_completos_qsoc.json")
    p.add_argument("--dataset", type=Path, default=comun.rf_original.RUTA_DATASET_KALI)
    p.add_argument("--data-dir", type=Path, default=comun.RAIZ/"data")
    p.add_argument("--modelo", type=Path, default=RUTA_MODELO_RED)
    p.add_argument("--salida", type=Path, default=comun.RAIZ/"outputs/integracion_red")
    p.add_argument("--max-reportes", type=int, default=90)
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--verificacion-tecnica", action="store_true")
    a = p.parse_args(argv)
    if a.hilos < 1 or a.max_reportes < 1: p.error("Los hilos y reportes deben ser positivos.")
    return ejecutar(a)


if __name__ == "__main__":
    main()
