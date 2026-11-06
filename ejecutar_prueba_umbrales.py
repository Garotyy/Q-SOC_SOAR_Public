"""Prueba de umbrales con validación separada y capturas reservadas de Q-SOC."""
from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src import comparar_modelos as comun
from src import comparar_representaciones as rep
from src.ml_predictor_red import cargar_modelo_red
from src.telemetria_red import VARIABLES_RED, matriz_red
from src.umbrales_experimentales import (
    CLASES, NORMAL, SEMILLA, FRACCION_GRUPOS_VALIDACION, UMBRALES,
    aplicar_politica, dividir_desarrollo, firmas_vectores, medir_decision,
    seleccionar_politica, sha_indices, sumar_metricas, transiciones,
    validar_probabilidades, predecir_estable, comprobar_serializacion,
)

RAIZ = Path(__file__).resolve().parent
VERSION = "umbrales_reservados_v1_1"
CODIGO_ORIGEN_RECUPERABLE = "049ee75fca61560d5cda56c03fea6ae2601d3e2740e6d52e085ed17b93001560"
NUCLEO_ORIGEN_RECUPERABLE = "0afa1293288f0b0c6c0ddc27146225ad43512341985e9560d9e83c00c2f9d289"
RESERVADAS = (
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)
BASE = {"regla": "argmax_tres_clases", "umbral": None}
CODIGOS = {c: i for i, c in enumerate(CLASES)}


def guardar(ruta, datos, exclusivo=False):
    modo = "x" if exclusivo else "w"
    with Path(ruta).open(modo, encoding="utf-8") as f:
        f.write(json.dumps(comun._json_seguro(datos), ensure_ascii=False, indent=2, allow_nan=False))


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))


def detectar_codificacion(ruta):
    """Reconoce bytes sin interpretar etiquetas ni métricas de la captura."""
    for nombre in ("utf-8-sig", "cp1252", "latin-1"):
        decoder = codecs.getincrementaldecoder(nombre)(errors="strict")
        try:
            with Path(ruta).open("rb") as f:
                for bloque in iter(lambda: f.read(1024 * 1024), b""):
                    decoder.decode(bloque)
                decoder.decode(b"", final=True)
            return nombre
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo identificar la codificación del CSV.")


def localizar_reservadas(carpeta, hashes_entrenamiento):
    disponibles = {p.name.casefold(): p for p in Path(carpeta).glob("*pcap_ISCX.csv")}
    rutas, info, vistos = [], [], set()
    for nombre in RESERVADAS:
        ruta = disponibles.get(nombre.casefold())
        if ruta is None:
            raise FileNotFoundError(f"Falta la captura reservada {nombre} en {carpeta}.")
        sha = comun._sha256(ruta)
        if sha in hashes_entrenamiento or sha in vistos:
            raise ValueError("Una captura reservada duplica bytes de desarrollo o de otra captura reservada.")
        vistos.add(sha)
        rutas.append(ruta)
        info.append({"archivo": ruta.name, "sha256": sha, "rol": "prueba_reservada_no_usada_para_seleccion"})
    return rutas, info


def resolver_inicial(esperado, argumento):
    if argumento is not None:
        opciones = [Path(argumento)]
    else:
        opciones = [Path.home() / "Downloads/resultados_completos_qsoc.json"]
        opciones += sorted((RAIZ / "outputs/comparacion_modelos").glob("*/resultados_completos.json"))
    for p in opciones:
        if p.is_file() and comun._sha256(p) == esperado:
            return p
    raise FileNotFoundError("No se encontró la referencia inicial correcta. Usa --referencia-inicial con resultados_completos_qsoc.json.")


def protecciones(raiz, modelo, dataset):
    rutas = [raiz / nombre for nombre in (
        "dashboard.py", "dashboard_red.py", "main.py", "src/qwen_agent.py",
        "src/random_forest.py", "src/ml_predictor.py", "src/ml_predictor_red.py",
        "src/procesar_kali_dataset.py", "data/modelo_random_forest.joblib",
        "data/metricas_random_forest.json", "data/reportes_generados.json",
        "data/modelo_random_forest_red.json",
    )] + [Path(modelo), Path(dataset)]
    return {str(p.resolve()): comun._sha256(p) if p.is_file() else None for p in rutas}


def verificar_protecciones(huellas):
    for nombre, antes in huellas.items():
        p = Path(nombre)
        despues = comun._sha256(p) if p.is_file() else None
        if antes != despues:
            raise RuntimeError(f"Cambió un archivo protegido durante la prueba: {p}")


def localizar_entrenamiento_previo(args, referencia_sha):
    explicito = getattr(args, "reutilizar_entrenamiento", None)
    ultimo = getattr(args, "retomar_ultimo", False)
    if explicito is None and not ultimo:
        return None
    if explicito is not None:
        candidatos = [Path(explicito).resolve()]
    else:
        candidatos = sorted((p.parent for p in args.salida.resolve().glob("*/modelo_experimental.joblib")), reverse=True)
    for origen in candidatos:
        if (not (origen / "protocolo.json").is_file() or not (origen / "politica_congelada.json").is_file()
                or not (origen / "modelo_experimental.joblib").is_file()):
            continue
        p = leer_json(origen / "protocolo.json")
        codigos = {k.replace("\\", "/"): v for k, v in p.get("codigo_sha256", {}).items()}
        expuestos = list(origen.glob("predicciones_*.csv.gz"))
        if (p.get("estado") == "iniciado" and p.get("version") == "umbrales_reservados_v1"
                and p.get("referencia_tres_modelos_sha256") == referencia_sha
                and not (origen / "resultados_umbrales.json").exists()
                and not (origen / "avance_capturas.json").exists() and not expuestos
                and codigos.get("ejecutar_prueba_umbrales.py") == CODIGO_ORIGEN_RECUPERABLE
                and codigos.get("src/umbrales_experimentales.py") == NUCLEO_ORIGEN_RECUPERABLE):
            return origen
    raise ValueError("No se encontró una ejecución v1 recuperable detenida antes de las capturas reservadas.")


def recuperar_modelo(origen, parametros, ajuste, validacion, info_reservadas):
    original = leer_json(origen / "protocolo.json")
    congelada = leer_json(origen / "politica_congelada.json")
    verificar_protecciones(original["archivos_protegidos_sha256"])
    if (congelada["sha256_indices_ajuste"] != sha_indices(ajuste)
            or congelada["sha256_indices_validacion"] != sha_indices(validacion)
            or congelada["capturas_reservadas"] != info_reservadas):
        raise ValueError("Las particiones o capturas no coinciden con el entrenamiento guardado.")
    ruta = origen / "modelo_experimental.joblib"
    bundle = joblib.load(ruta)
    if (bundle.get("formato") != "qsoc.experimento_umbral.v1"
            or bundle.get("features") != VARIABLES_RED or tuple(bundle.get("clases", [])) != CLASES
            or bundle.get("scikit_learn") != sklearn.__version__
            or bundle.get("politica_sha256") != comun._sha256(origen / "politica_congelada.json")):
        raise ValueError("El artefacto recuperado no cumple el contrato de esta prueba.")
    modelo = bundle["modelo"]
    if (modelo.get_params(deep=False) != parametros or list(modelo.classes_) != [0, 1, 2]
            or modelo.n_features_in_ != 12 or len(modelo.estimators_) != parametros["n_estimators"]):
        raise ValueError("Cambió la configuración del modelo guardado.")
    return modelo, {"directorio_origen": str(origen), "modelo_origen_sha256": comun._sha256(ruta),
                    "politica_origen_sha256": comun._sha256(origen / "politica_congelada.json"),
                    "validacion_origen": congelada["seleccion"],
                    "nota": "Se reutilizan los árboles; la selección se recalcula solo en la misma validación con suma secuencial. "
                            "No se habían evaluado capturas reservadas en esta ejecución de origen."}


def resumen_probabilidades(y, probabilidades, politica):
    p = validar_probabilidades(probabilidades)
    base = aplicar_politica(p, BASE)
    propuesta = aplicar_politica(p, politica)
    return {
        "clasificador_argmax": comun.calcular_metricas(np.asarray(y), p.argmax(axis=1)),
        "regla_base": medir_decision(y, base),
        "regla_propuesta": medir_decision(y, propuesta),
        "transiciones": transiciones(y, base, propuesta),
    }


def evaluar_captura(ruta, sha_esperado, modelo, modelo_actual, politica, firmas_desarrollo,
                    salida, politica_fijada):
    # No se leen filas ni etiquetas reservadas antes de persistir la política.
    if not politica_fijada.is_file():
        raise RuntimeError("La política debe estar fijada antes de abrir la captura de prueba.")
    if comun._sha256(ruta) != sha_esperado:
        raise ValueError(f"Cambió la captura reservada {ruta.name}.")
    codificacion = detectar_codificacion(ruta)
    partes, n_raw, n_limpio = [], 0, 0
    descartados = {}
    for bloque in pd.read_csv(ruta, skipinitialspace=True, encoding=codificacion, chunksize=100000):
        n_raw += len(bloque)
        faltan = set(VARIABLES_RED + ["Label"]) - set(bloque.columns)
        if faltan:
            raise ValueError(f"Faltan columnas en {ruta.name}: {sorted(faltan)}")
        limpio = bloque.replace([np.inf, -np.inf], np.nan).dropna()
        fuera = bloque.loc[~bloque.index.isin(limpio.index), "Label"].fillna("SIN_ETIQUETA").astype(str)
        for etiqueta, n in fuera.value_counts().items():
            descartados[etiqueta] = descartados.get(etiqueta, 0) + int(n)
        if limpio.empty:
            continue
        entradas = limpio[VARIABLES_RED].apply(pd.to_numeric, errors="raise")
        valores, validos = matriz_red(entradas)
        if not validos.all():
            raise ValueError(f"{ruta.name}: {int((~validos).sum())} mediciones incumplen el contrato. No se ocultaron esos casos del cálculo.")
        # Las predicciones reciben únicamente las doce mediciones, nunca Label.
        pc = predecir_estable(modelo, valores)
        pa = validar_probabilidades(modelo_actual.predict_proba(valores))
        etiquetas = limpio["Label"].astype(str).str.strip()
        if etiquetas.eq("").any():
            raise ValueError("Hay etiquetas vacías después de la limpieza.")
        upper = etiquetas.str.upper()
        clase = np.full(len(limpio), 2, dtype=np.int64)
        clase[upper.eq("BENIGN").to_numpy()] = NORMAL
        clase[(~upper.eq("BENIGN") & (upper.str.contains("PATATOR") | upper.str.contains("PORT"))).to_numpy()] = 0
        rbase = aplicar_politica(pc, BASE)
        rpropuesta = aplicar_politica(pc, politica)
        ractual = aplicar_politica(pa, BASE)
        coincidentes = np.isin(firmas_vectores(valores), firmas_desarrollo)
        partes.append(pd.DataFrame({
            "fila_captura_cero_based": limpio.index.to_numpy(dtype=np.int64),
            "etiqueta_original": etiquetas.to_numpy(), "clase_real_codigo": clase,
            "clase_real": np.asarray(CLASES)[clase],
            "clase_rf_experimental": np.asarray(CLASES)[pc.argmax(axis=1)],
            "clase_rf_integrado": np.asarray(CLASES)[pa.argmax(axis=1)],
            "puntuacion_alerta_rf_experimental": pc[:, 0] + pc[:, 2],
            "revision_base_experimental": rbase,
            "revision_propuesta_experimental": rpropuesta,
            "revision_rf_integrado_referencia": ractual,
            "vector_presente_en_ajuste_o_validacion": coincidentes,
        }))
        n_limpio += len(limpio)
    if not partes:
        raise ValueError(f"No hay flujos válidos en {ruta.name}.")
    tabla = pd.concat(partes, ignore_index=True)
    y = tabla["clase_real_codigo"].to_numpy()
    b = tabla["revision_base_experimental"].to_numpy(dtype=bool)
    c = tabla["revision_propuesta_experimental"].to_numpy(dtype=bool)
    a = tabla["revision_rf_integrado_referencia"].to_numpy(dtype=bool)
    por_etiqueta = []
    for etiqueta, grupo in tabla.groupby("etiqueta_original", sort=True):
        yg = grupo["clase_real_codigo"].to_numpy()
        por_etiqueta.append({
            "etiqueta_original": str(etiqueta), "n": len(grupo),
            "clase_qsoc": str(grupo["clase_real"].iloc[0]),
            "base_experimental": medir_decision(yg, grupo["revision_base_experimental"].to_numpy(dtype=bool)),
            "propuesta_experimental": medir_decision(yg, grupo["revision_propuesta_experimental"].to_numpy(dtype=bool)),
            "rf_integrado_referencia": medir_decision(yg, grupo["revision_rf_integrado_referencia"].to_numpy(dtype=bool)),
        })
    sin_coincidencia = ~tabla["vector_presente_en_ajuste_o_validacion"].to_numpy(dtype=bool)
    diagnostico = {
        "n_vectores_presentes_en_desarrollo": int((~sin_coincidencia).sum()),
        "n_vectores_ausentes_en_desarrollo": int(sin_coincidencia.sum()),
        "alcance": "Igualdad de hashes de 64 bits de las 12 mediciones float32. No prueba identidad de eventos ni independencia de sesiones.",
    }
    if sin_coincidencia.any():
        diagnostico["sin_coincidencia_exacta"] = {
            "base_experimental": medir_decision(y[sin_coincidencia], b[sin_coincidencia]),
            "propuesta_experimental": medir_decision(y[sin_coincidencia], c[sin_coincidencia]),
            "rf_integrado_referencia": medir_decision(y[sin_coincidencia], a[sin_coincidencia]),
        }
    nombre = "predicciones_" + ruta.name.removesuffix(".csv") + ".csv.gz"
    tabla.to_csv(salida / nombre, index=False, compression="gzip")
    resultado = {
        "archivo": ruta.name, "sha256": sha_esperado, "codificacion": codificacion,
        "registros_originales": n_raw, "registros_evaluados": n_limpio,
        "descartados_limpieza_todas_las_columnas": n_raw - n_limpio,
        "descartados_por_etiqueta": descartados,
        "base_experimental": medir_decision(y, b),
        "propuesta_experimental": medir_decision(y, c),
        "rf_integrado_referencia": medir_decision(y, a),
        "transiciones_experimentales": transiciones(y, b, c),
        "por_etiqueta_original": por_etiqueta,
        "solapamiento_mediciones": diagnostico,
        "archivo_predicciones": nombre,
    }
    if comun._sha256(ruta) != sha_esperado:
        raise ValueError("La captura cambió durante su evaluación.")
    print(f"  {ruta.name}: FP {resultado['base_experimental']['falsos_positivos']} -> {resultado['propuesta_experimental']['falsos_positivos']}; "
          f"FN {resultado['base_experimental']['falsos_negativos']} -> {resultado['propuesta_experimental']['falsos_negativos']}", flush=True)
    return resultado


def ejecutar(args):
    referencia = args.experimento.resolve() / "resultados_tres_modelos_red.json"
    anterior = leer_json(referencia)
    p = anterior["protocolo"]
    if p.get("estado") != "completado" or p.get("modo") != "comparacion_tres_modelos_red":
        raise ValueError("Se requiere la comparación de tres modelos completada.")
    if p["scikit_learn"] != sklearn.__version__:
        raise ValueError("Conserva la versión de scikit-learn de la comparación previa.")
    if p["features_originales"] != VARIABLES_RED or tuple(comun.CLASES) != CLASES:
        raise ValueError("Cambió el contrato de mediciones o clases.")
    inicial_ruta = resolver_inicial(p["referencia_inicial_sha256"], args.referencia_inicial)
    inicial = leer_json(inicial_ruta)
    if comun._sha256(args.dataset) != inicial["protocolo"]["dataset"]["sha256"]:
        raise ValueError("El CSV homologado ya no corresponde a la cohorte histórica.")
    for captura in p["cohorte"]["archivos"]:
        nombre = captura["archivo"]
        if Path(nombre).name != nombre or comun._sha256(args.data_dir / nombre) != captura["sha256"]:
            raise ValueError(f"Cambió una captura original de desarrollo: {nombre}")
    rutas_reservadas, info_reservadas = localizar_reservadas(
        args.capturas_reservadas, {c["sha256"] for c in p["cohorte"]["archivos"]})
    integrado, huella_integrado, bundle = cargar_modelo_red(args.modelo_actual)
    if bundle.get("referencia_sha256") != comun._sha256(referencia):
        raise ValueError("El RF integrado no corresponde a esta comparación.")
    parametros = dict(anterior["resultados"]["random_forest_alternativo_red"]["parametros_efectivos"])
    perfil = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5,
              "min_samples_split": 15, "class_weight": "balanced", "random_state": SEMILLA}
    if any(parametros.get(k) != v for k, v in perfil.items()):
        raise ValueError("La configuración no es la RF seleccionada anteriormente.")
    parametros["n_jobs"] = args.hilos
    huellas = protecciones(RAIZ, args.modelo_actual, args.dataset)
    origen = localizar_entrenamiento_previo(args, comun._sha256(referencia))
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida = args.salida.resolve() / marca
    salida.mkdir(parents=True, exist_ok=False)
    protocolo = {
        "version": VERSION, "estado": "iniciado", "fecha_utc": marca,
        "modo": "umbral_validacion_y_capturas_reservadas",
        "referencia_tres_modelos_sha256": comun._sha256(referencia),
        "referencia_inicial_sha256": comun._sha256(inicial_ruta),
        "modelo_integrado_sha256": huella_integrado,
        "codigo_sha256": {str(q.relative_to(RAIZ)): comun._sha256(q) for q in (
            Path(__file__), RAIZ / "src/umbrales_experimentales.py",
            RAIZ / "src/comparar_modelos.py", RAIZ / "src/comparar_representaciones.py")},
        "entorno": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__, "hilos": args.hilos},
        "parametros_rf_experimental": parametros,
        "inferencia_rf_experimental": "Suma secuencial de árboles, n_jobs=1 solo durante predict_proba; ajuste con los hilos indicados.",
        "grilla_umbrales_prefijada": UMBRALES,
        "capturas_reservadas_prefijadas": info_reservadas,
        "archivos_protegidos_sha256": huellas,
        "limites": [
            "Capturas posteriores del mismo CICIDS2017; no son datos de otro SOC ni garantizan independencia de sesiones.",
            "El test histórico de 416.633 flujos ya se examinó; su nueva evaluación es descriptiva, no independiente.",
            "La configuración y las variables provienen de la comparación exploratoria previa.",
            "El RF experimental se entrena con menos registros para reservar validación. No comparar sus cambios con los 1.799 FP como efecto aislado del umbral.",
            "El efecto del umbral se compara entre dos reglas aplicadas al mismo RF experimental.",
            "El RF integrado se evalúa como referencia separada; fue ajustado con otra cantidad de registros.",
            "Las políticas son sobre flujos. No miden tiempos humanos ni el flujo completo con reglas contextuales de Q-SOC.",
            "No se usa Qwen y no se integra automáticamente ningún modelo o política.",
        ],
    }
    guardar(salida / "protocolo.json", protocolo)
    Xhom, yhom, _ = comun.cargar_datos(args.dataset)
    Xhom[comun.OBJETIVO] = yhom
    X, ystr, cohorte = rep.reconstruir_cohorte(args.data_dir, Xhom, salida)
    del Xhom, yhom
    if {c["archivo"]: c["sha256"] for c in cohorte["archivos"]} != {c["archivo"]: c["sha256"] for c in p["cohorte"]["archivos"]}:
        raise ValueError("Cambió la cohorte durante la reconstrucción.")
    y = ystr.map(CODIGOS).to_numpy(dtype=np.int64)
    pool, historico = train_test_split(np.arange(len(X)), test_size=p["particion"]["test_size"],
                                       stratify=y, random_state=p["particion"]["semilla"])
    for nombre, indices in (("train", pool), ("test", historico)):
        if sha_indices(indices) != p["particion"]["sha256_indices_" + nombre]:
            raise ValueError("La partición histórica no se reproduce.")
    valores, validos = matriz_red(X)
    if not validos.all():
        raise ValueError("Las mediciones reconstruidas no cumplen el contrato nativo.")
    del X
    firmas = firmas_vectores(valores)
    ajuste, validacion = dividir_desarrollo(pool, y, firmas)
    def detalle(indices):
        return {"n": len(indices), "sha256_indices": sha_indices(indices),
                "clases": {c: int(np.sum(y[indices] == i)) for i, c in enumerate(CLASES)}}
    protocolo["particiones"] = {"ajuste": detalle(ajuste), "validacion": detalle(validacion),
                                "test_historico_explorado": detalle(historico),
                                "fraccion_grupos_validacion": FRACCION_GRUPOS_VALIDACION,
                                "metodo": "GroupShuffleSplit, un split, seed 42, grupos por hash de 12 mediciones float32",
                                "grupos_compartidos_ajuste_validacion": 0,
                                "advertencia": "20% de grupos no equivale necesariamente a 20% de filas; no es separación por sesiones."}
    pd.DataFrame({"fila_csv_homologado_cero_based": np.concatenate([ajuste, validacion, historico]),
                  "rol": np.repeat(["ajuste", "validacion", "test_historico_explorado"], [len(ajuste), len(validacion), len(historico)])}).to_csv(
                      salida / "particiones.csv.gz", index=False, compression="gzip")
    print(f"RF experimental: ajuste {len(ajuste):,}; validación {len(validacion):,}; test histórico {len(historico):,}.", flush=True)
    with threadpool_limits(limits=args.hilos):
        if origen is not None:
            print("Reutilizando el entrenamiento experimental guardado en:", origen, flush=True)
            modelo, recuperacion = recuperar_modelo(origen, parametros, ajuste, validacion, info_reservadas)
            protocolo["entrenamiento_reutilizado"] = recuperacion
            tiempo_ajuste = None
        else:
            modelo = RandomForestClassifier(**parametros)
            inicio = perf_counter()
            modelo.fit(valores[ajuste], y[ajuste])
            tiempo_ajuste = perf_counter() - inicio
        if list(modelo.classes_) != [0, 1, 2]:
            raise RuntimeError("El modelo experimental no conserva las tres clases.")
        pvalidacion = predecir_estable(modelo, valores[validacion])
        seleccion, curva = seleccionar_politica(y[validacion], pvalidacion)
        if origen is not None:
            vieja = recuperacion["validacion_origen"]
            conteos = ("n", "verdaderos_positivos", "falsos_negativos", "falsos_positivos", "verdaderos_negativos")
            for clave in ("base_validacion", "propuesta_validacion"):
                if any(vieja[clave][k] != seleccion[clave][k] for k in conteos):
                    raise RuntimeError("El modelo recuperado no reproduce los conteos de validación previos. Se detuvo antes de las capturas reservadas.")
            if vieja["politica"] != seleccion["politica"]:
                raise RuntimeError("La selección de validación cambió al comprobar el entrenamiento recuperado. No se abrieron las capturas reservadas.")
            recuperacion["conteos_validacion_y_politica_reproducidos"] = True
    pd.DataFrame(curva).to_csv(salida / "curva_umbrales_validacion.csv", index=False, encoding="utf-8-sig")
    politica_ruta = salida / "politica_congelada.json"
    politica = seleccion["politica"]
    congelada = {
        "version": VERSION, "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "seleccion": seleccion, "parametros_rf": parametros,
        "sha256_indices_ajuste": sha_indices(ajuste), "sha256_indices_validacion": sha_indices(validacion),
        "capturas_reservadas": info_reservadas,
        "nota": "Escrita antes de leer filas y etiquetas de las capturas reservadas y antes de evaluar el test histórico.",
        "inferencia": "Suma secuencial de los árboles para evitar diferencias por orden de acumulación en paralelo.",
    }
    guardar(politica_ruta, congelada, exclusivo=True)
    sha_politica = comun._sha256(politica_ruta)
    bundle_experimental = {
        "formato": "qsoc.experimento_umbral.v1", "modelo": modelo,
        "features": VARIABLES_RED, "clases": CLASES, "politica": politica,
        "politica_sha256": sha_politica, "scikit_learn": sklearn.__version__,
        "nota": "Artefacto experimental; incompatible deliberadamente con la activación automática del predictor productivo.",
    }
    ruta_candidato = salida / "modelo_experimental.joblib"
    np.savez_compressed(salida / "puntuaciones_validacion.npz", filas=validacion,
                        clases_reales=y[validacion], puntuaciones=pvalidacion)
    joblib.dump(bundle_experimental, ruta_candidato, compress=3)
    recargado = joblib.load(ruta_candidato)
    control = comprobar_serializacion(pvalidacion, predecir_estable(recargado["modelo"], valores[validacion]))
    protocolo["comprobacion_serializacion"] = control
    guardar(salida / "comprobacion_serializacion.json", control)
    modelo = recargado["modelo"]
    print("Política fijada con validación:", json.dumps(politica), flush=True)
    if not seleccion["mejora_en_validacion"]:
        print("Ningún umbral cumplió el criterio; se conserva argmax como propuesta.", flush=True)
    with threadpool_limits(limits=args.hilos):
        ph = predecir_estable(modelo, valores[historico])
    evaluacion_historica = resumen_probabilidades(y[historico], ph, politica)
    evaluacion_historica["alcance"] = "Test histórico ya explorado; no selecciona el umbral ni constituye validación independiente."
    pactual_historico = validar_probabilidades(integrado.predict_proba(valores[historico]))
    matriz_actual = comun.calcular_metricas(y[historico], pactual_historico.argmax(axis=1))["matriz_confusion"]["valores"]
    matriz_referencia = anterior["resultados"]["random_forest_alternativo_red"]["prueba"]["matriz_confusion"]["valores"]
    if matriz_actual != matriz_referencia:
        raise RuntimeError("El RF integrado no reproduce su matriz histórica; se detuvo antes de abrir las capturas reservadas.")
    evaluacion_historica["rf_integrado_referencia"] = medir_decision(y[historico], aplicar_politica(pactual_historico, BASE))
    firmas_desarrollo = np.unique(firmas[np.concatenate([ajuste, validacion])])
    resultados_capturas = []
    for ruta, info in zip(rutas_reservadas, info_reservadas):
        with threadpool_limits(limits=args.hilos):
            r = evaluar_captura(ruta, info["sha256"], modelo, integrado, politica,
                               firmas_desarrollo, salida, politica_ruta)
        resultados_capturas.append(r)
        if comun._sha256(politica_ruta) != sha_politica:
            raise RuntimeError("La política cambió después de consultar las capturas reservadas.")
        guardar(salida / "avance_capturas.json", resultados_capturas)
    agregado = {k: sumar_metricas([r[k] for r in resultados_capturas]) for k in (
        "base_experimental", "propuesta_experimental", "rf_integrado_referencia")}
    cambios = {k: sum(r["transiciones_experimentales"][k] for r in resultados_capturas)
               for k in resultados_capturas[0]["transiciones_experimentales"]}
    agregado["transiciones_experimentales"] = cambios
    agregado["mejora_observada_sin_nuevas_omisiones"] = bool(
        agregado["propuesta_experimental"]["falsos_positivos"] < agregado["base_experimental"]["falsos_positivos"]
        and cambios["ataques_antes_detectados_ahora_omitidos"] == 0)
    filas = []
    for r in resultados_capturas:
        for nombre in ("base_experimental", "propuesta_experimental", "rf_integrado_referencia"):
            filas.append({"captura": r["archivo"], "regla": nombre, **r[nombre]})
    pd.DataFrame(filas).to_csv(salida / "comparacion_capturas.csv", index=False, encoding="utf-8-sig")
    verificar_protecciones(huellas)
    protocolo.update({"estado": "completado", "tiempo_ajuste_rf_segundos": tiempo_ajuste,
                      "politica_congelada_sha256": sha_politica,
                      "modelo_experimental_sha256": comun._sha256(ruta_candidato),
                      "archivos_protegidos_conservados": True})
    resultado = {"protocolo": protocolo, "validacion": seleccion,
                 "test_historico_explorado": evaluacion_historica,
                 "evaluacion_capturas_reservadas": {"agregado": agregado, "por_captura": resultados_capturas},
                 "seleccion_o_integracion_automatica": None}
    guardar(salida / "resultados_umbrales.json", resultado)
    guardar(salida / "protocolo.json", protocolo)
    print("\nResultados guardados en:", salida, flush=True)
    print("Modelo integrado, métricas, reportes y ambos paneles conservados. No se consultó Qwen.", flush=True)
    return salida


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", type=Path, required=True)
    parser.add_argument("--referencia-inicial", type=Path)
    parser.add_argument("--dataset", type=Path, default=RAIZ / "data/dataset_kali_homologado.csv")
    parser.add_argument("--data-dir", type=Path, default=RAIZ / "data")
    parser.add_argument("--capturas-reservadas", type=Path, default=Path.home() / "Downloads/MachineLearningCSV/MachineLearningCVE")
    parser.add_argument("--modelo-actual", type=Path, default=RAIZ / "data/modelo_random_forest_red.joblib")
    parser.add_argument("--salida", type=Path, default=RAIZ / "outputs/prueba_umbrales")
    parser.add_argument("--hilos", type=int, default=4)
    recuperar = parser.add_mutually_exclusive_group()
    recuperar.add_argument("--retomar-ultimo", action="store_true")
    recuperar.add_argument("--reutilizar-entrenamiento", type=Path)
    args = parser.parse_args(argv)
    if args.hilos < 1:
        parser.error("Los hilos deben ser positivos.")
    return ejecutar(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}") from error
