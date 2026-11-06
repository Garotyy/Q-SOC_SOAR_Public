"""Comparación experimental de clasificadores para Q-SOC.

Ejecutar desde la raíz: python -m src.comparar_modelos
Lee el CSV homologado existente. No ejecuta main, Qwen ni el entrenamiento
productivo, y escribe cada experimento en una carpeta nueva de outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from src import modelo_arbol_decision as arbol_original
from src import random_forest as rf_original


RAIZ = Path(__file__).resolve().parent.parent
VERSION = "1.0.0"
CLASES = tuple(sorted(rf_original.CLASES_OBJETIVO))
FEATURES = list(rf_original.COLUMNAS_CATEGORICAS + rf_original.COLUMNAS_NUMERICAS)
OBJETIVO = rf_original.COLUMNA_OBJETIVO
NORMAL = CLASES.index("normal")


def _json_seguro(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {str(k): _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_seguro(v) for v in valor]
    if isinstance(valor, np.ndarray):
        return _json_seguro(valor.tolist())
    if isinstance(valor, np.generic):
        return _json_seguro(valor.item())
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor


def _guardar_json(ruta: Path, datos: Any) -> None:
    ruta.write_text(json.dumps(_json_seguro(datos), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding="utf-8")


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def _tasa(n: int, d: int) -> float | None:
    return float(n / d) if d else None


def calcular_metricas(reales: np.ndarray, predichas: np.ndarray) -> dict[str, Any]:
    """Métricas sin ponderar sobre etiquetas enteras; ataque = no normal.

    Confundir dos clases de ataque es un error multiclase, pero no un falso
    negativo del filtro. FNR se divide por ataques; FPR, por benignos.
    """
    reales, predichas = np.asarray(reales), np.asarray(predichas)
    if len(reales) == 0 or reales.shape != predichas.shape:
        raise ValueError("Etiquetas vacías o dimensiones diferentes.")
    if not np.isin(reales, range(len(CLASES))).all() or not np.isin(
        predichas, range(len(CLASES))
    ).all():
        raise ValueError("Se recibieron etiquetas fuera de las tres clases.")
    reporte = classification_report(
        reales, predichas, labels=list(range(len(CLASES))),
        target_names=list(CLASES), output_dict=True, zero_division=0,
    )
    ataque = reales != NORMAL
    alerta = predichas != NORMAL
    vp = int(np.sum(ataque & alerta))
    fn = int(np.sum(ataque & ~alerta))
    fp = int(np.sum(~ataque & alerta))
    vn = int(np.sum(~ataque & ~alerta))
    return {
        "accuracy": float(np.mean(reales == predichas)),
        "precision_macro": reporte["macro avg"]["precision"],
        "recall_macro": reporte["macro avg"]["recall"],
        "f1_macro": reporte["macro avg"]["f1-score"],
        "f1_ponderado": reporte["weighted avg"]["f1-score"],
        "por_clase": {c: reporte[c] for c in CLASES},
        "matriz_confusion": {
            "labels": CLASES,
            "valores": confusion_matrix(reales, predichas,
                                         labels=list(range(len(CLASES)))).tolist(),
        },
        "filtrado_binario": {
            "verdaderos_positivos": vp, "falsos_negativos": fn,
            "falsos_positivos": fp, "verdaderos_negativos": vn,
            "total_ataques": vp + fn, "total_benignos": vn + fp,
            "total_alertas": vp + fp, "total_filtrados": vn + fn,
            "recall_ataques": _tasa(vp, vp + fn),
            "precision_alertas": _tasa(vp, vp + fp),
            "tasa_falsos_negativos": _tasa(fn, vp + fn),
            "tasa_falsos_positivos": _tasa(fp, vn + fp),
            "proporcion_benignos_entre_alertas": _tasa(fp, vp + fp),
            "proporcion_filtrada": _tasa(vn + fn, len(reales)),
        },
    }


def comparar_predicciones(reales: np.ndarray, referencia: np.ndarray,
                         candidata: np.ndarray) -> dict[str, int]:
    """Cuenta transiciones sobre los mismos flujos, sin inferencia causal."""
    bien_ref, bien_cand = referencia == reales, candidata == reales
    ataque = reales != NORMAL
    alerta_ref, alerta_cand = referencia != NORMAL, candidata != NORMAL
    return {
        "ambos_aciertan": int(np.sum(bien_ref & bien_cand)),
        "aciertos_recuperados": int(np.sum(~bien_ref & bien_cand)),
        "errores_introducidos": int(np.sum(bien_ref & ~bien_cand)),
        "ambos_fallan": int(np.sum(~bien_ref & ~bien_cand)),
        "ataques_antes_omitidos_ahora_detectados": int(np.sum(ataque & ~alerta_ref & alerta_cand)),
        "ataques_antes_detectados_ahora_omitidos": int(np.sum(ataque & alerta_ref & ~alerta_cand)),
        "falsas_alertas_resueltas": int(np.sum(~ataque & alerta_ref & ~alerta_cand)),
        "nuevas_falsas_alertas": int(np.sum(~ataque & ~alerta_ref & alerta_cand)),
    }


def cargar_datos(ruta: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Usa el dataset homologado que ya utilizaste; "
            "este experimento no lo regenera."
        )
    encabezado = pd.read_csv(ruta, nrows=0).columns.tolist()
    faltan = sorted(set(FEATURES + [OBJETIVO]) - set(encabezado))
    if faltan:
        raise ValueError(f"Faltan columnas: {', '.join(faltan)}")
    datos = pd.read_csv(ruta, usecols=FEATURES + [OBJETIVO])
    if datos.empty or datos.isna().any().any():
        raise ValueError("El CSV está vacío o contiene valores ausentes en las entradas.")
    for c in rf_original.COLUMNAS_NUMERICAS:
        datos[c] = pd.to_numeric(datos[c], errors="raise")
        if not np.isfinite(datos[c].to_numpy()).all():
            raise ValueError(f"La columna {c} contiene valores no finitos.")
    y = datos[OBJETIVO]
    if sorted(y.unique().tolist()) != list(CLASES):
        raise ValueError(f"Se requieren exactamente las clases {CLASES}; se recibió {y.unique().tolist()}.")
    if y.value_counts().min() < 4:
        raise ValueError("Hay menos de cuatro registros de alguna clase.")
    X = datos[FEATURES].copy()
    rf_original._validar_columnas_entrenamiento(X)
    return X, y, encabezado


def construir_modelos(hilos: int, incluir_rf_alternativo: bool) -> dict[str, Any]:
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise RuntimeError(
            "Falta XGBoost. Ejecuta: python -m pip install -r requirements-comparacion.txt"
        ) from e
    rf = clone(rf_original.crear_pipeline_random_forest().named_steps["modelo"])
    rf.set_params(n_jobs=hilos)
    arbol = clone(arbol_original.crear_pipeline_arbol_decision().named_steps["modelo"])
    modelos = {"arbol_decision": arbol, "random_forest_actual": rf}
    if incluir_rf_alternativo:
        # Configuración declarada en el JSON histórico, NO la ejecutada en la
        # fábrica actual. Se ensaya como alternativa independiente y explícita.
        modelos["random_forest_alternativo"] = clone(rf).set_params(
            max_depth=12, min_samples_leaf=5, min_samples_split=15,
        )
    modelos["xgboost"] = XGBClassifier(
        objective="multi:softprob", num_class=len(CLASES),
        n_estimators=200, max_depth=3, learning_rate=0.1,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist", eval_metric="mlogloss",
        random_state=rf_original.RANDOM_STATE, n_jobs=hilos,
    )
    return modelos


def auditar_representacion(X: pd.DataFrame, y: pd.Series,
                          train: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    """Diagnóstico descriptivo. No altera muestras ni entrena con etiquetas test."""
    unicos = {c: int(X[c].iloc[train].nunique()) for c in FEATURES}
    firmas = pd.util.hash_pandas_object(X, index=False).to_numpy()
    tabla = pd.DataFrame({"firma": firmas, "clase": y.to_numpy()})
    conteos = tabla.groupby(["firma", "clase"]).size().unstack(fill_value=0)
    ambiguos = conteos.loc[(conteos > 0).sum(axis=1) > 1]
    f_train, f_test = np.unique(firmas[train]), np.unique(firmas[test])
    compartidas = np.intersect1d(f_train, f_test)
    return {
        "valores_distintos_en_entrenamiento": unicos,
        "columnas_constantes_en_entrenamiento": [c for c, n in unicos.items() if n == 1],
        "total_vectores_distintos": int(len(conteos)),
        "vectores_con_etiquetas_distintas": int(len(ambiguos)),
        "registros_en_vectores_con_etiquetas_distintas": int(ambiguos.to_numpy().sum()),
        "vectores_presentes_en_train_y_test": int(len(compartidas)),
        "registros_test_cuyo_vector_aparece_en_train": int(np.isin(firmas[test], f_train).sum()),
        "cota_empirica_accuracy_de_un_clasificador_determinista_sobre_estos_vectores":
            float(conteos.max(axis=1).sum() / len(X)),
        "advertencia": (
            "Las firmas son hashes de 64 bits de las variables homologadas. "
            "Vectores iguales no prueban que sean el mismo flujo ni fuga de datos: "
            "la homologación puede colapsar flujos distintos. La cota usa todo el "
            "CSV como diagnóstico posterior, no es una métrica de prueba ni de "
            "generalización, y no se usa para ajustar modelos."
        ),
    }


def contrastar_json_historico(resultados: dict[str, Any], n: int, ntest: int) -> dict[str, Any]:
    ruta = rf_original.RUTA_METRICAS
    if not ruta.is_file():
        return {"estado": "sin_json_historico"}
    historico = json.loads(ruta.read_text(encoding="utf-8-sig"))
    actuales = resultados["random_forest_actual"]["parametros_efectivos"]
    declarados = historico.get("parametros_modelo", {})
    diferencias = {k: {"json_historico": v, "codigo_actual": actuales.get(k)}
                   for k, v in declarados.items() if actuales.get(k) != v}
    matriz = historico.get("matriz_confusion", {})
    nueva = resultados["random_forest_actual"]["prueba"]["matriz_confusion"]
    poblacion = (historico.get("total_eventos") == n and
                 historico.get("total_prueba") == ntest)
    return {
        "ruta": str(ruta), "sha256": _sha256(ruta),
        "diferencias_parametros_declarados_vs_codigo": diferencias,
        "mismos_totales_de_poblacion": poblacion,
        "misma_matriz_confusion_agregada": bool(
            poblacion and matriz.get("labels") == list(CLASES) and
            matriz.get("valores") == nueva["valores"]),
        "advertencia": (
            "La coincidencia agregada no prueba identidad de los registros. "
            "Las métricas y el modelo oficiales no se reemplazan."
        ),
    }


def auditar_incidentes(modelos: dict[str, Any], preprocesador: Any,
                      Xtrain: pd.DataFrame, encoder: LabelEncoder,
                      salida: Path) -> dict[str, Any]:
    ruta = RAIZ / "data" / "alertas_simuladas.json"
    if not ruta.is_file():
        return {"estado": "sin_alertas_simuladas"}
    from src.detector import detectar_incidentes
    from src.ml_predictor import _construir_features_incidente
    from src.severity_engine import calcular_severidad
    alertas = json.loads(ruta.read_text(encoding="utf-8-sig"))
    incidentes = detectar_incidentes(alertas)
    if not incidentes:
        return {"estado": "sin_incidentes"}
    entradas = pd.DataFrame([
        _construir_features_incidente(i, calcular_severidad(i)) for i in incidentes
    ], columns=FEATURES)
    resumen = {
        "estado": "diagnostico_exploratorio", "n_incidentes": len(incidentes),
        "sha256_alertas": _sha256(ruta), "categorias_no_vistas": {},
        "numericas_fuera_del_rango_de_entrenamiento": {}, "clases_predichas": {},
        "advertencia": "No se calculan métricas de detección de incidentes sin etiquetas "
                       "de referencia independientes; tampoco se ejecutan Qwen ni main.",
    }
    for c in rf_original.COLUMNAS_CATEGORICAS:
        fuera = ~entradas[c].isin(Xtrain[c].unique())
        resumen["categorias_no_vistas"][c] = {
            "n_incidentes": int(fuera.sum()),
            "valores": sorted(str(v) for v in entradas.loc[fuera, c].unique()),
        }
    for c in rf_original.COLUMNAS_NUMERICAS:
        minimo, maximo = float(Xtrain[c].min()), float(Xtrain[c].max())
        resumen["numericas_fuera_del_rango_de_entrenamiento"][c] = {
            "min_train": minimo, "max_train": maximo,
            "n_incidentes_fuera": int(((entradas[c] < minimo) | (entradas[c] > maximo)).sum()),
        }
    Z = preprocesador.transform(entradas).astype(np.float32)
    tabla = entradas.copy()
    tabla.insert(0, "id_incidente", [i.get("id_alerta", i.get("id_incidente", "")) for i in incidentes])
    for nombre, modelo in modelos.items():
        pred = encoder.inverse_transform(np.asarray(modelo.predict(Z), dtype=int))
        tabla[nombre] = pred
        resumen["clases_predichas"][nombre] = pd.Series(pred).value_counts().to_dict()
    tabla.to_csv(salida / "predicciones_incidentes_exploratorias.csv", index=False, encoding="utf-8-sig")
    return resumen


def generar_figuras(resultados: dict[str, Any], salida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter
    nombres = list(resultados)
    etiquetas = [n.replace("_", " ").replace("random forest", "Random Forest")
                 .replace("arbol decision", "Árbol de Decisión").replace("xgboost", "XGBoost")
                 for n in nombres]
    figuras = salida / "figuras"
    figuras.mkdir()
    def guardar(fig: Any, nombre: str) -> None:
        fig.tight_layout()
        fig.savefig(figuras / f"{nombre}.png", dpi=250, bbox_inches="tight")
        fig.savefig(figuras / f"{nombre}.pdf", bbox_inches="tight")
        plt.close(fig)
    series = {
        "f1_macro": ("F1 macro (%)", [100 * resultados[n]["prueba"]["f1_macro"] for n in nombres]),
        "falsos_positivos": ("Flujos benignos marcados como alerta", [resultados[n]["prueba"]["filtrado_binario"]["falsos_positivos"] for n in nombres]),
        "falsos_negativos": ("Ataques clasificados como normales", [resultados[n]["prueba"]["filtrado_binario"]["falsos_negativos"] for n in nombres]),
    }
    for clave, (titulo, valores) in series.items():
        fig, ax = plt.subplots(figsize=(8, 4.6))
        barras = ax.bar(range(len(nombres)), valores, color=["#64748b", "#2563eb", "#a16207", "#0f766e"][:len(nombres)])
        ax.set_xticks(range(len(nombres)), [e.replace(" ", "\n", 1) for e in etiquetas])
        ax.set_ylabel(titulo)
        ax.set_ylim(0, (max(valores) or 1) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for barra, valor in zip(barras, valores):
            texto = f"{valor:.1f}".replace(".", ",") if clave == "f1_macro" else f"{valor:,}".replace(",", ".")
            ax.annotate(texto, (barra.get_x()+barra.get_width()/2, valor), xytext=(0, 5), textcoords="offset points", ha="center")
        if clave == "f1_macro":
            ax.set_ylim(0, 108)
            ax.axhline(90, color="#991b1b", linestyle="--", linewidth=1, label="Meta interna 90 %")
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
        guardar(fig, f"comparacion_{clave}")
    for nombre, etiqueta in zip(nombres, etiquetas):
        matriz = np.asarray(resultados[nombre]["prueba"]["matriz_confusion"]["valores"])
        proporciones = np.divide(matriz, matriz.sum(axis=1, keepdims=True),
                                out=np.zeros(matriz.shape, dtype=float),
                                where=matriz.sum(axis=1, keepdims=True) != 0)
        fig, ax = plt.subplots(figsize=(6.2, 5))
        im = ax.imshow(proporciones, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                numero = f"{matriz[i,j]:,}".replace(",", ".")
                pct = f"{100*proporciones[i,j]:.1f}".replace(".", ",")
                ax.text(j, i, f"{numero}\n{pct} %", ha="center", va="center",
                        color="white" if proporciones[i,j] > 0.55 else "#172554")
        ax.set(xticks=range(len(CLASES)), yticks=range(len(CLASES)),
               xticklabels=CLASES, yticklabels=CLASES, xlabel="Clase predicha",
               ylabel="Clase real", title=etiqueta)
        fig.colorbar(im, ax=ax, label="Proporción de cada clase real")
        guardar(fig, f"matriz_confusion_{nombre}")


def _huellas_codigo() -> dict[str, str]:
    rutas = [Path(__file__), RAIZ / "src/random_forest.py",
             RAIZ / "src/modelo_arbol_decision.py", RAIZ / "src/procesar_kali_dataset.py",
             RAIZ / "src/ml_predictor.py"]
    return {p.relative_to(RAIZ).as_posix(): _sha256(p) for p in rutas if p.is_file()}


def ejecutar(args: argparse.Namespace) -> Path:
    hilos = min(args.hilos, os.cpu_count() or 1)
    modelos = construir_modelos(hilos, args.incluir_rf_alternativo)
    import xgboost
    ruta_dataset = args.dataset.resolve()
    print("Leyendo el CSV homologado existente...", flush=True)
    X, y, encabezado = cargar_datos(ruta_dataset)
    encoder = LabelEncoder().fit(list(CLASES))  # Vocabulario predefinido, no aprendido del test.
    train, test = train_test_split(np.arange(len(X)), test_size=0.25, stratify=y,
                                  random_state=rf_original.RANDOM_STATE)
    yt, yv = encoder.transform(y.iloc[train]), encoder.transform(y.iloc[test])
    Xtrain, Xtest = X.iloc[train], X.iloc[test]
    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida = args.salida.resolve() / fecha
    salida.mkdir(parents=True, exist_ok=False)
    huellas = _huellas_codigo()
    protegidos = [RAIZ / "data/metricas_random_forest.json", RAIZ / "data/modelo_random_forest.joblib",
                  RAIZ / "data/reportes_generados.json", RAIZ / "dashboard.py", RAIZ / "main.py"]
    antes = {str(p): _sha256(p) for p in protegidos if p.is_file()}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ,
                                capture_output=True, text=True, check=False).stdout.strip() or None
        estado_git = subprocess.run(["git", "status", "--short"], cwd=RAIZ,
                                    capture_output=True, text=True, check=False).stdout
    except OSError:
        commit, estado_git = None, "git no disponible"
    protocolo = {
        "version_experimento": VERSION, "estado": "iniciado", "fecha_utc": fecha,
        "modo": "verificacion_tecnica" if args.verificacion_tecnica else "experimento_dataset_homologado",
        "dataset": {"ruta": str(ruta_dataset), "sha256": _sha256(ruta_dataset),
                    "total": len(X), "distribucion": y.value_counts().to_dict(),
                    "columnas_presentes": encabezado},
        "features": FEATURES, "objetivo": OBJETIVO, "clases": CLASES,
        "particion": {"semilla": rf_original.RANDOM_STATE, "test_size": 0.25,
                      "estratificada": True, "n_train": len(train), "n_test": len(test),
                      "sha256_indices_train": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
                      "sha256_indices_test": hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
                      "clases_train": y.iloc[train].value_counts().to_dict(),
                      "clases_test": y.iloc[test].value_counts().to_dict()},
        "entorno": {"python": platform.python_version(), "sistema": platform.platform(),
                    "procesador": platform.processor(), "cpu_logicos": os.cpu_count(),
                    "hilos": hilos, "numpy": np.__version__, "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
                    "xgboost": xgboost.__version__},
        "codigo_sha256": huellas, "commit_observado": commit, "estado_git": estado_git,
        "parametros_antes_del_ajuste": {n: m.get_params(deep=False) for n, m in modelos.items()},
        "ponderacion": "balanced de scikit-learn en árboles y RF; sample_weight "
                       "con la misma fórmula y solo y_train en XGBoost; evaluación sin ponderar",
        "pesos_por_clase_en_entrenamiento": {
            clase: float(len(yt) / (len(CLASES) * np.sum(yt == i)))
            for i, clase in enumerate(CLASES)
        },
        "seleccion": "Configuraciones prefijadas, sin búsqueda ni early stopping sobre test. "
                     "Comparación exploratoria sobre la partición ya usada en el estudio. "
                     "No constituye una evaluación confirmatoria sobre un nuevo conjunto externo.",
    }
    _guardar_json(salida / "protocolo.json", protocolo)
    np.savez_compressed(salida / "indices_particion.npz", train=train, test=test)
    preprocesador = clone(rf_original.crear_pipeline_random_forest().named_steps["preprocesador"])
    inicio = perf_counter()
    Ztrain = preprocesador.fit_transform(Xtrain).astype(np.float32)
    Ztest = preprocesador.transform(Xtest).astype(np.float32)
    tiempo_preproceso = perf_counter() - inicio
    pesos = compute_sample_weight(class_weight="balanced", y=yt)
    resultados, predicciones, filas = {}, {}, []
    print(f"Registros: {len(X):,}; entrenamiento: {len(train):,}; prueba: {len(test):,}", flush=True)
    with threadpool_limits(limits=hilos):
        for nombre, modelo in modelos.items():
            print(f"Entrenando {nombre}...", flush=True)
            inicio = perf_counter()
            if nombre == "xgboost":
                modelo.fit(Ztrain, yt, sample_weight=pesos, verbose=False)
            else:
                modelo.fit(Ztrain, yt)
            ajuste_s = perf_counter() - inicio
            pred_train = np.asarray(modelo.predict(Ztrain), dtype=int)
            modelo.predict(Ztest[:min(1024, len(test))])  # Calentamiento, fuera de la medición.
            inicio = perf_counter()
            pred = np.asarray(modelo.predict(Ztest), dtype=int)
            pred_s = perf_counter() - inicio
            resultado = {
                "parametros_efectivos": modelo.get_params(deep=False),
                "entrenamiento": calcular_metricas(yt, pred_train),
                "prueba": calcular_metricas(yv, pred),
                "tiempo_ajuste_s": ajuste_s, "tiempo_prediccion_lote_prueba_s": pred_s,
                "nota_tiempos": "Una ejecución, misma máquina; excluye lectura, "
                                "preprocesamiento, métricas, Qwen y revisión humana.",
            }
            if hasattr(modelo, "estimators_"):
                resultado["profundidad_arboles"] = {
                    "min": min(t.get_depth() for t in modelo.estimators_),
                    "max": max(t.get_depth() for t in modelo.estimators_),
                }
            elif hasattr(modelo, "get_depth"):
                resultado["profundidad_arbol"] = modelo.get_depth()
            resultados[nombre], predicciones[nombre] = resultado, pred
            m, f = resultado["prueba"], resultado["prueba"]["filtrado_binario"]
            filas.append({"modelo": nombre, "f1_macro_train": resultado["entrenamiento"]["f1_macro"],
                          "accuracy_test": m["accuracy"], "f1_macro_test": m["f1_macro"],
                          "precision_macro_test": m["precision_macro"], "recall_macro_test": m["recall_macro"],
                          **f, "ajuste_s": ajuste_s, "prediccion_lote_s": pred_s})
            _guardar_json(salida / f"metricas_{nombre}.json", resultado)
            # Evidencia acotada con las mismas variables, no solo cifras agregadas.
            for error, mascara in {"falso_negativo": (yv != NORMAL) & (pred == NORMAL),
                                   "falso_positivo": (yv == NORMAL) & (pred != NORMAL)}.items():
                pos = np.flatnonzero(mascara)[:50]
                muestra = Xtest.iloc[pos].copy()
                muestra.insert(0, "fila_csv_cero_based", test[pos])
                muestra["clase_real"] = encoder.inverse_transform(yv[pos])
                muestra["clase_predicha"] = encoder.inverse_transform(pred[pos])
                muestra.to_csv(salida / f"muestra_{error}_{nombre}.csv", index=False, encoding="utf-8-sig")
            if args.guardar_modelos:
                import joblib
                joblib.dump({"preprocesador": preprocesador, "modelo": modelo,
                             "encoder_etiquetas": encoder, "features": FEATURES,
                             "uso": "experimental; no reemplazar directamente el predictor productivo"},
                            salida / f"modelo_experimental_{nombre}.joblib")
            print(f"  F1 macro={m['f1_macro']:.4f}; FP={f['falsos_positivos']}; FN={f['falsos_negativos']}", flush=True)
    tabla = pd.DataFrame({"fila_csv_cero_based": test, "clase_real": encoder.inverse_transform(yv),
                          **{n: encoder.inverse_transform(p) for n, p in predicciones.items()}})
    tabla.to_csv(salida / "predicciones_prueba.csv.gz", index=False, compression="gzip")
    pd.DataFrame(filas).to_csv(salida / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
    referencia = predicciones["random_forest_actual"]
    transiciones = {n: comparar_predicciones(yv, referencia, p)
                    for n, p in predicciones.items() if n != "random_forest_actual"}
    auditoria = auditar_representacion(X, y, train, test)
    diagnostico = {
        "representacion": auditoria,
        "contraste_json_historico": contrastar_json_historico(resultados, len(X), len(test)),
        "comparaciones_pareadas_frente_a_rf_actual": transiciones,
        "referencia_siempre_clase_mayoritaria": calcular_metricas(
            yv, np.full_like(yv, int(np.bincount(yt).argmax()))),
    }
    if args.auditar_incidentes:
        diagnostico["incidentes_soar"] = auditar_incidentes(modelos, preprocesador, Xtrain, encoder, salida)
    _guardar_json(salida / "diagnostico.json", diagnostico)
    if not args.sin_figuras:
        generar_figuras(resultados, salida)
    despues = {p: _sha256(Path(p)) for p in antes}
    if antes != despues or huellas != _huellas_codigo():
        raise RuntimeError("Cambió un archivo protegido durante el experimento; revisa la ejecución.")
    protocolo.update({"estado": "completado", "tiempo_preprocesamiento_s": tiempo_preproceso,
                      "matriz_preprocesada": {"columnas": Ztrain.shape[1], "dtype": str(Ztrain.dtype),
                                               "dispersa": bool(sparse.issparse(Ztrain))},
                      "archivos_productivos_sin_cambios": True})
    _guardar_json(salida / "protocolo.json", protocolo)
    _guardar_json(salida / "resultados_completos.json", {
        "protocolo": protocolo, "resultados": resultados, "diagnostico": diagnostico,
    })
    print(f"\nResultados guardados en: {salida}", flush=True)
    print("Comparación experimental terminada. No se seleccionó ni integró automáticamente un modelo.")
    return salida


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=rf_original.RUTA_DATASET_KALI)
    parser.add_argument("--salida", type=Path, default=RAIZ / "outputs/comparacion_modelos")
    parser.add_argument("--hilos", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--incluir-rf-alternativo", action="store_true",
                        help="Añade RF con profundidad 12, hoja 5 y split 15 como configuración alternativa.")
    parser.add_argument("--auditar-incidentes", action="store_true",
                        help="Diagnóstico de entradas y predicciones de incidentes; no ejecuta Qwen.")
    parser.add_argument("--guardar-modelos", action="store_true")
    parser.add_argument("--sin-figuras", action="store_true")
    parser.add_argument("--verificacion-tecnica", action="store_true",
                        help="Identifica salidas de pruebas técnicas, no resultados de la tesis.")
    args = parser.parse_args(argv)
    if args.hilos < 1:
        parser.error("--hilos debe ser positivo")
    return ejecutar(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
