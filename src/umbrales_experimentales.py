"""Políticas de revisión sobre puntuaciones; no modifica el predictor de Q-SOC."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASES = ("fallido", "normal", "sospechoso")
NORMAL = 1
SEMILLA = 42
FRACCION_GRUPOS_VALIDACION = 0.20
UMBRALES = tuple(sorted(set(
    [round(i / 100, 2) for i in range(5, 100)]
    + [0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999999, 1.0]
)))


def predecir_estable(modelo, valores):
    """Suma los árboles en orden secuencial; restaura n_jobs al terminar."""
    tiene_hilos = hasattr(modelo, "n_jobs")
    anteriores = getattr(modelo, "n_jobs", None)
    try:
        if tiene_hilos:
            modelo.n_jobs = 1
        return validar_probabilidades(modelo.predict_proba(valores))
    finally:
        if tiene_hilos:
            modelo.n_jobs = anteriores


def comprobar_serializacion(antes, despues):
    """Tolera redondeo mínimo, nunca decisiones distintas en la grilla fijada."""
    a, b = validar_probabilidades(antes), validar_probabilidades(despues)
    if a.shape != b.shape:
        raise RuntimeError("El artefacto guardado devuelve otras dimensiones.")
    diferencia = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.allclose(a, b, rtol=0.0, atol=1e-14):
        raise RuntimeError(f"El artefacto guardado cambia las puntuaciones; diferencia máxima {diferencia:.3g}.")
    if not np.array_equal(a.argmax(axis=1), b.argmax(axis=1)):
        raise RuntimeError("La diferencia de puntuaciones cambia al menos una clase argmax.")
    for t in UMBRALES:
        if not np.array_equal(a[:, 0] + a[:, 2] >= t, b[:, 0] + b[:, 2] >= t):
            raise RuntimeError(f"La diferencia de puntuaciones cambia decisiones en el umbral {t}.")
    return {"diferencia_absoluta_maxima": diferencia, "tolerancia_absoluta": 1e-14,
            "tolerancia_relativa": 0.0, "clases_identicas": True,
            "decisiones_identicas_en_toda_la_grilla": True,
            "puntuaciones_bit_a_bit_identicas": bool(np.array_equal(a, b))}


def validar_probabilidades(probabilidades):
    p = np.asarray(probabilidades, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-8)):
        raise ValueError("Se requieren tres puntuaciones finitas, no negativas y con suma uno.")
    return p


def firmas_vectores(valores):
    x = np.asarray(valores, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Las entradas deben ser una matriz finita.")
    return pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy(dtype=np.uint64)


def sha_indices(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def dividir_desarrollo(indices_pool, y, firmas):
    """Un solo split por firmas: ningún vector exacto cruza ajuste/validación."""
    pool = np.asarray(indices_pool, dtype=np.int64)
    grupos = np.asarray(firmas)[pool]
    if len(np.unique(grupos)) < 3:
        raise ValueError("No hay suficientes grupos para separar ajuste y validación.")
    dividir = GroupShuffleSplit(n_splits=1, test_size=FRACCION_GRUPOS_VALIDACION,
                                random_state=SEMILLA)
    t, v = next(dividir.split(np.zeros(len(pool)), groups=grupos))
    ajuste, validacion = pool[t], pool[v]
    if set(np.asarray(y)[ajuste]) != {0, 1, 2} or set(np.asarray(y)[validacion]) != {0, 1, 2}:
        raise ValueError("Alguna clase falta en ajuste o validación; no se cambió la semilla para forzar resultados.")
    if np.intersect1d(firmas[ajuste], firmas[validacion]).size:
        raise RuntimeError("Se mezclaron grupos entre ajuste y validación.")
    return ajuste, validacion


def aplicar_politica(probabilidades, politica):
    p = validar_probabilidades(probabilidades)
    if politica["regla"] == "argmax_tres_clases":
        return p.argmax(axis=1) != NORMAL
    if politica["regla"] != "puntuacion_alerta":
        raise ValueError("Regla de revisión desconocida.")
    umbral = politica.get("umbral")
    if isinstance(umbral, bool) or not isinstance(umbral, (float, int)) or not np.isfinite(umbral) or not 0 <= umbral <= 1:
        raise ValueError("El umbral debe estar entre cero y uno.")
    return p[:, 0] + p[:, 2] >= umbral


def medir_decision(reales, revision):
    y = np.asarray(reales)
    alerta = np.asarray(revision)
    if y.ndim != 1 or alerta.ndim != 1 or len(y) != len(alerta) or alerta.dtype != np.bool_:
        raise ValueError("Se requieren etiquetas y decisiones booleanas de la misma longitud.")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Hay etiquetas fuera de las tres clases.")
    ataque = y != NORMAL
    tp = int(np.sum(ataque & alerta))
    fn = int(np.sum(ataque & ~alerta))
    fp = int(np.sum(~ataque & alerta))
    tn = int(np.sum(~ataque & ~alerta))
    return resumir_conteos(tp, fn, fp, tn)


def resumir_conteos(tp, fn, fp, tn):
    def tasa(a, b):
        return float(a / b) if b else None
    return {
        "n": int(tp + fn + fp + tn),
        "verdaderos_positivos": int(tp), "falsos_negativos": int(fn),
        "falsos_positivos": int(fp), "verdaderos_negativos": int(tn),
        "ataques_reales": int(tp + fn), "benignos_reales": int(fp + tn),
        "revision_sugerida": int(tp + fp), "filtrados": int(tn + fn),
        "tasa_falsos_positivos": tasa(fp, fp + tn),
        "tasa_falsos_negativos": tasa(fn, tp + fn),
        "recall_ataques": tasa(tp, tp + fn),
        "precision_alertas": tasa(tp, tp + fp),
        "proporcion_filtrada": tasa(tn + fn, tp + fn + fp + tn),
    }


def sumar_metricas(metricas):
    claves = ("verdaderos_positivos", "falsos_negativos", "falsos_positivos", "verdaderos_negativos")
    return resumir_conteos(*(sum(m[k] for m in metricas) for k in claves))


def transiciones(reales, base, candidata):
    y, b, c = np.asarray(reales), np.asarray(base), np.asarray(candidata)
    if y.shape != b.shape or y.shape != c.shape or b.dtype != np.bool_ or c.dtype != np.bool_:
        raise ValueError("Dimensiones o tipos incompatibles en las transiciones.")
    ataque = y != NORMAL
    return {
        "falsas_alertas_resueltas": int(np.sum(~ataque & b & ~c)),
        "nuevas_falsas_alertas": int(np.sum(~ataque & ~b & c)),
        "ataques_antes_omitidos_ahora_detectados": int(np.sum(ataque & ~b & c)),
        "ataques_antes_detectados_ahora_omitidos": int(np.sum(ataque & b & ~c)),
    }


def seleccionar_politica(y_validacion, p_validacion):
    """Elige solo con validación; nunca recibe datos ni métricas de prueba."""
    p = validar_probabilidades(p_validacion)
    if len(p) == 0:
        raise ValueError("La validación está vacía.")
    regla_base = {"regla": "argmax_tres_clases", "umbral": None}
    base = aplicar_politica(p, regla_base)
    mb = medir_decision(y_validacion, base)
    if mb["ataques_reales"] == 0 or mb["benignos_reales"] == 0:
        raise ValueError("La validación necesita ataques y flujos benignos.")
    curva, elegibles = [], []
    for t in UMBRALES:
        politica = {"regla": "puntuacion_alerta", "umbral": float(t)}
        candidata = aplicar_politica(p, politica)
        m = medir_decision(y_validacion, candidata)
        cambios = transiciones(y_validacion, base, candidata)
        admisible = (m["falsos_positivos"] < mb["falsos_positivos"]
                     and cambios["ataques_antes_detectados_ahora_omitidos"] == 0)
        fila = {"umbral": float(t), **m, **cambios, "admisible_validacion": bool(admisible)}
        curva.append(fila)
        if admisible:
            elegibles.append(fila)
    if elegibles:
        mejor = min(elegibles, key=lambda r: (r["falsos_positivos"], r["falsos_negativos"],
                                              abs(r["umbral"] - 0.5), r["umbral"]))
        politica = {"regla": "puntuacion_alerta", "umbral": mejor["umbral"]}
    else:
        politica = regla_base
    propuesta = aplicar_politica(p, politica)
    return {
        "politica": politica,
        "mejora_en_validacion": bool(elegibles),
        "criterio": "Menos FP sin perder ningún ataque detectado por argmax en validación. "
                    "Desempates: menos FN, cercanía a 0.5 y umbral menor.",
        "origen_seleccion": "solo_validacion",
        "puntuacion": "p_fallido + p_sospechoso; puntuación sin calibración",
        "base_validacion": mb,
        "propuesta_validacion": medir_decision(y_validacion, propuesta),
        "transiciones_validacion": transiciones(y_validacion, base, propuesta),
        "limite": "La condición se exige en validación y no garantiza ausencia de pérdidas en otra captura.",
    }, curva
