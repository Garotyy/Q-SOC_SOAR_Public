"""Auditoría de etiquetas y cobertura usando particiones ya guardadas de Q-SOC.

No entrena, predice, consulta Qwen ni cambia modelos. Los CSV originales se
leen únicamente por su columna Label. La limpieza y los índices se toman de
la procedencia verificada de la ejecución de umbrales indicada.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd

VERSION = "auditoria_cobertura_guardada_v1"
FASES = ("ajuste", "validacion", "test_historico_explorado")
REGLAS = ("base_experimental", "propuesta_experimental", "rf_integrado_referencia")
CLASES = ("fallido", "normal", "sospechoso")
COLUMNA_ID = "fila_csv_homologado_cero_based"
COLUMNA_FILA = "fila_captura_cero_based"
MEDIDAS = ("n", "verdaderos_positivos", "falsos_negativos", "falsos_positivos", "verdaderos_negativos")


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_json(ruta: Path) -> dict:
    with ruta.open(encoding="utf-8-sig") as f:
        dato = json.load(f)
    if not isinstance(dato, dict):
        raise ValueError(f"Se esperaba un objeto JSON: {ruta.name}")
    return dato


def clave(etiqueta: str) -> str:
    return str(etiqueta).strip().upper()


def clase_qsoc(etiqueta: str) -> str:
    valor = clave(etiqueta)
    if valor == "BENIGN":
        return "normal"
    return "fallido" if "PATATOR" in valor or "PORT" in valor else "sospechoso"


def nombre_seguro(valor: str) -> str:
    if not isinstance(valor, str) or not valor or "/" in valor or "\\" in valor or Path(valor).name != valor:
        raise ValueError("Nombre de captura inválido en la referencia.")
    return valor


def resolver_referencia(raiz: Path, carpeta: str, nombre: str, huella: str) -> Path:
    if not isinstance(huella, str) or len(huella) != 64:
        raise ValueError("Falta una huella de referencia válida.")
    candidatas = sorted((raiz / "outputs" / carpeta).glob("*/" + nombre))
    for ruta in candidatas:
        if sha256(ruta) == huella:
            return ruta
    raise FileNotFoundError(f"No se encontró {nombre} con la huella de la ejecución. Conserva su carpeta original en outputs/{carpeta}.")


def verificar_indices(valores, n: int, esperado: str, nombre: str) -> np.ndarray:
    a = np.asarray(valores)
    if a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0:
        raise ValueError(f"Índices inválidos: {nombre}.")
    if a.min() < 0 or a.max() >= n or len(np.unique(a)) != len(a):
        raise ValueError(f"Hay índices repetidos o fuera de rango: {nombre}.")
    a = a.astype("<i8", copy=False)
    if hashlib.sha256(a.tobytes()).hexdigest() != esperado:
        raise ValueError(f"Los índices no reproducen su huella: {nombre}.")
    return a


def cargar_particiones(carpeta: Path, inicial_ruta: Path, resultado: dict, inicial: dict) -> tuple[pd.DataFrame, dict]:
    protocolo = resultado["protocolo"]
    detalles = protocolo["particiones"]
    n = sum(int(detalles[f]["n"]) for f in FASES)
    if n != int(inicial["protocolo"]["dataset"]["total"]):
        raise ValueError("Las referencias declaran poblaciones distintas.")
    ruta_npz = inicial_ruta.parent / "indices_particion.npz"
    with np.load(ruta_npz, allow_pickle=False) as archivo:
        train = verificar_indices(archivo["train"], n, inicial["protocolo"]["particion"]["sha256_indices_train"], "train integrado")
        test = verificar_indices(archivo["test"], n, inicial["protocolo"]["particion"]["sha256_indices_test"], "test histórico")
    if len(train) + len(test) != n or np.intersect1d(train, test).size:
        raise ValueError("El entrenamiento y el test históricos se superponen o están incompletos.")
    ruta_fases = carpeta / "particiones.csv.gz"
    fases = pd.read_csv(ruta_fases, dtype={COLUMNA_ID: "int64", "rol": "string"})
    if set(fases.columns) != {COLUMNA_ID, "rol"} or len(fases) != n or fases[COLUMNA_ID].duplicated().any():
        raise ValueError("El archivo de particiones tiene columnas, tamaño o índices inesperados.")
    if set(fases["rol"].dropna()) != set(FASES) or fases["rol"].isna().any():
        raise ValueError("Hay roles de partición ausentes o desconocidos.")
    for f in FASES:
        indices = verificar_indices(fases.loc[fases["rol"] == f, COLUMNA_ID].to_numpy(), n, detalles[f]["sha256_indices"], f)
        if len(indices) != int(detalles[f]["n"]):
            raise ValueError(f"Tamaño incorrecto de la partición {f}.")
    historico = fases.loc[fases["rol"] == "test_historico_explorado", COLUMNA_ID].to_numpy()
    if not np.array_equal(historico, test):
        raise ValueError("El test histórico guardado no es el de la referencia inicial.")
    pool = fases.loc[fases["rol"].isin(["ajuste", "validacion"]), COLUMNA_ID].to_numpy()
    if not np.array_equal(np.sort(pool), np.sort(train)):
        raise ValueError("Ajuste y validación no reconstruyen el entrenamiento integrado.")
    ruta_meta = carpeta / "procedencia_muestras.csv.gz"
    meta = pd.read_csv(ruta_meta, dtype={COLUMNA_ID: "int64", COLUMNA_FILA: "int64", "archivo_captura": "string", "clase": "string"})
    requeridas = {COLUMNA_ID, COLUMNA_FILA, "archivo_captura", "clase"}
    if not requeridas.issubset(meta.columns) or len(meta) != n or meta[list(requeridas)].isna().any().any():
        raise ValueError("La procedencia está incompleta.")
    if meta[COLUMNA_ID].duplicated().any() or not np.array_equal(np.sort(meta[COLUMNA_ID].to_numpy()), np.arange(n)):
        raise ValueError("La procedencia no cubre cada posición de la cohorte exactamente una vez.")
    if meta[COLUMNA_FILA].lt(0).any() or meta.duplicated(["archivo_captura", COLUMNA_FILA]).any():
        raise ValueError("La procedencia contiene filas originales inválidas o repetidas.")
    if not set(meta["clase"]).issubset(CLASES):
        raise ValueError("La procedencia contiene una clase desconocida.")
    meta = meta.merge(fases, on=COLUMNA_ID, how="inner", validate="one_to_one").sort_values(COLUMNA_ID)
    for f in FASES:
        observadas = {str(k): int(v) for k, v in meta.loc[meta["rol"] == f, "clase"].value_counts().items()}
        esperadas = {k: int(v) for k, v in detalles[f]["clases"].items() if int(v)}
        if observadas != esperadas:
            raise ValueError(f"Las clases de {f} no coinciden con el protocolo.")
    return meta, {str(p): sha256(p) for p in (ruta_npz, ruta_fases, ruta_meta)}


def contar_captura(ruta: Path, info: dict, meta: pd.DataFrame, chunk_size: int = 100000) -> dict:
    if sha256(ruta) != info["sha256"]:
        raise ValueError(f"Cambió la captura de desarrollo: {ruta.name}.")
    registros = meta.sort_values(COLUMNA_FILA)
    filas = registros[COLUMNA_FILA].to_numpy(dtype=np.int64)
    if len(filas) != int(info["registros_validos"]) or len(filas) == 0:
        raise ValueError(f"La cantidad de filas conservadas no coincide en {ruta.name}.")
    cabecera = pd.read_csv(ruta, nrows=0, skipinitialspace=True, encoding="utf-8-sig")
    columnas_label = [c for c in cabecera.columns if c.strip() == "Label"]
    if len(columnas_label) != 1:
        raise ValueError(f"No hay una única columna Label en {ruta.name}.")
    por_fase = {f: Counter() for f in FASES}
    crudas = Counter()
    n_raw = procesadas = 0
    print(f"Leyendo solo Label: {ruta.name}...", flush=True)
    with pd.read_csv(ruta, usecols=columnas_label, skipinitialspace=True, encoding="utf-8-sig", chunksize=chunk_size) as lector:
        for bloque in lector:
            etiqueta = bloque.iloc[:, 0].astype("string").str.strip()
            crudas.update({str(k): int(v) for k, v in etiqueta.fillna("SIN_ETIQUETA").value_counts().items()})
            inicio = n_raw
            n_raw += len(bloque)
            lo, hi = np.searchsorted(filas, [inicio, n_raw])
            if hi == lo:
                continue
            posiciones = filas[lo:hi] - inicio
            seleccion = etiqueta.iloc[posiciones]
            if seleccion.isna().any() or seleccion.eq("").any():
                raise ValueError(f"Una fila conservada no tiene etiqueta en {ruta.name}.")
            originales = seleccion.to_numpy(dtype=object)
            clases = np.asarray([clase_qsoc(s) for s in originales], dtype=object)
            if not np.array_equal(clases, registros["clase"].iloc[lo:hi].to_numpy(dtype=object)):
                raise ValueError(f"Las etiquetas originales contradicen las clases guardadas en {ruta.name}.")
            roles = registros["rol"].iloc[lo:hi].to_numpy(dtype=object)
            for f in FASES:
                por_fase[f].update(originales[roles == f].tolist())
            procesadas += int(hi - lo)
    if n_raw != int(info["registros_iniciales"]) or procesadas != len(filas):
        raise ValueError(f"Las filas originales o sus posiciones no coinciden en {ruta.name}.")
    if sha256(ruta) != info["sha256"]:
        raise ValueError(f"La captura cambió durante la lectura: {ruta.name}.")
    limpias = sum(por_fase.values(), Counter())
    descartadas = crudas - limpias
    if sum(descartadas.values()) != n_raw - procesadas:
        raise ValueError("El balance de etiquetas descartadas no coincide.")
    return {"archivo": ruta.name, "sha256": info["sha256"], "registros_originales": n_raw,
            "registros_conservados": procesadas, "registros_descartados": n_raw - procesadas,
            "etiquetas_crudas": dict(sorted(crudas.items())),
            "etiquetas_descartadas": dict(sorted(descartadas.items())),
            "por_fase": {f: dict(sorted(por_fase[f].items())) for f in FASES}}


def verificar_metricas(m: dict) -> None:
    for k in MEDIDAS:
        if not isinstance(m.get(k), int) or isinstance(m[k], bool) or m[k] < 0:
            raise ValueError("Hay conteos inválidos en el resultado reservado.")
    if m["n"] != sum(m[k] for k in MEDIDAS[1:]):
        raise ValueError("Una tabla de resultados reservados no suma su población.")


def consolidar_cobertura(desarrollo: list[dict], reservado: dict) -> tuple[list[dict], dict]:
    nombres = defaultdict(set)
    fases = {f: Counter() for f in FASES}
    for captura in desarrollo:
        for f in FASES:
            for etiqueta, n in captura["por_fase"][f].items():
                nombres[clave(etiqueta)].add(etiqueta)
                fases[f][clave(etiqueta)] += int(n)
    metricas = defaultdict(lambda: {r: Counter() for r in REGLAS})
    capturas = reservado["por_captura"]
    for c in capturas:
        for r in REGLAS:
            verificar_metricas(c[r])
            if any(sum(g[r][k] for g in c["por_etiqueta_original"]) != c[r][k] for k in MEDIDAS):
                raise ValueError("Las etiquetas reservadas no suman la captura.")
        for g in c["por_etiqueta_original"]:
            etiqueta = str(g["etiqueta_original"]).strip()
            if g["clase_qsoc"] != clase_qsoc(etiqueta):
                raise ValueError("El mapeo de etiquetas reservadas no coincide con el documentado.")
            k = clave(etiqueta)
            nombres[k].add(etiqueta)
            for r in REGLAS:
                verificar_metricas(g[r])
                if g[r]["n"] != int(g["n"]):
                    raise ValueError("La población por etiqueta es inconsistente.")
                incompatibles = ("verdaderos_positivos", "falsos_negativos") if clase_qsoc(etiqueta) == "normal" else ("falsos_positivos", "verdaderos_negativos")
                if any(g[r][x] for x in incompatibles):
                    raise ValueError("Una etiqueta reservada contiene conteos de otra clase real.")
                metricas[k][r].update({x: g[r][x] for x in MEDIDAS})
    for r in REGLAS:
        verificar_metricas(reservado["agregado"][r])
        if any(sum(c[r][k] for c in capturas) != reservado["agregado"][r][k] for k in MEDIDAS):
            raise ValueError("Las capturas no suman su resultado agregado.")
    filas = []
    for k in sorted(nombres):
        fila = {"etiqueta_canonica": k, "etiquetas_originales": sorted(nombres[k]), "clase_qsoc": clase_qsoc(k)}
        fila.update({f: int(fases[f][k]) for f in FASES})
        fila["entrenamiento_integrado"] = fila["ajuste"] + fila["validacion"]
        fila["capturas_reservadas"] = int(metricas[k][REGLAS[0]]["n"])
        fila["ausente_del_entrenamiento_integrado"] = fila["entrenamiento_integrado"] == 0
        fila["ausente_del_ajuste_experimental"] = fila["ajuste"] == 0
        fila["hay_caracter_de_sustitucion_en_etiqueta"] = any("\ufffd" in x for x in nombres[k])
        fila["resultados_reservados"] = {}
        for r in REGLAS:
            m = {x: int(metricas[k][r][x]) for x in MEDIDAS}
            ataques = m["verdaderos_positivos"] + m["falsos_negativos"]
            m["recall_ataques"] = m["verdaderos_positivos"] / ataques if ataques else None
            fila["resultados_reservados"][r] = m
        filas.append(fila)
    ataques = [f for f in filas if f["clase_qsoc"] != "normal" and f["capturas_reservadas"]]
    ausentes = [f for f in ataques if f["ausente_del_entrenamiento_integrado"]]
    return filas, {
        "etiquetas_ataque_con_ejemplos_en_entrenamiento_integrado": [f["etiquetas_originales"] for f in filas if f["clase_qsoc"] != "normal" and f["entrenamiento_integrado"]],
        "etiquetas_reservadas_ausentes_del_entrenamiento_integrado": [f["etiquetas_originales"] for f in ausentes],
        "flujos_ataque_reservados_de_etiquetas_ausentes": sum(f["capturas_reservadas"] for f in ausentes),
        "flujos_ataque_reservados_totales": sum(f["capturas_reservadas"] for f in ataques),
    }


def protegidos(raiz: Path) -> dict[str, str]:
    rutas = [raiz / x for x in ("dashboard.py", "dashboard_red.py", "main.py", "data/reportes_generados.json", "data/dataset_kali_homologado.csv")]
    rutas += list((raiz / "src").glob("*.py"))
    rutas += list((raiz / "data").glob("*.joblib"))
    rutas += list((raiz / "data").glob("metricas*.json"))
    rutas += list((raiz / "data").glob("modelo*.json"))
    return {str(p): sha256(p) for p in sorted(set(rutas)) if p.is_file()}


def resumen_md(resultado: dict) -> str:
    texto = ["# Auditoría de cobertura de entrenamiento Q-SOC", "", "Conteos sobre las particiones ya utilizadas; no se entrenó ni se evaluó un modelo nuevo.", "",
             "| Etiqueta original | Entrenamiento integrado | Ajuste experimental | Validación | Test histórico | Capturas reservadas | Detección del RF integrado en reservadas |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f in resultado["cobertura_por_etiqueta"]:
        m = f["resultados_reservados"]["rf_integrado_referencia"]
        tasa = "No aplica" if m["recall_ataques"] is None else f"{100*m['recall_ataques']:.2f}% ({m['verdaderos_positivos']}/{m['verdaderos_positivos']+m['falsos_negativos']})"
        etiqueta = " / ".join(f["etiquetas_originales"]).replace("|", "\\|")
        texto.append(f"| {etiqueta} | {f['entrenamiento_integrado']} | {f['ajuste']} | {f['validacion']} | {f['test_historico_explorado']} | {f['capturas_reservadas']} | {tasa} |")
    texto += ["", "## Interpretación y límites", ""] + ["- " + s for s in resultado["limites"]]
    return "\n".join(texto) + "\n"


def ejecutar(args) -> Path:
    inicio = perf_counter()
    raiz = args.raiz.resolve()
    ruta_resultado = args.resultado_umbrales.resolve()
    resultado = leer_json(ruta_resultado)
    p = resultado["protocolo"]
    if p.get("estado") != "completado" or p.get("modo") != "umbral_validacion_y_capturas_reservadas":
        raise ValueError("Se necesita el resultado completo de la prueba de umbrales.")
    referencia_ruta = resolver_referencia(raiz, "comparacion_final_red", "resultados_tres_modelos_red.json", p["referencia_tres_modelos_sha256"])
    inicial_ruta = resolver_referencia(raiz, "comparacion_modelos", "resultados_completos.json", p["referencia_inicial_sha256"])
    referencia = leer_json(referencia_ruta)
    inicial = leer_json(inicial_ruta)
    if referencia["protocolo"]["referencia_inicial_sha256"] != p["referencia_inicial_sha256"]:
        raise ValueError("Las referencias no provienen del mismo experimento inicial.")
    print("Verificando referencias y particiones guardadas...", flush=True)
    antes = protegidos(raiz)
    ruta_hom = raiz / "data" / "dataset_kali_homologado.csv"
    if antes.get(str(ruta_hom)) != inicial["protocolo"]["dataset"]["sha256"]:
        raise ValueError("El CSV homologado no coincide con la comparación inicial.")
    meta, insumos = cargar_particiones(ruta_resultado.parent, inicial_ruta, resultado, inicial)
    for r in (ruta_resultado, referencia_ruta, inicial_ruta):
        insumos[str(r)] = sha256(r)
    archivos = referencia["protocolo"]["cohorte"]["archivos"]
    nombres = [nombre_seguro(f["archivo"]) for f in archivos]
    if len(set(nombres)) != len(nombres) or set(meta["archivo_captura"].unique()) != set(nombres):
        raise ValueError("Las capturas de procedencia difieren de la cohorte de referencia.")
    desarrollo = []
    for info in archivos:
        nombre = nombre_seguro(info["archivo"])
        r = raiz / "data" / nombre
        desarrollo.append(contar_captura(r, info, meta.loc[meta["archivo_captura"] == nombre], args.tamano_bloque))
        insumos[str(r)] = info["sha256"]
    filas, hallazgos = consolidar_cobertura(desarrollo, resultado["evaluacion_capturas_reservadas"])
    if protegidos(raiz) != antes or any(sha256(Path(r)) != h for r, h in insumos.items()):
        raise RuntimeError("Un insumo o archivo protegido cambió durante la auditoría. No se guardaron conclusiones.")
    informe = {
        "protocolo": {"version": VERSION, "estado": "completado", "fecha_utc": datetime.now(timezone.utc).isoformat(),
                      "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                      "codigo_sha256": sha256(Path(__file__)), "insumos_sha256": insumos,
                      "archivos_protegidos_sha256": antes, "archivos_protegidos_conservados": True,
                      "n_integrado_train": sum(f["entrenamiento_integrado"] for f in filas),
                      "n_ajuste": sum(f["ajuste"] for f in filas), "n_validacion": sum(f["validacion"] for f in filas),
                      "n_test_historico": sum(f["test_historico_explorado"] for f in filas),
                      "etiquetas_normalizadas_solo_para_comparar": "strip y upper; se conservan los nombres originales",
                      "tiempo_auditoria_segundos": round(perf_counter() - inicio, 3)},
        "hallazgos_cobertura": hallazgos, "capturas_desarrollo": desarrollo,
        "cobertura_por_etiqueta": filas,
        "resultados_ya_observados_capturas_reservadas": resultado["evaluacion_capturas_reservadas"]["agregado"],
        "limites": [
            "Las etiquetas originales identifican escenarios del dataset; no son una taxonomía completa de amenazas ni identidades de incidentes.",
            "Una etiqueta ausente del entrenamiento es una brecha de cobertura. Esta auditoría no prueba que sea la única causa de las omisiones.",
            "La clase fallido agrupa PATATOR y PORT por el mapeo histórico; no demuestra fallos de autenticación en PortScan.",
            "Se reutilizan las filas conservadas por la limpieza previa, verificando captura y particiones; no se cambia ni repite la limpieza.",
            "Los conteos reservados proceden del JSON ya generado; no se vuelven a predecir flujos ni se buscan umbrales.",
            "Las capturas reservadas ya se examinaron; si se usan para ajustar el sistema pasan a ser datos de desarrollo.",
            "La ausencia de una familia y su mala detección no demuestran por sí solas sobreajuste; tampoco permiten culpar al dataset.",
            "Los resultados son sobre flujos y no demuestran eliminación del cuello de botella ni reducción medida del trabajo humano.",
            "Los caracteres de sustitución en nombres de etiquetas se señalan y se conservan; no se reparan datos silenciosamente.",
        ],
    }
    base = (raiz / "outputs" / "auditoria_cobertura").resolve()
    if not base.is_relative_to(raiz):
        raise ValueError("La carpeta de salida debe pertenecer al proyecto.")
    salida = base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida.mkdir(parents=True, exist_ok=False)
    (salida / "auditoria_cobertura.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tabla = []
    for f in filas:
        fila = {k: v for k, v in f.items() if k not in ("resultados_reservados", "etiquetas_originales")}
        fila["etiquetas_originales"] = " / ".join(f["etiquetas_originales"])
        for r, m in f["resultados_reservados"].items():
            fila.update({r + "__" + k: v for k, v in m.items()})
        tabla.append(fila)
    pd.DataFrame(tabla).to_csv(salida / "cobertura_por_etiqueta.csv", index=False, encoding="utf-8-sig")
    (salida / "resumen_cobertura.md").write_text(resumen_md(informe), encoding="utf-8")
    print(f"\nCobertura comprobada: entrenamiento integrado {informe['protocolo']['n_integrado_train']:,}; ajuste {informe['protocolo']['n_ajuste']:,}; validación {informe['protocolo']['n_validacion']:,}.", flush=True)
    print("Etiquetas de ataque reservadas sin ejemplos en entrenamiento integrado:", flush=True)
    for nombres in hallazgos["etiquetas_reservadas_ausentes_del_entrenamiento_integrado"]:
        print("  - " + " / ".join(nombres), flush=True)
    print(f"Resultados guardados en: {salida}", flush=True)
    print("Archivo para revisar: auditoria_cobertura.json", flush=True)
    print("Auditoría terminada. Se conservaron modelos, métricas, reportes y ambos dashboards. No se entrenó ni se consultó Qwen.", flush=True)
    return salida


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resultado-umbrales", type=Path, required=True, help="JSON de la ejecución completada que se auditará.")
    parser.add_argument("--raiz", type=Path, default=Path.cwd(), help="Carpeta SOAR-IA; por defecto, el directorio actual.")
    parser.add_argument("--tamano-bloque", type=int, default=100000)
    args = parser.parse_args(argv)
    if args.tamano_bloque < 1:
        parser.error("El tamaño de bloque debe ser positivo.")
    try:
        ejecutar(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
