"""Prueba un reporte guardado con Qwen; no ejecuta el pipeline ni entrena ML."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from src.qwen_agent import MODELO, VERSION_ANALISIS, analizar_con_qwen3

RAIZ = Path(__file__).resolve().parent


def verificar(reportes: Path, id_flujo: str | None, salida: Path | None = None) -> Path:
    datos_originales = reportes.read_bytes()
    datos = json.loads(datos_originales.decode("utf-8-sig"))
    if not isinstance(datos, list):
        raise ValueError("El archivo de reportes debe contener una lista.")
    candidatos = [r for r in datos if isinstance(r, dict) and (
        r.get("id_alerta") == id_flujo if id_flujo else
        r.get("decision_analisis", {}).get("habilitar_analisis_avanzado")
    )]
    if not candidatos:
        raise ValueError("No se encontró el flujo solicitado o un reporte habilitado.")
    reporte = candidatos[0]
    if not reporte.get("decision_analisis", {}).get("habilitar_analisis_avanzado"):
        raise ValueError("El reporte no está habilitado para análisis avanzado según su política.")
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    carpeta = (salida or RAIZ / "outputs" / "verificacion_qwen") / marca
    carpeta.mkdir(parents=True, exist_ok=False)
    inicio = time.monotonic()
    print(f"Consultando una vez a {MODELO} para {reporte['id_alerta']}...", flush=True)
    try:
        analisis = analizar_con_qwen3(reporte)
    except Exception as error:
        analisis = {"error": "consulta_no_disponible", "detalle": str(error)}
    resultado = {
        "version_analisis": VERSION_ANALISIS,
        "modelo_generativo": MODELO,
        "opciones": {"temperature": 0.0, "seed": 42},
        "id_alerta": reporte["id_alerta"],
        "reportes_origen": str(reportes.resolve()),
        "sha256_reportes_origen": hashlib.sha256(datos_originales).hexdigest(),
        "reporte_evaluado": reporte,
        "tiempo_consulta_segundos": round(time.monotonic() - inicio, 3),
        "estado": "error" if analisis.get("error") else "estructura_y_referencias_validas",
        "alcance_validacion": (
            "Se comprueban estructura, referencias literales de las hipótesis y presencia "
            "de IPv4, CVE e identificadores MITRE, además de límites de severidad, MITRE "
            "e historial y patrones concretos de confianza y horario sin sustento. "
            "Se requiere revisión humana del sentido "
            "de las hipótesis y de todas las conclusiones; no es una prueba de veracidad."
        ),
        "analisis": analisis,
    }
    ruta = carpeta / "verificacion_qwen.json"
    ruta.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Estado:", resultado["estado"])
    print("Hipótesis:", json.dumps(analisis.get("posibles_falsos_positivos"), ensure_ascii=False))
    if analisis.get("error"):
        print("Detalle:", analisis.get("detalle", analisis["error"]))
    print("Resultado guardado en:", ruta)
    print("Se conservan los reportes de origen, los modelos y las métricas de ML.")
    return ruta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reportes", type=Path)
    p.add_argument("--id", dest="id_flujo")
    a = p.parse_args()
    ruta = a.reportes
    if ruta is None:
        rutas = sorted((RAIZ / "outputs" / "integracion_red").glob("*/reportes_red.json"))
        if not rutas:
            p.error("No hay reportes de red guardados; especifica --reportes.")
        ruta = rutas[-1]
    try:
        destino = verificar(ruta, a.id_flujo)
        return 1 if json.loads(destino.read_text(encoding="utf-8"))["estado"] == "error" else 0
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
