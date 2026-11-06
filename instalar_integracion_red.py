"""Instala la ampliación con comprobación previa y respaldo de los archivos."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import py_compile
import tempfile

PATRONES = ["/data/modelo_random_forest_red.*", "/outputs/integracion_red/",
            "/respaldos_integracion_red/", "/integracion_red_archivos/", "/instalar_integracion_red.py"]


def sha(datos):
    return hashlib.sha256(datos).hexdigest()


def instalar(raiz, comprobar=False):
    raiz=Path(raiz).resolve()
    payload=Path(__file__).resolve().parent/"integracion_red_archivos"
    manifiesto=json.loads((payload/"manifiesto.json").read_text(encoding="utf-8"))
    cambios=[]
    for archivo in manifiesto["archivos"]:
        relativo=Path(archivo["ruta"])
        destino=raiz/relativo
        fuente=payload/relativo
        if relativo.is_absolute() or ".." in relativo.parts or not destino.resolve().is_relative_to(raiz):
            raise ValueError("El manifiesto contiene una ruta fuera del proyecto.")
        nuevos=fuente.read_bytes()
        if sha(nuevos)!=archivo["sha256_nuevo"]:
            raise ValueError(f"El paquete está incompleto o alterado: {relativo}")
        actuales=destino.read_bytes() if destino.is_file() else None
        if actuales==nuevos:continue
        esperado=archivo.get("sha256_anterior")
        if esperado and (actuales is None or sha(actuales)!=esperado):
            raise ValueError(f"{relativo} cambió desde la revisión. No se modificó ningún archivo; comparte su versión actual.")
        if esperado is None and actuales is not None:
            raise ValueError(f"Ya existe otra versión de {relativo}. No se modificó ningún archivo.")
        cambios.append((relativo,nuevos,actuales))
    ignore=raiz/".gitignore"
    old_ignore=ignore.read_bytes() if ignore.is_file() else None
    texto=(old_ignore or b"").decode("utf-8-sig")
    faltan=[p for p in PATRONES if p not in texto.splitlines()]
    if faltan:
        nuevo=(texto.rstrip("\r\n")+"\n\n# Artefactos locales de la ruta RF de red\n"+"\n".join(faltan)+"\n").encode("utf-8")
        cambios.append((Path(".gitignore"),nuevo,old_ignore))
    # Compila antes de escribir; una instalación incompatible se detiene íntegra.
    with tempfile.TemporaryDirectory() as tmp:
        for relativo,nuevos,_ in cambios:
            if relativo.suffix==".py":
                p=Path(tmp)/relativo;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(nuevos)
                py_compile.compile(str(p),doraise=True)
    if comprobar:
        print("Comprobación correcta. Archivos que se instalarían:")
        for p,_,_ in cambios:print(p)
        return None
    if not cambios:
        print("La integración ya está instalada; no se modificó ningún archivo.")
        return None
    fecha=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    respaldo=raiz/"respaldos_integracion_red"/fecha
    respaldo.mkdir(parents=True,exist_ok=False)
    registro=[]
    # Guarda todos los originales antes del primer reemplazo.
    for relativo,_,anteriores in cambios:
        registro.append({"ruta":relativo.as_posix(),"existia":anteriores is not None})
        if anteriores is not None:
            p=respaldo/relativo;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(anteriores)
    (respaldo/"archivos.json").write_text(json.dumps(registro,indent=2),encoding="utf-8")
    escritos=[]
    try:
        for relativo,nuevos,anteriores in cambios:
            p=raiz/relativo;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(nuevos);escritos.append((p,anteriores))
    except Exception:
        for p,anteriores in reversed(escritos):
            if anteriores is None:p.unlink(missing_ok=True)
            else:p.write_bytes(anteriores)
        raise
    print(f"Integración instalada. Respaldo: {respaldo}")
    print("Se conservan dashboard.py, el modelo contextual, sus métricas y los reportes históricos.")
    print("Siguiente paso: ejecutar src.activar_rf_red para guardar y validar el modelo de red.")
    return respaldo


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz",type=Path,default=Path.cwd())
    p.add_argument("--comprobar",action="store_true")
    a=p.parse_args()
    try:instalar(a.raiz,a.comprobar)
    except Exception as e:raise SystemExit(f"ERROR: {e}") from e
