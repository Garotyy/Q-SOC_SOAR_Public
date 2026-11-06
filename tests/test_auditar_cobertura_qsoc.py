"""Casos técnicos pequeños; no son datos ni resultados de la tesis."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

RUTA = Path(__file__).resolve().parents[1] / "auditar_cobertura_qsoc.py"
spec = importlib.util.spec_from_file_location("auditoria_cobertura", RUTA)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def guardar(ruta, dato):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(dato, ensure_ascii=False), encoding="utf-8")


def huella_indices(indices):
    import hashlib
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def medir(n, etiqueta):
    benigno = a.clase_qsoc(etiqueta) == "normal"
    return {"n": n, "verdaderos_positivos": 0, "falsos_negativos": 0 if benigno else n,
            "falsos_positivos": 0, "verdaderos_negativos": n if benigno else 0}


def crear_proyecto(raiz):
    (raiz / "data").mkdir()
    (raiz / "src").mkdir()
    for nombre in ["main.py", "dashboard.py", "dashboard_red.py", "src/qwen_agent.py"]:
        (raiz / nombre).write_text("# Archivo protegido\n", encoding="utf-8")
    (raiz / "data/modelo_random_forest.joblib").write_bytes(b"No se debe cargar ni ejecutar este archivo.")
    (raiz / "data/modelo_random_forest_red.joblib").write_bytes(b"Modelo nativo protegido.")
    guardar(raiz / "data/reportes_generados.json", {"historico": True})
    guardar(raiz / "data/metricas_random_forest.json", {"historico": True})
    hom = raiz / "data/dataset_kali_homologado.csv"
    hom.write_text("fixture_tecnica\n", encoding="utf-8")
    etiquetas = {
        "Monday-WorkingHours.pcap_ISCX.csv": ["BENIGN"] * 5,
        "Tuesday-WorkingHours.pcap_ISCX.csv": ["BENIGN", "FTP-Patator", "SSH-Patator", "BENIGN", "FTP-Patator", "BENIGN"],
        "Wednesday-workingHours.pcap_ISCX.csv": ["DoS Hulk", "BENIGN", "DoS GoldenEye", "Heartbleed", "BENIGN", "DoS Hulk", "BENIGN"],
    }
    procedencia = []
    infos = []
    # La fila Monday 1 se excluyó por la limpieza previa sobre otra columna.
    # Contar Label sin usar procedencia la incluiría erróneamente.
    for nombre, labels in etiquetas.items():
        frame = pd.DataFrame({"Label": labels, "auxiliar": [0.0] * len(labels)})
        if nombre.startswith("Monday"):
            frame.loc[1, "auxiliar"] = np.nan
        ruta = raiz / "data" / nombre
        frame.to_csv(ruta, index=False)
        limpias = frame.dropna()
        for i in limpias.index:
            procedencia.append({a.COLUMNA_ID: len(procedencia), "archivo_captura": nombre,
                                a.COLUMNA_FILA: int(i), "clase": a.clase_qsoc(labels[i])})
        infos.append({"archivo": nombre, "sha256": a.sha256(ruta), "registros_iniciales": len(frame), "registros_validos": len(limpias)})
    meta = pd.DataFrame(procedencia)
    n = len(meta)
    ajuste = np.array([0, 2, 4, 5, 6, 8, 10, 11, 13, 15], dtype=np.int64)
    validacion = np.array([1, 7, 12, 16], dtype=np.int64)
    test = np.array([3, 9, 14], dtype=np.int64)
    train = np.concatenate([validacion, ajuste])
    assert n == len(train) + len(test)
    detalle = {f: {"n": len(i), "sha256_indices": huella_indices(i),
                    "clases": {str(k): int(v) for k, v in meta.iloc[i]["clase"].value_counts().items()}}
               for f, i in zip(a.FASES, [ajuste, validacion, test])}
    primera = raiz / "outputs/comparacion_modelos/primera/resultados_completos.json"
    guardar(primera, {"protocolo": {"dataset": {"total": n, "sha256": a.sha256(hom)},
                                    "particion": {"sha256_indices_train": huella_indices(train), "sha256_indices_test": huella_indices(test)}}})
    np.savez_compressed(primera.parent / "indices_particion.npz", train=train, test=test)
    final = raiz / "outputs/comparacion_final_red/final/resultados_tres_modelos_red.json"
    guardar(final, {"protocolo": {"referencia_inicial_sha256": a.sha256(primera), "cohorte": {"archivos": infos}}})
    salida = raiz / "outputs/prueba_umbrales/terminada"
    salida.mkdir(parents=True)
    meta.to_csv(salida / "procedencia_muestras.csv.gz", index=False, compression="gzip")
    fases = pd.DataFrame({a.COLUMNA_ID: np.concatenate([ajuste, validacion, test]),
                          "rol": np.repeat(a.FASES, [len(ajuste), len(validacion), len(test)])})
    fases.to_csv(salida / "particiones.csv.gz", index=False, compression="gzip")
    grupos = []
    for etiqueta, n_f in [("BENIGN", 9), ("PortScan", 4), ("DoS Hulk", 3), ("Bot", 2)]:
        grupos.append({"etiqueta_original": etiqueta, "clase_qsoc": a.clase_qsoc(etiqueta), "n": n_f,
                       **{r: medir(n_f, etiqueta) for r in a.REGLAS}})
    totales = {r: {k: sum(g[r][k] for g in grupos) for k in a.MEDIDAS} for r in a.REGLAS}
    reserved = {"agregado": totales, "por_captura": [{"por_etiqueta_original": grupos, **totales}]}
    resultado = {"protocolo": {"estado": "completado", "modo": "umbral_validacion_y_capturas_reservadas",
                               "referencia_tres_modelos_sha256": a.sha256(final), "referencia_inicial_sha256": a.sha256(primera),
                               "particiones": detalle}, "evaluacion_capturas_reservadas": reserved}
    ruta = salida / "resultados_umbrales.json"
    guardar(ruta, resultado)
    return argparse.Namespace(raiz=raiz, resultado_umbrales=ruta, tamano_bloque=2)


class AuditoriaCoberturaTest(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        self.args = crear_proyecto(self.raiz)

    def correr(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return a.ejecutar(self.args)

    def test_integral_conserva_fuentes_y_respeta_filas_y_roles_guardados(self):
        antes = a.protegidos(self.raiz)
        salida = self.correr()
        self.assertEqual(a.protegidos(self.raiz), antes)
        informe = a.leer_json(salida / "auditoria_cobertura.json")
        self.assertEqual(informe["protocolo"]["n_integrado_train"], 14)
        self.assertEqual(informe["protocolo"]["n_ajuste"], 10)
        self.assertEqual(informe["protocolo"]["n_validacion"], 4)
        self.assertEqual(informe["protocolo"]["n_test_historico"], 3)
        lunes = informe["capturas_desarrollo"][0]
        self.assertEqual(lunes["registros_originales"], 5)
        self.assertEqual(lunes["registros_conservados"], 4)
        self.assertEqual(lunes["etiquetas_descartadas"], {"BENIGN": 1})
        por_nombre = {f["etiqueta_canonica"]: f for f in informe["cobertura_por_etiqueta"]}
        self.assertGreater(por_nombre["FTP-PATATOR"]["entrenamiento_integrado"], 0)
        self.assertEqual(por_nombre["PORTSCAN"]["clase_qsoc"], "fallido")
        self.assertTrue(por_nombre["PORTSCAN"]["ausente_del_entrenamiento_integrado"])
        self.assertFalse(por_nombre["DOS HULK"]["ausente_del_entrenamiento_integrado"])
        self.assertTrue((salida / "resumen_cobertura.md").is_file())
        self.assertTrue((salida / "cobertura_por_etiqueta.csv").is_file())

    def test_etiquetas_ausentes_no_confunden_clase_coarse_o_benignos(self):
        salida = self.correr()
        hallazgos = a.leer_json(salida / "auditoria_cobertura.json")["hallazgos_cobertura"]
        self.assertEqual(hallazgos["etiquetas_reservadas_ausentes_del_entrenamiento_integrado"], [["Bot"], ["PortScan"]])
        self.assertEqual(hallazgos["flujos_ataque_reservados_de_etiquetas_ausentes"], 6)
        self.assertEqual(hallazgos["flujos_ataque_reservados_totales"], 9)

    def test_rechaza_particiones_que_no_reproducen_huella(self):
        p = self.args.resultado_umbrales.parent / "particiones.csv.gz"
        df = pd.read_csv(p)
        df = pd.concat([df.iloc[1:2], df.iloc[0:1], df.iloc[2:]], ignore_index=True)
        df.to_csv(p, index=False, compression="gzip")
        with self.assertRaisesRegex(ValueError, "huella"):
            self.correr()
        self.assertFalse((self.raiz / "outputs/auditoria_cobertura").exists())

    def test_rechaza_procedencia_con_filas_repetidas(self):
        p = self.args.resultado_umbrales.parent / "procedencia_muestras.csv.gz"
        df = pd.read_csv(p)
        df.loc[1, a.COLUMNA_FILA] = df.loc[0, a.COLUMNA_FILA]
        df.to_csv(p, index=False, compression="gzip")
        with self.assertRaisesRegex(ValueError, "repetidas"):
            self.correr()

    def test_rechaza_clases_cambiadas_aunque_los_totales_sean_iguales(self):
        p = self.args.resultado_umbrales.parent / "procedencia_muestras.csv.gz"
        df = pd.read_csv(p)
        # Dos casos de ajuste: BENIGN (fila cohorte 4) y FTP-Patator (5).
        df.loc[[4, 5], "clase"] = df.loc[[5, 4], "clase"].to_numpy()
        df.to_csv(p, index=False, compression="gzip")
        with self.assertRaisesRegex(ValueError, "contradicen"):
            self.correr()

    def test_rechaza_captura_original_modificada(self):
        p = self.raiz / "data/Tuesday-WorkingHours.pcap_ISCX.csv"
        p.write_text(p.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cambió la captura"):
            self.correr()

    def test_rechaza_conteos_reservados_inconsistentes(self):
        r = a.leer_json(self.args.resultado_umbrales)
        r["evaluacion_capturas_reservadas"]["por_captura"][0]["base_experimental"]["n"] += 1
        guardar(self.args.resultado_umbrales, r)
        with self.assertRaisesRegex(ValueError, "no suma"):
            self.correr()

    def test_repeticion_crea_otra_salida_y_conserva_la_anterior(self):
        una = self.correr()
        huella = a.sha256(una / "auditoria_cobertura.json")
        otra = self.correr()
        self.assertNotEqual(una, otra)
        self.assertEqual(a.sha256(una / "auditoria_cobertura.json"), huella)

    def test_campo_sustitucion_se_conserva_y_senala(self):
        desarrollo = [{"por_fase": {f: {} for f in a.FASES}}]
        etiqueta = "Web Attack \ufffd XSS"
        grupos = [{"etiqueta_original": etiqueta, "n": 2, "clase_qsoc": "sospechoso", **{r: medir(2, etiqueta) for r in a.REGLAS}}]
        totales = {r: grupos[0][r] for r in a.REGLAS}
        filas, _ = a.consolidar_cobertura(desarrollo, {"agregado": totales, "por_captura": [{"por_etiqueta_original": grupos, **totales}]})
        self.assertTrue(filas[0]["hay_caracter_de_sustitucion_en_etiqueta"])
        self.assertEqual(filas[0]["etiquetas_originales"], [etiqueta])


if __name__ == "__main__":
    unittest.main()
