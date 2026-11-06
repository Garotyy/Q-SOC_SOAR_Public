"""Verifica la auditoría con predicciones conocidas, sin entrenar modelos."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import auditar_errores_red as a
from src import comparar_modelos as c


def fixture(d: Path) -> Path:
    exp = d/"experimento"
    exp.mkdir()
    metadatos, capturas = [], []
    offset = 0
    for i, (archivo, etiqueta, clase, n) in enumerate([
        ("Monday-WorkingHours.pcap_ISCX.csv", "BENIGN", "normal", 6),
        ("Tuesday-WorkingHours.pcap_ISCX.csv", "SSH-Patator", "fallido", 4),
        ("Wednesday-workingHours.pcap_ISCX.csv", "DoS Hulk", "sospechoso", 4),
    ]):
        f = pd.DataFrame({v: np.arange(n)+1+i*100 for v in a.VARIABLES_RED})
        f["Destination Port"] = [80, 22, 443][i]
        if i == 0:
            f.loc[3] = f.loc[0]  # Un vector del test repetido en train.
        f["Label"] = etiqueta
        ruta = d/archivo
        f.to_csv(ruta, index=False)
        capturas.append({"archivo": archivo, "sha256": c._sha256(ruta)})
        metadatos.append(pd.DataFrame({a.ID: np.arange(n)+offset, a.ARCHIVO: archivo,
                                       a.FILA_RAW: np.arange(n), "clase": clase}))
        offset += n
    proc = pd.concat(metadatos, ignore_index=True)
    indices = np.array([0, 1, 6, 7, 10, 11, 2])
    real = proc.set_index(a.ID).loc[indices, "clase"].to_numpy()
    pred = pd.DataFrame({"fila_csv_cero_based": indices, "clase_real": real,
                         a.MODELOS[0]: ["sospechoso", "normal", "normal", "fallido", "normal", "sospechoso", "fallido"],
                         a.MODELOS[1]: ["fallido", "normal", "fallido", "normal", "normal", "sospechoso", "normal"]})
    codigos = {c: i for i, c in enumerate(c.CLASES)}
    datos = {"protocolo": {"estado": "completado", "modo": "verificacion_tecnica",
                           "n_train": 7, "n_test": 7, "features_originales": a.VARIABLES_RED,
                           "particion": {"sha256_indices_test": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()},
                           "cohorte": {"archivos": capturas}}, "resultados": {}}
    for modelo in a.MODELOS:
        datos["resultados"][modelo] = {"prueba": c.calcular_metricas(
            pred["clase_real"].map(codigos).to_numpy(), pred[modelo].map(codigos).to_numpy())}
    c._guardar_json(exp/"resultados_representacion.json", datos)
    pred.to_csv(exp/"predicciones_variables_red.csv.gz", index=False, compression="gzip")
    proc.to_csv(exp/"procedencia_muestras.csv.gz", index=False, compression="gzip")
    return exp


class AuditoriaTest(unittest.TestCase):
    def test_separa_filtro_y_confusion_entre_ataques(self):
        real = pd.Series(["normal", "fallido", "fallido", "sospechoso", "normal"])
        pred = pd.Series(["sospechoso", "normal", "sospechoso", "sospechoso", "normal"])
        self.assertEqual(list(a.categorizar(real, pred)),
                         ["falso_positivo", "falso_negativo", "confusion_entre_ataques", "acierto", "acierto"])

    def test_rechaza_predicciones_que_no_reproducen_la_matriz(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = fixture(Path(tmp))
            ref = json.loads((exp/"resultados_representacion.json").read_text(encoding="utf-8"))
            pred = pd.read_csv(exp/"predicciones_variables_red.csv.gz")
            pred.loc[0, a.MODELOS[0]] = "normal"
            with self.assertRaisesRegex(ValueError, "matriz de confusión"):
                a.validar_predicciones(pred, ref["protocolo"], ref["resultados"])

    def test_audita_casos_conocidos_y_preserva_fuentes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            exp = fixture(d)
            originales = list(exp.iterdir())+list(d.glob("*.csv"))
            protegidos = [c.RAIZ/"main.py", c.RAIZ/"dashboard.py", c.RAIZ/"src/ml_predictor.py",
                          c.RAIZ/"src/random_forest.py", c.RAIZ/"src/qwen_agent.py"]
            originales += [p for p in protegidos if p.is_file()]
            antes = {p: c._sha256(p) for p in originales}
            salida = a.main(["--experimento", str(exp), "--data-dir", str(d), "--salida", str(d/"salida"),
                             "--verificacion-tecnica"])
            r = json.loads((salida/"auditoria_errores_red.json").read_text(encoding="utf-8"))
            self.assertEqual(r["solapamiento_errores"], {
                "ataques_omitidos_por_ambos": 1, "ataques_omitidos_solo_RF": 1,
                "ataques_omitidos_solo_XGBoost": 1, "falsos_positivos_de_ambos": 1,
                "falsos_positivos_solo_RF": 1, "falsos_positivos_solo_XGBoost": 0})
            self.assertEqual(r["solapamiento_vectores_train_test"]["registros_test_con_vector_presente_en_train"], 1)
            self.assertEqual(r["modelos"][a.MODELOS[0]]["conteos"]["falso_positivo"], 2)
            tabla = r["modelos"][a.MODELOS[0]]["por_etiqueta_original"]
            self.assertEqual(next(f for f in tabla if f["etiqueta_original"] == "DoS Hulk")["falso_negativo"], 1)
            errores = pd.read_csv(salida/"casos_mal_clasificados.csv.gz")
            self.assertEqual(set(errores[a.ID]), {0, 2, 6, 7, 10})
            self.assertEqual(antes, {p: c._sha256(p) for p in originales})

    def test_rechaza_capturas_alteradas(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            exp = fixture(d)
            ruta = d/"Monday-WorkingHours.pcap_ISCX.csv"
            ruta.write_text(ruta.read_text()+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cambió la captura"):
                a.main(["--experimento", str(exp), "--data-dir", str(d), "--salida", str(d/"salida"),
                        "--verificacion-tecnica"])
            self.assertFalse((d/"salida").exists())


if __name__ == "__main__":
    unittest.main()
