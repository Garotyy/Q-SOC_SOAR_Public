"""Comprueba identidad de cohorte y evita comparar pruebas diferentes."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import comparar_modelos as c
from src import comparar_representaciones as r


def crear_capturas(directorio: Path) -> Path:
    frames = {}
    for i, (nombre, etiqueta, puerto) in enumerate([
        ("Monday-WorkingHours.pcap_ISCX.csv", "BENIGN", 80),
        ("Tuesday-WorkingHours.pcap_ISCX.csv", "SSH-Patator", 22),
        ("Wednesday-workingHours.pcap_ISCX.csv", "DoS Hulk", 443),
    ]):
        n = 24
        f = pd.DataFrame({campo: np.arange(n) + 1 + i for campo in r.VARIABLES_RED})
        f["Destination Port"] = puerto
        f["Flow Duration"] = 100 * (np.arange(n) + 1 + i)
        f["Bwd Packets/s"] = np.arange(n) + i * 20
        f["Label"] = etiqueta
        f["campo_adicional_fuera_de_features"] = 5.0
        if i == 0:
            f.loc[3, "campo_adicional_fuera_de_features"] = np.nan
        f.to_csv(directorio/nombre, index=False)
        frames[nombre] = f
    # Orden deliberadamente diferente de la enumeración por nombre.
    orden = ["Tuesday-WorkingHours.pcap_ISCX.csv", "Monday-WorkingHours.pcap_ISCX.csv",
             "Wednesday-workingHours.pcap_ISCX.csv"]
    hom = pd.concat([r.homologar_para_verificar(frames[n].dropna()) for n in orden], ignore_index=True)
    ruta = directorio/"dataset_kali_homologado.csv"
    hom.to_csv(ruta, index=False)
    return ruta


class RepresentacionesTest(unittest.TestCase):
    def test_verifica_la_semantica_del_homologador_con_un_caso_conocido(self):
        f = pd.DataFrame({"Destination Port": [22], "Flow Duration": [99],
                          "Bwd Packets/s": [105.9], "Label": ["SSH-Patator"]})
        h = r.homologar_para_verificar(f).iloc[0]
        self.assertEqual(h["servicio"], "ssh")
        self.assertEqual(h[c.OBJETIVO], "fallido")
        self.assertEqual(h["conteo_eventos_ip_ventana"], 4)
        self.assertEqual(h["conteo_fallidos_ventana"], 100)
        self.assertEqual(h["host_critico"], 1)

    def test_encuentra_el_orden_y_conserva_limpieza_sobre_todas_las_columnas(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ruta = crear_capturas(d)
            ref = pd.read_csv(ruta)
            X, y, meta = r.reconstruir_cohorte(d, ref, d)
            self.assertEqual(len(X), 71)  # Excluye el NaN de una columna no seleccionada.
            self.assertEqual(list(X.columns), r.VARIABLES_RED)
            self.assertEqual(meta["orden_capturas"][0], "Tuesday-WorkingHours.pcap_ISCX.csv")
            self.assertTrue(np.array_equal(y.to_numpy(), ref[c.OBJETIVO].to_numpy()))
            self.assertEqual(y.iloc[0], "fallido")

    def test_detiene_el_experimento_si_una_etiqueta_no_coincide(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ruta = crear_capturas(d)
            ref = pd.read_csv(ruta)
            ref.loc[0, c.OBJETIVO] = "normal"
            with self.assertRaisesRegex(ValueError, "no coinciden fila por fila"):
                r.reconstruir_cohorte(d, ref, d)
            self.assertTrue((d/"cohorte_no_coincide.json").is_file())

    def test_ejecucion_integrada_con_datos_tecnicos(self):
        protegidos = [c.RAIZ/"main.py", c.RAIZ/"dashboard.py", c.RAIZ/"src/random_forest.py",
                      c.RAIZ/"data/reportes_generados.json", c.RAIZ/"data/metricas_random_forest.json"]
        antes = {p: c._sha256(p) for p in protegidos if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ruta = crear_capturas(d)
            primera = c.main(["--dataset", str(ruta), "--salida", str(d/"fase1"),
                              "--hilos", "1", "--incluir-rf-alternativo", "--sin-figuras",
                              "--verificacion-tecnica"])
            segunda = r.main(["--referencia", str(primera/"resultados_completos.json"),
                              "--dataset", str(ruta), "--data-dir", str(d),
                              "--salida", str(d/"fase2"), "--hilos", "1",
                              "--verificacion-tecnica"])
            datos = json.loads((segunda/"resultados_representacion.json").read_text(encoding="utf-8"))
            self.assertEqual(datos["protocolo"]["estado"], "completado")
            self.assertEqual(datos["protocolo"]["n_test"], 18)
            self.assertEqual(len(datos["resultados"]), 4)
            for nombre in ["random_forest_alternativo_red", "xgboost_red"]:
                self.assertEqual(sum(map(sum, datos["resultados"][nombre]["prueba"]["matriz_confusion"]["valores"])), 18)
        self.assertEqual(antes, {p: c._sha256(p) for p in antes})


if __name__ == "__main__":
    unittest.main()
