"""Controles técnicos del árbol y la comparación; no son resultados de tesis."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import comparar_modelos as c
from src import comparar_representaciones as r
from src import completar_comparacion_red as completar
from test_comparar_representaciones import crear_capturas


class CompletarTest(unittest.TestCase):
    def test_firmas_usadas_por_el_modelo_son_float32(self):
        X = pd.DataFrame({"medicion": [.1, np.nextafter(.1, np.inf), .2]})
        firmas = completar.firmas_float32(X)
        self.assertEqual(firmas[0], firmas[1])
        self.assertNotEqual(firmas[0], firmas[2])

    def test_subconjuntos_conservan_soportes_y_errores(self):
        real, pred = np.array([0, 1, 2, 2]), np.array([0, 0, 2, 1])
        segmentos = completar.evaluar_subconjuntos(real, pred, np.array([True, True, False, False]))
        visto, nuevo = segmentos.values()
        self.assertEqual(visto["n"], 2)
        self.assertEqual(nuevo["n"], 2)
        self.assertEqual(visto["metricas"]["filtrado_binario"]["falsos_positivos"], 1)
        self.assertEqual(nuevo["metricas"]["filtrado_binario"]["falsos_negativos"], 1)
        self.assertEqual(nuevo["soporte_por_clase"]["fallido"], 0)

    def test_rechaza_orden_de_predicciones_distinto(self):
        guardadas = pd.DataFrame({"fila_csv_cero_based": [2, 1], "clase_real": ["normal", "fallido"],
                                 completar.MODELOS_PREVIOS[0]: ["normal", "fallido"],
                                 completar.MODELOS_PREVIOS[1]: ["normal", "fallido"]})
        with self.assertRaisesRegex(ValueError, "orden"):
            completar.validar_predicciones_previas(guardadas, np.array([1, 2]), np.array([0, 1]), {})

    def test_comparacion_integrada_conserva_datos_modelos_y_parametros(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            dataset = crear_capturas(d)
            protegidos = [c.RAIZ/"main.py", c.RAIZ/"dashboard.py", c.RAIZ/"src/ml_predictor.py",
                          c.RAIZ/"src/random_forest.py", c.RAIZ/"src/modelo_arbol_decision.py",
                          c.RAIZ/"src/qwen_agent.py", c.RAIZ/"data/reportes_generados.json"]
            antes = {p: c._sha256(p) for p in protegidos if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                fase1 = c.main(["--dataset", str(dataset), "--salida", str(d/"fase1"),
                                "--hilos", "1", "--incluir-rf-alternativo", "--sin-figuras", "--verificacion-tecnica"])
                referencia = fase1/"resultados_completos.json"
                fase2 = r.main(["--dataset", str(dataset), "--data-dir", str(d), "--referencia", str(referencia),
                                "--salida", str(d/"fase2"), "--hilos", "1", "--verificacion-tecnica"])
                previos = [referencia, dataset, *fase2.iterdir(), *d.glob("*pcap_ISCX.csv")]
                antes.update({p: c._sha256(p) for p in previos if p.is_file()})
                fase3 = completar.main(["--experimento", str(fase2), "--referencia-inicial", str(referencia),
                                        "--dataset", str(dataset), "--data-dir", str(d), "--salida", str(d/"fase3"),
                                        "--hilos", "1", "--verificacion-tecnica"])
            datos = json.loads((fase3/"resultados_tres_modelos_red.json").read_text(encoding="utf-8"))
            anterior = json.loads((fase2/"resultados_representacion.json").read_text(encoding="utf-8"))
            self.assertEqual(datos["protocolo"]["estado"], "completado")
            self.assertEqual(datos["protocolo"]["modelos_entrenados_en_esta_ejecucion"], ["arbol_decision_red"])
            self.assertEqual(len(datos["resultados"]), 3)
            self.assertIsNone(datos["protocolo"]["seleccion_automatica"])
            arbol = datos["resultados"]["arbol_decision_red"]
            self.assertEqual(arbol["parametros_efectivos"]["max_depth"], 3)
            self.assertEqual(arbol["parametros_efectivos"]["min_samples_leaf"], 20)
            for nombre, modelo in datos["resultados"].items():
                self.assertEqual(sum(s["n"] for s in modelo["subconjuntos_prueba"].values()), 18)
                self.assertEqual(sum(map(sum, modelo["prueba"]["matriz_confusion"]["valores"])), 18)
                if nombre in completar.MODELOS_PREVIOS:
                    original = {k:v for k,v in modelo.items() if k != "subconjuntos_prueba"}
                    self.assertEqual(original, anterior["resultados"][nombre])
            self.assertEqual(antes, {p: c._sha256(p) for p in antes})


if __name__ == "__main__":
    unittest.main()
