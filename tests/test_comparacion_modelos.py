"""Pruebas técnicas; sus datos artificiales no son resultados de la tesis."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import comparar_modelos as c


def crear_fixture(ruta: Path, n: int = 360) -> None:
    rng = np.random.default_rng(17)
    objetivo = np.array(["normal"] * (n // 2) + ["fallido"] * (n // 4) +
                        ["sospechoso"] * (n - n // 2 - n // 4))
    rng.shuffle(objetivo)
    datos = pd.DataFrame({
        "usuario": "network_system", "servicio": rng.choice(["http", "ssh", "otros"], n),
        "hora_evento": 12, "es_horario_laboral": 1, "ip_origen_es_privada": 1,
        "usuario_privilegiado": 0, "host_critico": rng.integers(0, 2, n),
        "conteo_eventos_ip_ventana": rng.integers(0, 18, n),
        "conteo_usuarios_distintos_ip": 1, "conteo_hosts_distintos_ip": 1,
        "conteo_fallidos_ventana": rng.integers(0, 101, n), "servicio_sensible": rng.integers(0, 2, n),
        # Estas columnas peligrosas están presentes para comprobar que se excluyen.
        "tipo_evento": objetivo, "severidad_inicial": objetivo, "hostname": "host-prueba",
        c.OBJETIVO: objetivo,
    })
    datos.to_csv(ruta, index=False)


class MetricasTest(unittest.TestCase):
    def test_filtro_no_confunde_dos_clases_de_ataque_con_omision(self):
        # fallido=0, normal=1, sospechoso=2.
        reales = np.array([0, 0, 2, 2, 1, 1])
        pred = np.array([0, 2, 1, 2, 0, 1])
        r = c.calcular_metricas(reales, pred)
        f = r["filtrado_binario"]
        self.assertEqual((f["verdaderos_positivos"], f["falsos_negativos"],
                          f["falsos_positivos"], f["verdaderos_negativos"]), (3, 1, 1, 1))
        self.assertEqual(f["tasa_falsos_negativos"], 0.25)
        self.assertEqual(f["tasa_falsos_positivos"], 0.5)
        self.assertEqual(f["proporcion_benignos_entre_alertas"], 0.25)
        self.assertEqual(r["accuracy"], 0.5)

    def test_transiciones_pareadas_identifican_el_costo_de_un_cambio(self):
        reales = np.array([0, 0, 2, 2, 1, 1])
        referencia = np.array([1, 0, 1, 2, 0, 1])
        candidata = np.array([0, 1, 2, 2, 1, 0])
        r = c.comparar_predicciones(reales, referencia, candidata)
        self.assertEqual(r["aciertos_recuperados"], 3)
        self.assertEqual(r["errores_introducidos"], 2)
        self.assertEqual(r["ataques_antes_omitidos_ahora_detectados"], 2)
        self.assertEqual(r["ataques_antes_detectados_ahora_omitidos"], 1)
        self.assertEqual(r["falsas_alertas_resueltas"], 1)
        self.assertEqual(r["nuevas_falsas_alertas"], 1)

    def test_no_divide_por_cero_si_no_hay_alertas(self):
        r = c.calcular_metricas(np.array([0, 1, 2]), np.array([1, 1, 1]))
        self.assertIsNone(r["filtrado_binario"]["precision_alertas"])
        self.assertEqual(r["filtrado_binario"]["tasa_falsos_negativos"], 1.0)


class DatosTest(unittest.TestCase):
    def test_excluye_columnas_que_codifican_la_respuesta(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "fixture.csv"
            crear_fixture(ruta)
            X, y, columnas = c.cargar_datos(ruta)
            self.assertEqual(list(X.columns), c.FEATURES)
            self.assertNotIn("severidad_inicial", X)
            self.assertNotIn("tipo_evento", X)
            self.assertIn("severidad_inicial", columnas)
            a, b = train_test_split(np.arange(len(X)), stratify=y, test_size=0.25, random_state=42)
            self.assertEqual(len(set(a) & set(b)), 0)
            audit = c.auditar_representacion(X, y, a, b)
            self.assertIn("usuario", audit["columnas_constantes_en_entrenamiento"])

    def test_rechaza_una_entrada_incompleta(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "incompleto.csv"
            pd.DataFrame({c.OBJETIVO: ["normal"]}).to_csv(ruta, index=False)
            with self.assertRaisesRegex(ValueError, "Faltan columnas"):
                c.cargar_datos(ruta)


@unittest.skipUnless(importlib.util.find_spec("xgboost"), "Requiere XGBoost instalado")
class IntegracionTest(unittest.TestCase):
    def test_ejecuta_los_modelos_sin_modificar_archivos_productivos(self):
        protegidos = [c.RAIZ / "main.py", c.RAIZ / "dashboard.py",
                      c.RAIZ / "src/random_forest.py", c.RAIZ / "data/reportes_generados.json",
                      c.RAIZ / "data/metricas_random_forest.json"]
        antes = {p: c._sha256(p) for p in protegidos if p.exists()}
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "fixture.csv"
            crear_fixture(ruta)
            salida = c.main(["--dataset", str(ruta), "--salida", str(Path(tmp) / "resultados"),
                             "--hilos", "1", "--incluir-rf-alternativo", "--auditar-incidentes",
                             "--sin-figuras", "--verificacion-tecnica"])
            resumen = json.loads((salida / "resultados_completos.json").read_text(encoding="utf-8"))
            protocolo = resumen["protocolo"]
            self.assertEqual(protocolo["estado"], "completado")
            self.assertEqual(protocolo["modo"], "verificacion_tecnica")
            self.assertEqual(protocolo["particion"]["n_test"], 90)
            self.assertEqual(len(resumen["resultados"]), 4)
            rf = resumen["resultados"]["random_forest_actual"]
            self.assertEqual(rf["parametros_efectivos"]["max_depth"], 3)
            self.assertEqual(rf["parametros_efectivos"]["min_samples_leaf"], 500)
            alternativas = resumen["resultados"]["random_forest_alternativo"]
            self.assertEqual(alternativas["parametros_efectivos"]["max_depth"], 12)
            self.assertEqual(resumen["diagnostico"]["incidentes_soar"]["n_incidentes"], 27)
            for resultado in resumen["resultados"].values():
                self.assertEqual(sum(map(sum, resultado["prueba"]["matriz_confusion"]["valores"])), 90)
            pred = pd.read_csv(salida / "predicciones_prueba.csv.gz")
            self.assertEqual(len(pred), 90)
            self.assertEqual(pred["fila_csv_cero_based"].nunique(), 90)
        self.assertEqual(antes, {p: c._sha256(p) for p in antes})


if __name__ == "__main__":
    unittest.main()
