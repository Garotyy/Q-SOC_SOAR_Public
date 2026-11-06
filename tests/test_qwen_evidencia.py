"""Regresiones técnicas de la capa generativa; no son métricas de CICIDS2017."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import qwen_agent as q
import verificar_qwen


def reporte_red():
    return {
        "id_alerta": "FLOW-000000049",
        "tipo_incidente": "Network Flow Observed",
        "severidad": "No determinada",
        "prediccion_ml": {"clase_predicha": "sospechoso", "confianza": 0.98},
        "telemetria_red": {"Destination Port": 80, "Total Fwd Packets": 10},
        "evidencia": [
            "Flujo de red recibido con mediciones de telemetría.",
            "Procedencia: Wednesday-workingHours.pcap_ISCX.csv; fila 116843.",
            "La clase ML no demuestra por sí sola una técnica MITRE ni fallos de autenticación.",
        ],
        "contexto_historico": {
            "incidentes_relacionados_previos": 0,
            "resumen_contexto": "No se identifican incidentes previos relacionados en esta ejecución.",
        },
        "decision_analisis": {"habilitar_analisis_avanzado": True},
    }


def respuesta(hipotesis=None):
    contenido = {
        "resumen_ejecutivo": "Flujo clasificado como sospechoso; requiere revisión.",
        "hechos_observados": ["El puerto de destino es 80."],
        "inferencias_razonables": ["Las mediciones por sí solas no confirman un ataque."],
        "analisis_riesgo": "La severidad no está determinada en el reporte.",
        "explicacion_mitre": "No disponible en el reporte",
        "uso_contexto_historico": "No hay incidentes previos relacionados en esta ejecución.",
        "recomendacion_soc": ["Corroborar el flujo con otras fuentes."],
        "posibles_falsos_positivos": [] if hipotesis is None else hipotesis,
        "siguiente_accion_sugerida": "Revisar las mediciones del flujo.",
    }
    contenido.update(q._limites_del_reporte(reporte_red()))
    return contenido


def consultar(reporte, contenido, **extra):
    sesion = Mock()
    http = Mock()
    http.json.return_value = {"response": json.dumps(contenido, ensure_ascii=False), "done": True, **extra}
    sesion.post.return_value = http
    with patch.object(q, "requests", sesion):
        resultado = q.analizar_con_qwen3(reporte)
    return resultado, sesion


class EvidenciaQwenTests(unittest.TestCase):
    def test_flujo_sin_contexto_legitimo_no_fuerza_hipotesis(self):
        r = reporte_red()
        original = copy.deepcopy(r)
        resultado, sesion = consultar(r, respuesta())
        self.assertEqual(resultado["posibles_falsos_positivos"], [q.SIN_HIPOTESIS])
        self.assertEqual(set(resultado), set(q.CLAVES_RESPUESTA_QWEN3))
        self.assertEqual(r, original)
        payload = sesion.post.call_args.kwargs["json"]
        self.assertEqual(payload["format"]["properties"]["posibles_falsos_positivos"]["maxItems"], 0)
        self.assertEqual(payload["options"], {"temperature": 0.0, "seed": 42})
        self.assertIn('"puntuacion_ml_no_calibrada": 0.98', payload["prompt"])
        self.assertNotIn('"confianza": 0.98', payload["prompt"])
        self.assertEqual(sesion.post.call_args.kwargs["timeout"], 180)
        sesion.post.assert_called_once()

    def test_contexto_autorizado_con_cita_literal(self):
        r = reporte_red()
        dato = "El proceso de backup del servidor está autorizado para esta ventana."
        r["evidencia"].append(dato)
        h = {"hipotesis": "El flujo podría corresponder al backup autorizado descrito en el reporte.",
             "campo": "/evidencia/3", "evidencia": dato}
        salida, _ = consultar(r, respuesta([h]))
        self.assertNotIn("error", salida)
        self.assertIn(dato, salida["posibles_falsos_positivos"][0])
        self.assertIn("/evidencia/3", salida["posibles_falsos_positivos"][0])

    def test_rechaza_frases_copiadas_de_la_captura(self):
        salida, _ = consultar(reporte_red(), respuesta([
            "Un login exitoso solo puede considerarse legítimo si el reporte contiene señales de que el acceso estaba autorizado."
        ]))
        self.assertEqual(salida["error"], "estructura_json_invalida")
        self.assertIn("respuesta_original", salida)

    def test_rechaza_fuente_inventada_aunque_texto_suene_plausible(self):
        h = {"hipotesis": "Puede tratarse de un backup autorizado.",
             "campo": "/evidencia/99", "evidencia": "Backup autorizado."}
        salida, _ = consultar(reporte_red(), respuesta([h]))
        self.assertEqual(salida["error"], "hipotesis_sin_sustento")

    def test_rechaza_cita_modificada_que_elimina_una_negacion(self):
        r = reporte_red()
        r["evidencia"].append("El proceso de backup no estaba autorizado.")
        h = {"hipotesis": "El flujo podría ser un backup autorizado.",
             "campo": "/evidencia/3", "evidencia": "El proceso de backup estaba autorizado."}
        salida, _ = consultar(r, respuesta([h]))
        self.assertEqual(salida["error"], "hipotesis_sin_sustento")

    def test_rechaza_metadatos_y_prediccion_como_justificacion(self):
        for campo, dato in (("/telemetria_red/Destination Port", "80"),
                            ("/prediccion_ml/clase_predicha", "sospechoso")):
            with self.subTest(campo=campo):
                h = {"hipotesis": "El flujo podría corresponder a administración autorizada.",
                     "campo": campo, "evidencia": dato}
                salida, _ = consultar(reporte_red(), respuesta([h]))
                self.assertEqual(salida["error"], "hipotesis_sin_sustento")

    def test_rechaza_regla_pegada_incluso_con_cita_existente(self):
        r = reporte_red()
        r["evidencia"].append("Se observa una conexión al servidor.")
        h = {"hipotesis": "Un login exitoso solo puede considerarse legítimo si está autorizado.",
             "campo": "/evidencia/3", "evidencia": r["evidencia"][3]}
        salida, _ = consultar(r, respuesta([h]))
        self.assertEqual(salida["error"], "instrucciones_copiadas")

    def test_rechaza_ip_cve_y_mitre_no_aportados(self):
        for dato in ("203.0.113.254", "CVE-2026-12345", "T1059.001"):
            with self.subTest(dato=dato):
                contenido = respuesta()
                contenido["resumen_ejecutivo"] += " Indicador: " + dato
                salida, _ = consultar(reporte_red(), contenido)
                self.assertEqual(salida["error"], "indicador_no_presente_en_reporte")

    def test_preserva_identificadores_que_si_estan_en_el_reporte(self):
        r = reporte_red()
        r["evidencia"].append("IP origen 203.0.113.45; técnica T1110.001.")
        contenido = respuesta()
        contenido["hechos_observados"] = [r["evidencia"][-1]]
        salida, _ = consultar(r, contenido)
        self.assertNotIn("error", salida)

    def test_no_reutiliza_analisis_anteriores(self):
        r = reporte_red()
        r["analisis_qwen3_real"] = {"texto": "CITA_ANTERIOR_INVENTADA"}
        r["analisis_qwen3_placeholder"] = {"texto": "PLACEHOLDER_NO_ES_EVIDENCIA"}
        _, sesion = consultar(r, respuesta())
        payload = sesion.post.call_args.kwargs["json"]
        self.assertNotIn("CITA_ANTERIOR_INVENTADA", payload["prompt"])
        self.assertNotIn("PLACEHOLDER_NO_ES_EVIDENCIA", payload["prompt"])

    def test_respuesta_truncada_no_se_acepta_ni_reintenta(self):
        salida, sesion = consultar(reporte_red(), respuesta(), done_reason="length")
        self.assertEqual(salida["error"], "respuesta_incompleta")
        sesion.post.assert_called_once()

    def test_estructura_publica_y_parser_historico_se_conservan(self):
        contenido = respuesta()
        contenido["posibles_falsos_positivos"] = [q.SIN_HIPOTESIS]
        self.assertEqual(q._parsear_respuesta_json(json.dumps(contenido)), contenido)
        salida, _ = consultar(reporte_red(), {**respuesta(), "clave_extra": "dato"})
        self.assertEqual(salida["error"], "estructura_json_invalida")

    def test_rechaza_riesgo_moderado_sin_severidad_en_el_reporte(self):
        contenido = respuesta()
        contenido["analisis_riesgo"] = "Moderado"
        salida, _ = consultar(reporte_red(), contenido)
        self.assertEqual(salida["error"], "limite_del_reporte_no_respetado")
        self.assertIn("Moderado", salida["respuesta_original"])

    def test_mitre_ausente_exige_reconocer_que_no_esta_asignado(self):
        contenido = respuesta()
        contenido["explicacion_mitre"] = "El flujo podría ser sospechoso según ML."
        salida, _ = consultar(reporte_red(), contenido)
        self.assertEqual(salida["error"], "limite_del_reporte_no_respetado")

    def test_prediccion_no_se_presenta_como_alta_confianza(self):
        for frase in ("El modelo clasificó el flujo como sospechoso con una alta confianza.",
                      "El resultado presenta alta confianza."):
            with self.subTest(frase=frase):
                contenido = respuesta()
                contenido["inferencias_razonables"] = [frase]
                salida, _ = consultar(reporte_red(), contenido)
                self.assertEqual(salida["error"], "confianza_cualitativa_no_validada")

    def test_workinghours_no_demuestra_horario_laboral(self):
        contenido = respuesta()
        contenido["inferencias_razonables"] = ["El flujo proviene de un horario de trabajo específico."]
        salida, _ = consultar(reporte_red(), contenido)
        self.assertEqual(salida["error"], "inferencia_temporal_no_sustentada")

    def test_sin_ip_no_se_afirma_ausencia_global_de_recurrencia(self):
        contenido = respuesta()
        contenido["uso_contexto_historico"] = "No se identificaron patrones recurrentes de actividad sospechosa."
        salida, _ = consultar(reporte_red(), contenido)
        self.assertEqual(salida["error"], "limite_del_reporte_no_respetado")
        limites = q._limites_del_reporte(reporte_red())
        self.assertIn("no aporta IP de origen", limites["uso_contexto_historico"])

    def test_no_fuerza_ausencias_si_los_campos_si_estan_aportados(self):
        r = reporte_red()
        r["severidad"] = "Alta"
        r["id_técnica"] = "T1110"
        r["timestamp"] = "2026-11-05T10:00:00"
        limites = q._limites_del_reporte(r)
        self.assertNotIn("analisis_riesgo", limites)
        self.assertNotIn("explicacion_mitre", limites)
        self.assertTrue(q._hay_contexto_temporal(r))

    def test_limita_inferencias_para_evitar_relleno(self):
        contenido = respuesta()
        contenido["inferencias_razonables"] = ["Se requiere corroboración."] * 8
        salida, _ = consultar(reporte_red(), contenido)
        self.assertEqual(salida["error"], "inferencias_excesivas")

    def test_verificador_guarda_resultado_sin_alterar_reportes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "reportes_red.json"
            ruta.write_text(json.dumps([reporte_red()]), encoding="utf-8")
            antes = ruta.read_bytes()
            salida = respuesta()
            salida["posibles_falsos_positivos"] = [q.SIN_HIPOTESIS]
            with patch.object(verificar_qwen, "analizar_con_qwen3", return_value=salida) as llamada, contextlib.redirect_stdout(io.StringIO()):
                guardado = verificar_qwen.verificar(ruta, "FLOW-000000049", base / "salidas")
            self.assertEqual(ruta.read_bytes(), antes)
            self.assertEqual(json.loads(guardado.read_text(encoding="utf-8"))["analisis"], salida)
            self.assertEqual(json.loads(guardado.read_text(encoding="utf-8"))["reporte_evaluado"], reporte_red())
            llamada.assert_called_once()

    def test_verificador_respeta_omision_de_flujos_normales(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = reporte_red()
            r["decision_analisis"]["habilitar_analisis_avanzado"] = False
            ruta = Path(tmp) / "reportes.json"
            ruta.write_text(json.dumps([r]), encoding="utf-8")
            with patch.object(verificar_qwen, "analizar_con_qwen3") as llamada:
                with self.assertRaises(ValueError):
                    verificar_qwen.verificar(ruta, r["id_alerta"], Path(tmp) / "salidas")
            llamada.assert_not_called()


if __name__ == "__main__":
    unittest.main()
