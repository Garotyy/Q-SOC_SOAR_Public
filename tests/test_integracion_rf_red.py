"""Pruebas técnicas: contrato, rutas, reglas y persistencia. Sin llamadas al LLM."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import main as pipeline
from src import ml_predictor
from src import comparar_modelos as c
from src import comparar_representaciones as rep
from src import completar_comparacion_red as completar
from src import activar_rf_red as activar
from src.entrada_red import ejecutar_tabla
from src.ml_predictor_red import predecir_tabla_red, predecir_incidente_red
from src.politica_red import decidir_revision
from src.telemetria_red import (VARIABLES_RED, CLASES_RED, FORMATO_MODELO_RED,
                               crear_alerta_flujo, anexar_telemetria, matriz_red)
from test_comparar_representaciones import crear_capturas
import instalar_integracion_red as instalador


def modelo_tecnico(d):
    X = pd.DataFrame({c:np.arange(36)+1 for c in VARIABLES_RED})
    X["Destination Port"] = np.tile([22,80,443],12)
    y = np.tile([0,1,2],12)
    Z, validas = matriz_red(X)
    assert validas.all()
    modelo = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42, n_jobs=1).fit(Z,y)
    ruta = d/"modelo_red.joblib"
    joblib.dump({"formato":FORMATO_MODELO_RED, "features":VARIABLES_RED, "clases":CLASES_RED,
                 "modelo":modelo, "modo":"verificacion_tecnica",
                 "evaluacion_referencia":c.calcular_metricas(y,modelo.predict(Z))},ruta)
    return ruta,X


class IntegracionTest(unittest.TestCase):
    def test_etiqueta_y_metadatos_no_intervienen_en_la_prediccion(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta,X=modelo_tecnico(Path(tmp))
            uno=predecir_tabla_red(X.assign(Label="BENIGN",usuario="root"),ruta)
            dos=predecir_tabla_red(X.assign(Label="DoS Hulk",usuario="otro"),ruta)
            pd.testing.assert_frame_equal(uno,dos)

    def test_mediciones_incompletas_no_se_convierten_en_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta,X=modelo_tecnico(Path(tmp))
            invalido=X.iloc[0].to_dict(); invalido.pop("Flow Bytes/s")
            pred=predecir_incidente_red({"telemetria_red":invalido},ruta)
            self.assertEqual(pred["estado"],"error")
            self.assertTrue(decidir_revision(pred,"Baja",{})["requiere_revision"])
            salida=predecir_tabla_red([X.iloc[1].to_dict(),invalido],ruta)
            self.assertEqual(list(salida["estado"]),["ok","error"])

    def test_un_flujo_invalido_impide_filtrar_un_incidente_agregado(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta,X=modelo_tecnico(Path(tmp))
            pred=predecir_incidente_red({"flujos_red":[X.iloc[1].to_dict(),{}]},ruta)
            self.assertEqual(pred["estado"],"error")

    def test_enteros_anulables_y_puertos_invalidos_quedan_para_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta,X=modelo_tecnico(Path(tmp))
            X=X.astype("Int64")
            X.loc[0,"Flow Bytes/s"]=pd.NA
            X.loc[1,"Destination Port"]=-1
            salida=predecir_tabla_red(X,ruta)
            self.assertEqual(list(salida["estado"].iloc[:3]),["error","error","ok"])

    def test_normal_respeta_critica_y_reglas_independientes(self):
        pred={"estado":"ok","clase_predicha":"normal"}
        self.assertTrue(decidir_revision(pred,"Crítica",{})["requiere_revision"])
        self.assertTrue(decidir_revision(pred,"Baja",{"requiere_escalamiento":True})["requiere_revision"])
        self.assertFalse(decidir_revision(pred,"No determinada",{})["requiere_revision"])

    def test_ruta_contextual_se_conserva_sin_mediciones(self):
        class Stub:
            named_steps={"modelo":type("Clases",(),{"classes_":CLASES_RED})()}
            def predict(self,datos):return np.array(["normal"])
            def predict_proba(self,datos):return np.array([[0.,1.,0.]])
        with patch.object(ml_predictor,"_obtener_pipeline_entrenado",return_value=Stub()), \
             patch("src.ml_predictor_red.predecir_incidente_red") as nuevo:
            resultado=ml_predictor.predecir_incidente_ml({"servicio":"ssh","usuario":"usuario"})
        self.assertEqual(resultado["estado"],"ok")
        self.assertEqual(resultado["representacion"],"contextual_homologada")
        nuevo.assert_not_called()

    def test_conserva_flujos_por_evento_sin_promediar(self):
        incidentes=[{"eventos_relacionados":["a","b"]}]
        alertas=[{"id_alerta":"a","telemetria_red":{"Flow Duration":3}},
                 {"id_alerta":"b","telemetria_red":{"Flow Duration":9}}]
        anexar_telemetria(incidentes,alertas)
        self.assertEqual([f["Flow Duration"] for f in incidentes[0]["flujos_red"]],[3,9])

    def test_ssh_contextual_critico_normal_no_omite_el_respaldo(self):
        alertas=[{"id_alerta":f"a{i}","timestamp":"2026-01-01T12:00:00Z","servicio":"ssh",
                  "tipo_evento":"ssh_authentication","resultado":"fallido","ip_origen":"192.0.2.1",
                  "ip_destino":"192.0.2.2","hostname":"host","usuario":["admin","root"][i%2]} for i in range(6)]
        pred={"estado":"ok","clase_predicha":"normal","representacion":"contextual_homologada"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline,"predecir_incidente_ml",return_value=pred), \
             patch.object(pipeline,"_intentar_analisis_qwen3_real") as llm, contextlib.redirect_stdout(io.StringIO()):
            reportes=pipeline.ejecutar_pipeline(False,alertas=alertas,ruta_reportes=Path(tmp)/"r.json",usar_qwen=False)
        llm.assert_not_called()
        self.assertEqual(reportes[0]["severidad"],"Crítica")
        self.assertTrue(reportes[0]["requiere_escalamiento"])
        self.assertEqual(reportes[0]["analisis_qwen3_real"]["estado"],"pendiente_bajo_demanda")

    def test_entrada_usa_pipeline_real_y_senaliza_calidad_sin_inventar_contexto(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); ruta,X=modelo_tecnico(d)
            X.loc[0,"Flow Bytes/s"]=np.nan
            with patch.object(pipeline,"_intentar_analisis_qwen3_real") as llm, contextlib.redirect_stdout(io.StringIO()):
                salida=ejecutar_tabla(X,archivo="captura.csv",salida=d/"salida",ruta_modelo=ruta,max_reportes=36)
            llm.assert_not_called()
            resumen=json.loads((salida/"resumen_integracion_red.json").read_text(encoding="utf-8"))
            reportes=json.loads((salida/"reportes_red.json").read_text(encoding="utf-8"),parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            self.assertEqual(resumen["n_errores_entrada"],1)
            self.assertFalse(resumen["llm_consultado"])
            self.assertTrue(all(r["ip_origen"] is None and r["timestamp"] is None for r in reportes))
            self.assertTrue(all(r["id_técnica"]=="N/A" for r in reportes))
            self.assertTrue(all(r["contexto_historico"]["incidentes_relacionados_previos"]==0 for r in reportes))
            self.assertEqual(next(r for r in reportes if r["id_alerta"]=="FLOW-000000000")["prediccion_ml"]["estado"],"error")
            for r in reportes:
                if r["prediccion_ml"].get("clase_predicha") in {"fallido","sospechoso"}:
                    self.assertNotIn("criterio independiente de escalamiento",r["decision_analisis"]["razones"])
            try:
                from streamlit.testing.v1 import AppTest
            except ImportError:
                return
            with patch.dict(os.environ,{"QSOC_RED_SALIDAS":str(d)}):
                app=AppTest.from_file(str(c.RAIZ/"dashboard_red.py"),default_timeout=20).run()
                self.assertEqual(len(app.exception),0)
                self.assertEqual(app.title[0].value,"Q-SOC | Telemetría de red")
                app.multiselect[0].set_value(["sospechoso"]).run()
                with patch("src.qwen_agent.analizar_con_qwen3",return_value={"resumen_ejecutivo":"Respuesta técnica de prueba"}) as consulta:
                    app.button[0].click().run()
                consulta.assert_called_once()
                self.assertEqual(len(app.exception),0)

    def test_csv_normal_con_severidad_critical_conserva_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);ruta,X=modelo_tecnico(d)
            pred=predecir_tabla_red(X,ruta)
            i=pred.index[pred["clase_predicha"].eq("normal")][0]
            dato=X.iloc[[i]].assign(severidad="critical")
            with contextlib.redirect_stdout(io.StringIO()):
                salida=ejecutar_tabla(dato,archivo="captura.csv",salida=d/"salida",ruta_modelo=ruta,max_reportes=1)
            r=json.loads((salida/"reportes_red.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(r["severidad"],"Crítica")
            self.assertTrue(r["requiere_escalamiento"])
            self.assertEqual(r["prediccion_ml"]["clase_predicha"],"normal")

    def test_flujo_sin_mediciones_no_utiliza_el_modelo_contextual(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ml_predictor,"_obtener_pipeline_entrenado") as viejo,contextlib.redirect_stdout(io.StringIO()):
            reportes=pipeline.ejecutar_pipeline(False,alertas=[{"id_alerta":"vacio","tipo_evento":"network_flow"}],
                                              ruta_reportes=Path(tmp)/"r.json",usar_qwen=False,
                                              ruta_modelo_red=Path(tmp)/"modelo_ausente.joblib")
        viejo.assert_not_called()
        self.assertEqual(reportes[0]["prediccion_ml"]["estado"],"error")
        self.assertTrue(reportes[0]["requiere_escalamiento"])

    def test_exportacion_reproduce_comparacion_y_preserva_evidencias(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);dataset=crear_capturas(d)
            protegidos=[c.RAIZ/"data/reportes_generados.json",c.RAIZ/"data/metricas_random_forest.json",c.RAIZ/"dashboard.py"]
            antes={p:c._sha256(p) for p in protegidos if p.exists()}
            with contextlib.redirect_stdout(io.StringIO()),patch.object(pipeline,"_intentar_analisis_qwen3_real") as llm:
                f1=c.main(["--dataset",str(dataset),"--salida",str(d/"f1"),"--hilos","1",
                           "--incluir-rf-alternativo","--sin-figuras","--verificacion-tecnica"])
                ref=f1/"resultados_completos.json"
                f2=rep.main(["--referencia",str(ref),"--dataset",str(dataset),"--data-dir",str(d),
                             "--salida",str(d/"f2"),"--hilos","1","--verificacion-tecnica"])
                f3=completar.main(["--experimento",str(f2),"--referencia-inicial",str(ref),"--dataset",str(dataset),
                                   "--data-dir",str(d),"--salida",str(d/"f3"),"--hilos","1","--verificacion-tecnica"])
                evidencias=[*f3.iterdir(),*d.glob("*pcap_ISCX.csv"),dataset]
                antes.update({p:c._sha256(p) for p in evidencias if p.is_file()})
                parametros=["--experimento",str(f3),"--referencia-inicial",str(ref),"--dataset",str(dataset),
                            "--data-dir",str(d),"--salida",str(d/"integracion"),"--modelo",str(d/"nuevo.joblib"),
                            "--max-reportes","18","--hilos","1","--verificacion-tecnica"]
                salida=activar.main(parametros)
                modelo_sha=c._sha256(d/"nuevo.joblib")
                activar.main(parametros)  # Reutiliza el mismo artefacto verificado.
            llm.assert_not_called()
            self.assertEqual(modelo_sha,c._sha256(d/"nuevo.joblib"))
            self.assertEqual(antes,{p:c._sha256(p) for p in antes})
            resumen=json.loads((salida/"resumen_integracion_red.json").read_text(encoding="utf-8"))
            anterior=json.loads((f3/"resultados_tres_modelos_red.json").read_text(encoding="utf-8"))
            self.assertEqual(resumen["n_flujos"],18)
            self.assertTrue(resumen["reportes_integrados_verificados"])
            self.assertEqual(resumen["evaluacion_clasificador"]["matriz_confusion"],
                             anterior["resultados"]["random_forest_alternativo_red"]["prueba"]["matriz_confusion"])

    def test_instalador_respalda_y_es_idempotente(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); paquete=d/"paquete"; raiz=d/"repo"
            payload=paquete/"integracion_red_archivos";payload.mkdir(parents=True);raiz.mkdir()
            anterior=b"x = 1\r\n";nuevo=b"x = 2\n"
            (raiz/"main.py").write_bytes(anterior)
            (raiz/".gitignore").write_text("*.pyc\n.env\n")
            (payload/"main.py").write_bytes(nuevo)
            (payload/"nuevo.py").write_bytes(b"y = 3\n")
            manifiesto={"archivos":[{"ruta":"main.py","sha256_anterior":instalador.sha(anterior),"sha256_nuevo":instalador.sha(nuevo)},
                                     {"ruta":"nuevo.py","sha256_anterior":None,"sha256_nuevo":instalador.sha(b"y = 3\n")} ]}
            (payload/"manifiesto.json").write_text(json.dumps(manifiesto))
            with patch.object(instalador,"__file__",str(paquete/"instalar_integracion_red.py")),contextlib.redirect_stdout(io.StringIO()):
                respaldo=instalador.instalar(raiz)
                self.assertIsNone(instalador.instalar(raiz))
            self.assertEqual((respaldo/"main.py").read_bytes(),anterior)
            self.assertEqual((raiz/"main.py").read_bytes(),nuevo)
            self.assertIn(".env\n",(raiz/".gitignore").read_text(encoding="utf-8"))
            self.assertEqual((raiz/".gitignore").read_text(encoding="utf-8").count("/outputs/integracion_red/"),1)

    def test_instalador_no_escribe_si_un_archivo_cambio(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);paquete=d/"paquete";raiz=d/"repo"
            payload=paquete/"integracion_red_archivos";payload.mkdir(parents=True);raiz.mkdir()
            (raiz/"main.py").write_text("cambio = 3\n")
            (payload/"main.py").write_bytes(b"x = 2\n")
            (payload/"manifiesto.json").write_text(json.dumps({"archivos":[{"ruta":"main.py", "sha256_anterior":instalador.sha(b"x = 1\n"),
                                                    "sha256_nuevo":instalador.sha(b"x = 2\n")}]}))
            with patch.object(instalador,"__file__",str(paquete/"instalar_integracion_red.py")):
                with self.assertRaisesRegex(ValueError,"cambió"):
                    instalador.instalar(raiz)
            self.assertFalse((raiz/"respaldos_integracion_red").exists())
            self.assertFalse((raiz/".gitignore").exists())


if __name__=="__main__":unittest.main()
