"""Pruebas técnicas con fixtures temporales; no producen resultados de tesis."""
import argparse
import contextlib
import io
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import ejecutar_prueba_umbrales as e
from src import umbrales_experimentales as u
from src.telemetria_red import VARIABLES_RED, FORMATO_MODELO_RED


class ModeloFijo:
    def predict_proba(self, valores):
        assert valores.shape[1] == 12
        return np.tile([0.05, 0.1, 0.85], (len(valores), 1))


def raw_fixture(n=20, etiqueta="BENIGN", seed=7):
    rng = np.random.default_rng(seed)
    datos = pd.DataFrame(rng.uniform(1, 100, (n, 12)), columns=VARIABLES_RED)
    datos["Destination Port"] = rng.choice([21, 22, 53, 80, 443], n)
    datos["SYN Flag Count"] = rng.integers(0, 2, n)
    datos["ACK Flag Count"] = rng.integers(0, 2, n)
    datos["Bwd Packets/s"] = rng.uniform(0, 20, n)
    datos["Label"] = etiqueta
    return datos


class PoliticaTests(unittest.TestCase):
    def test_reduce_fp_sin_perder_ataques_en_validacion(self):
        p = np.array([[.1,.4,.5], [.0,.1,.9], [.05,.9,.05], [.2,.6,.2]])
        y = np.array([1,2,1,0])
        s, curva = u.seleccionar_politica(y,p)
        self.assertTrue(s["mejora_en_validacion"])
        self.assertEqual(s["base_validacion"]["falsos_positivos"],1)
        self.assertEqual(s["propuesta_validacion"]["falsos_positivos"],0)
        self.assertEqual(s["transiciones_validacion"]["ataques_antes_detectados_ahora_omitidos"],0)
        self.assertEqual(len(curva),len(u.UMBRALES))

    def test_si_no_hay_mejora_conserva_argmax(self):
        p = np.array([[.1,.4,.5], [.1,.4,.5], [.05,.9,.05]])
        s,_ = u.seleccionar_politica(np.array([1,2,1]),p)
        self.assertFalse(s["mejora_en_validacion"])
        self.assertEqual(s["politica"],e.BASE)

    def test_fn_neto_igual_no_oculta_ataque_nuevo_perdido(self):
        p = np.array([[.5,.49,.01], [.3,.4,.3], [.5,.49,.01], [.01,.98,.01]])
        y = np.array([0,2,1,1])
        base = u.aplicar_politica(p,e.BASE)
        otra = u.aplicar_politica(p,{"regla":"puntuacion_alerta","umbral":.55})
        self.assertEqual(u.medir_decision(y,base)["falsos_negativos"],u.medir_decision(y,otra)["falsos_negativos"])
        self.assertEqual(u.transiciones(y,base,otra)["ataques_antes_detectados_ahora_omitidos"],1)
        s,_ = u.seleccionar_politica(y,p)
        self.assertFalse(s["mejora_en_validacion"])

    def test_denominadores_y_sumatoria_son_correctos(self):
        y=np.array([1,1,1,0,2])
        r=np.array([True,False,False,True,False])
        m=u.medir_decision(y,r)
        self.assertAlmostEqual(m["tasa_falsos_positivos"],1/3)
        self.assertAlmostEqual(m["tasa_falsos_negativos"],1/2)
        self.assertEqual(u.sumar_metricas([m,m])["n"],10)

    def test_particion_por_vectores_no_cruza_grupos(self):
        x=np.repeat(np.arange(300,dtype=float)[:,None],12,axis=1)
        x=np.repeat(x,2,axis=0)
        y=np.tile([0,1,2],200)
        f=u.firmas_vectores(x)
        a,v=u.dividir_desarrollo(np.arange(len(y)),y,f)
        self.assertEqual(np.intersect1d(f[a],f[v]).size,0)
        a2,v2=u.dividir_desarrollo(np.arange(len(y)),y,f)
        np.testing.assert_array_equal(a,a2)
        np.testing.assert_array_equal(v,v2)

    def test_rechaza_probabilidades_invalidas_y_umbrales_invalidos(self):
        for p in ([[.1,.1,.1]],[[np.nan,0,1]],[[1,0]],[[1.1,-.1,0]]):
            with self.subTest(p=p),self.assertRaises(ValueError):u.validar_probabilidades(p)
        with self.assertRaises(ValueError):
            u.aplicar_politica([[.1,.8,.1]],{"regla":"puntuacion_alerta","umbral":True})

    def test_faltan_clases_no_se_busca_otra_semilla(self):
        x=np.arange(100)[:,None]
        with self.assertRaises(ValueError):
            u.dividir_desarrollo(np.arange(100),np.ones(100,dtype=int),u.firmas_vectores(x))

    def test_redondeo_minimo_no_se_confunde_con_otro_modelo(self):
        a=np.array([[.117,.729,.154],[.811,.103,.086]])
        b=a.copy();b[:,0]+=2e-16;b[:,1]-=2e-16
        r=u.comprobar_serializacion(a,b)
        self.assertGreater(r["diferencia_absoluta_maxima"],0)
        self.assertTrue(r["decisiones_identicas_en_toda_la_grilla"])

    def test_tolerancia_no_oculta_un_cambio_real(self):
        a=np.array([[.117,.729,.154]])
        b=a.copy();b[:,0]+=.001;b[:,1]-=.001
        with self.assertRaises(RuntimeError):u.comprobar_serializacion(a,b)

    def test_redondeo_que_cambia_revision_se_rechaza(self):
        a=np.array([[.25,.5,.25]])
        b=a.copy();b[:,0]-=1e-16;b[:,1]+=1e-16
        with self.assertRaises(RuntimeError):u.comprobar_serializacion(a,b)

    def test_redondeo_que_cambia_clase_se_rechaza(self):
        a=np.array([[.5,.5,0.]])
        b=a.copy();b[:,0]-=1e-16;b[:,1]+=1e-16
        with self.assertRaises(RuntimeError):u.comprobar_serializacion(a,b)

    def test_rf_con_cuatro_hilos_predice_estable_tras_serializar(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(1200,12)).astype(np.float32);y=rng.integers(0,3,len(x))
        m=RandomForestClassifier(n_estimators=30,max_depth=8,min_samples_leaf=5,n_jobs=4,random_state=42).fit(x,y)
        a=u.predecir_estable(m,x)
        self.assertEqual(m.n_jobs,4)
        with tempfile.TemporaryDirectory() as tmp:
            ruta=Path(tmp)/"modelo.joblib";joblib.dump(m,ruta);recargado=joblib.load(ruta)
            b=u.predecir_estable(recargado,x)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(u.comprobar_serializacion(a,b)["clases_identicas"])
        self.assertEqual(recargado.n_jobs,4)


class CapturaTests(unittest.TestCase):
    def test_no_recupera_ejecucion_que_ya_consulto_reservadas(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);origen=p/"intento";origen.mkdir()
            (origen/"modelo_experimental.joblib").write_bytes(b"fixture sin carga")
            (origen/"politica_congelada.json").write_text("{}")
            protocolo={"estado":"iniciado","version":"umbrales_reservados_v1",
                       "referencia_tres_modelos_sha256":"referencia",
                       "codigo_sha256":{"ejecutar_prueba_umbrales.py":e.CODIGO_ORIGEN_RECUPERABLE,
                                         "src\\umbrales_experimentales.py":e.NUCLEO_ORIGEN_RECUPERABLE}}
            e.guardar(origen/"protocolo.json",protocolo)
            args=argparse.Namespace(reutilizar_entrenamiento=origen,retomar_ultimo=False,salida=p)
            self.assertEqual(e.localizar_entrenamiento_previo(args,"referencia"),origen)
            (origen/"predicciones_captura.csv.gz").write_bytes(b"ya evaluada")
            with self.assertRaises(ValueError):e.localizar_entrenamiento_previo(args,"referencia")

    def test_no_abre_csv_antes_de_fijar_politica(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with patch.object(e.pd,"read_csv") as lectura,self.assertRaises(RuntimeError):
                e.evaluar_captura(p/"sin.csv","",ModeloFijo(),ModeloFijo(),e.BASE,
                                  np.array([],dtype=np.uint64),p,p/"politica.json")
            lectura.assert_not_called()

    def test_encoding_y_limpieza_se_documentan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ruta=p/"captura.csv"
            df=raw_fixture(4,"Web Attack – XSS")
            df["Extra"]=[1,np.nan,1,1]
            df.to_csv(ruta,index=False,encoding="cp1252")
            politica=p/"politica.json";politica.write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()):
                r=e.evaluar_captura(ruta,e.comun._sha256(ruta),ModeloFijo(),ModeloFijo(),e.BASE,
                                    np.array([],dtype=np.uint64),p,politica)
            self.assertEqual(r["codificacion"],"cp1252")
            self.assertEqual(r["registros_evaluados"],3)
            self.assertEqual(r["descartados_limpieza_todas_las_columnas"],1)
            self.assertEqual(r["por_etiqueta_original"][0]["clase_qsoc"],"sospechoso")

    def test_cambiar_label_no_cambia_predicciones(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);politica=p/"politica.json";politica.write_text("{}")
            df=raw_fixture(10)
            pred=[]
            for label in ("BENIGN","DDoS"):
                df["Label"]=label;ruta=p/(label+".csv");df.to_csv(ruta,index=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    r=e.evaluar_captura(ruta,e.comun._sha256(ruta),ModeloFijo(),ModeloFijo(),
                                        {"regla":"puntuacion_alerta","umbral":.95},np.array([],dtype=np.uint64),p,politica)
                tabla=pd.read_csv(p/r["archivo_predicciones"])
                pred.append(tabla[["clase_rf_experimental","puntuacion_alerta_rf_experimental","revision_propuesta_experimental"]])
            pd.testing.assert_frame_equal(*pred)

    def test_rechaza_captura_cambiada(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ruta=p/"captura.csv";raw_fixture().to_csv(ruta,index=False)
            politica=p/"politica.json";politica.write_text("{}")
            with self.assertRaises(ValueError):
                e.evaluar_captura(ruta,"hash incorrecto",ModeloFijo(),ModeloFijo(),e.BASE,
                                  np.array([],dtype=np.uint64),p,politica)


class ProtocoloCompletoTests(unittest.TestCase):
    def test_entrena_fija_evalua_y_conserva_modelo_integrado(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp);data=raiz/"data";data.mkdir();futuras=raiz/"reservadas";futuras.mkdir()
            exp=raiz/"comparacion";exp.mkdir()
            frames=[];archivos=[]
            for i,nombre in enumerate(["Monday-WorkingHours.pcap_ISCX.csv","Tuesday-WorkingHours.pcap_ISCX.csv","Wednesday-workingHours.pcap_ISCX.csv"]):
                df=raw_fixture(300,seed=20+i)
                if i==1:df.loc[100:199,"Label"]="SSH-Patator";df.loc[200:,"Label"]="FTP-Patator"
                if i==2:df.loc[100:,"Label"]="DoS Hulk"
                ruta=data/nombre;df.to_csv(ruta,index=False);frames.append(df)
                archivos.append({"archivo":nombre,"sha256":e.comun._sha256(ruta)})
            raw=pd.concat(frames,ignore_index=True)
            hom=e.rep.homologar_para_verificar(raw)
            dataset=data/"dataset_kali_homologado.csv";hom.to_csv(dataset,index=False)
            y=hom[e.comun.OBJETIVO].map(e.CODIGOS).to_numpy()
            train,test=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=42)
            inicial=raiz/"inicial.json"
            inicial.write_text(json.dumps({"protocolo":{"dataset":{"sha256":e.comun._sha256(dataset)}}}))
            parametros=RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=5,
                min_samples_split=15,class_weight="balanced",random_state=42,n_jobs=1).get_params()
            referencia={"protocolo":{"estado":"completado","modo":"comparacion_tres_modelos_red",
                "scikit_learn":sklearn.__version__,"features_originales":VARIABLES_RED,
                "referencia_inicial_sha256":e.comun._sha256(inicial),"cohorte":{"archivos":archivos},
                "particion":{"test_size":.25,"semilla":42,"sha256_indices_train":u.sha_indices(train),"sha256_indices_test":u.sha_indices(test)}},
                "resultados":{"random_forest_alternativo_red":{"parametros_efectivos":parametros}}}
            ref=exp/"resultados_tres_modelos_red.json";ref.write_text(json.dumps(referencia))
            for i,nombre in enumerate(e.RESERVADAS):
                df=raw_fixture(20,seed=100+i);df.loc[10:,"Label"]="DDoS"
                df.to_csv(futuras/nombre,index=False)
            x,_=e.matriz_red(raw[VARIABLES_RED])
            modelo=RandomForestClassifier(**parametros).fit(x[train],y[train])
            referencia["resultados"]["random_forest_alternativo_red"]["prueba"]=e.comun.calcular_metricas(y[test],modelo.predict(x[test]))
            ref.write_text(json.dumps(referencia))
            modelo_ruta=data/"modelo_random_forest_red.joblib"
            joblib.dump({"formato":FORMATO_MODELO_RED,"features":VARIABLES_RED,"clases":list(e.CLASES),
                         "modelo":modelo,"referencia_sha256":e.comun._sha256(ref)},modelo_ruta)
            antes=modelo_ruta.read_bytes()
            args=argparse.Namespace(experimento=exp,referencia_inicial=inicial,dataset=dataset,data_dir=data,
                 capturas_reservadas=futuras,modelo_actual=modelo_ruta,salida=raiz/"salidas",hilos=1)
            real=e.evaluar_captura;orden=[]
            def evaluar(*args,**kwargs):
                self.assertTrue(args[-1].is_file())
                orden.append(args[0].name)
                return real(*args,**kwargs)
            with patch.object(e,"RAIZ",Path(e.__file__).resolve().parent),patch.object(e,"evaluar_captura",side_effect=evaluar),contextlib.redirect_stdout(io.StringIO()):
                salida=e.ejecutar(args)
            r=e.leer_json(salida/"resultados_umbrales.json")
            self.assertEqual(r["protocolo"]["estado"],"completado")
            self.assertEqual(r["validacion"]["origen_seleccion"],"solo_validacion")
            self.assertEqual(orden,list(e.RESERVADAS))
            self.assertEqual(modelo_ruta.read_bytes(),antes)
            self.assertEqual(r["evaluacion_capturas_reservadas"]["agregado"]["base_experimental"]["n"],100)
            self.assertTrue((salida/"politica_congelada.json").is_file())
            self.assertIsNone(r["seleccion_o_integracion_automatica"])
            # Recrea un checkpoint v1 detenido antes de abrir las reservadas.
            viejo=raiz/"interrumpido_v1";viejo.mkdir()
            congelada=e.leer_json(salida/"politica_congelada.json")
            congelada["version"]="umbrales_reservados_v1"
            e.guardar(viejo/"politica_congelada.json",congelada)
            bundle=joblib.load(salida/"modelo_experimental.joblib")
            bundle["politica_sha256"]=e.comun._sha256(viejo/"politica_congelada.json")
            joblib.dump(bundle,viejo/"modelo_experimental.joblib")
            anterior=dict(r["protocolo"])
            anterior.update({"estado":"iniciado","version":"umbrales_reservados_v1",
                "codigo_sha256":{"ejecutar_prueba_umbrales.py":e.CODIGO_ORIGEN_RECUPERABLE,
                                  "src\\umbrales_experimentales.py":e.NUCLEO_ORIGEN_RECUPERABLE}})
            e.guardar(viejo/"protocolo.json",anterior)
            args.reutilizar_entrenamiento=viejo
            self.assertEqual(e.localizar_entrenamiento_previo(args,e.comun._sha256(ref)),viejo.resolve())
            conservado=(viejo/"modelo_experimental.joblib").read_bytes()
            with patch.object(RandomForestClassifier,"fit",side_effect=AssertionError("No debe volver a entrenar")),contextlib.redirect_stdout(io.StringIO()):
                retomado=e.ejecutar(args)
            r2=e.leer_json(retomado/"resultados_umbrales.json")
            self.assertIsNone(r2["protocolo"]["tiempo_ajuste_rf_segundos"])
            self.assertIn("entrenamiento_reutilizado",r2["protocolo"])
            self.assertEqual(conservado,(viejo/"modelo_experimental.joblib").read_bytes())
            self.assertEqual(modelo_ruta.read_bytes(),antes)


if __name__ == "__main__":
    unittest.main()
