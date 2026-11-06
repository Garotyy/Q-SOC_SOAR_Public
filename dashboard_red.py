"""Panel de la ruta de telemetría; conserva el dashboard de incidentes existente."""
from pathlib import Path
import json
import os

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
SALIDAS = Path(os.environ.get("QSOC_RED_SALIDAS", str(RAIZ/"outputs/integracion_red")))


def numero(valor):
    return f"{int(valor):,}".replace(",", ".")


def porcentaje(valor):
    return f"{100*float(valor):.2f}".replace(".", ",")+" %"


def leer(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))


def main():
    st.set_page_config(page_title="Q-SOC | Telemetría de red", page_icon="🛡️", layout="wide")
    st.title("Q-SOC | Telemetría de red")
    st.caption("Clasificación de flujos, revisión asistida y análisis avanzado bajo demanda.")
    carpetas = sorted([p.parent for p in SALIDAS.glob("*/resumen_integracion_red.json")], reverse=True)
    if not carpetas:
        st.info("Todavía no hay una ejecución de la ruta de red. Ejecuta su activación o carga un CSV desde la consola.")
        st.stop()
    carpeta = st.selectbox("Ejecución", carpetas, format_func=lambda p:p.name)
    resumen = leer(carpeta/"resumen_integracion_red.json")
    ficha = leer(carpeta/"ficha_modelo.json")
    reportes = leer(carpeta/"reportes_red.json")
    columnas = st.columns(4)
    for c, etiqueta, clave in zip(columnas, ["Flujos procesados", "Revisión sugerida", "Filtrados", "Errores de entrada"],
                                  ["n_flujos", "n_revision_sugerida", "n_filtrados", "n_errores_entrada"]):
        c.metric(etiqueta, numero(resumen[clave]))
    st.caption("Las cifras describen decisiones sobre flujos; no cuentan incidentes consolidados ni acciones ejecutadas.")
    if ficha.get("modo") == "verificacion_tecnica":
        st.info("Ejecución técnica de verificación; estas cifras no son resultados de CICIDS2017.")
    tabs = st.tabs(["Resumen", "Evaluación del clasificador", "Detalle y análisis avanzado"])
    with tabs[0]:
        st.subheader("Distribución de predicciones")
        st.bar_chart(pd.DataFrame.from_dict(resumen["clases_predichas"], orient="index", columns=["Flujos"]))
        st.write(f"Reportes detallados de muestra: **{numero(resumen['n_reportes_muestra'])}**.")
        st.caption("Las predicciones y la cola completa de revisión se conservan en los archivos de esta ejecución.")
        if not resumen["identidad_ip_disponible"]:
            st.info("Esta entrada no aporta IP de origen. La correlación por identidad requiere esa información adicional.")
    with tabs[1]:
        metricas = resumen.get("evaluacion_clasificador")
        if metricas:
            st.subheader("Evaluación reproducida sobre el conjunto de prueba")
        else:
            st.subheader("Evaluación de referencia del modelo")
            metricas = ficha.get("evaluacion_referencia")
            st.caption("Estas métricas pertenecen a la prueba histórica del modelo y no evalúan el CSV recién procesado.")
        if metricas:
            cs = st.columns(4)
            for c, nombre, clave in zip(cs, ["Exactitud", "Precisión macro", "Recall macro", "F1 macro"],
                                        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]):
                c.metric(nombre, porcentaje(metricas[clave]))
            cm = metricas["matriz_confusion"]
            matriz = pd.DataFrame(cm["valores"], index=cm["labels"], columns=cm["labels"])
            matriz.index.name = "Clase real / predicha"
            st.dataframe(matriz.map(numero), width="stretch")
            f = metricas["filtrado_binario"]
            st.write(f"Falsas alarmas: **{numero(f['falsos_positivos'])}** · Ataques omitidos: **{numero(f['falsos_negativos'])}**")
            st.caption("La evaluación sobre las mismas capturas no equivale a validación externa ni mide trabajo humano.")
    with tabs[2]:
        clases = st.multiselect("Clasificación ML", ["normal", "fallido", "sospechoso", "error"])
        visibles = [r for r in reportes if not clases or r["prediccion_ml"].get("clase_predicha", "error") in clases]
        st.caption("Se muestran reportes de una muestra, elegida por predicción sin usar la etiqueta real.")
        if not visibles:
            st.info("No hay reportes de muestra para este filtro.")
            return
        elegido = st.selectbox("Flujo", visibles, format_func=lambda r:r["id_alerta"])
        st.write("**Clasificación:**", elegido["prediccion_ml"].get("clase_predicha", "Error de entrada"))
        st.write("**Prioridad:**", elegido["prioridad"], " · **Severidad:**", elegido["severidad"])
        st.write("**Procedencia:**", elegido.get("procedencia_red", {}))
        with st.expander("Evidencia y mediciones", expanded=True):
            for hecho in elegido["evidencia"]: st.write("•", hecho)
            st.json(elegido.get("telemetria_red", {}))
        with st.expander("Recomendaciones y contexto"):
            st.write(elegido.get("recomendación_playbook", []))
            st.write(elegido.get("contexto_historico", {}))
            st.caption("MITRE se mantiene sin asignación cuando las mediciones no identifican una técnica específica.")
        st.subheader("Análisis avanzado con Q-SOC")
        decision = elegido.get("decision_analisis", {})
        if not decision.get("habilitar_analisis_avanzado"):
            st.info("El modelo clasificó este flujo como normal y no se activaron criterios independientes de revisión.")
        else:
            clave = f"analisis-red-{carpeta.name}-{elegido['id_alerta']}"
            if st.button("Generar análisis avanzado con Q-SOC", key="boton-"+clave):
                with st.spinner("Generando análisis avanzado con Q-SOC..."):
                    try:
                        from src.qwen_agent import analizar_con_qwen3
                        st.session_state[clave] = analizar_con_qwen3(elegido)
                    except Exception as error:
                        st.session_state[clave] = {"error":"servicio_no_disponible", "detalle":str(error)}
            if clave in st.session_state:
                analisis = st.session_state[clave]
                if analisis.get("error"):
                    st.warning("El análisis avanzado no está disponible. El reporte y la revisión se conservan.")
                    with st.expander("Detalle técnico"): st.json(analisis)
                else:
                    for campo, valor in analisis.items():
                        st.markdown("**"+campo.replace("_", " ").capitalize()+"**")
                        st.write(valor)
        with st.expander("JSON / detalles técnicos"):
            st.json(elegido)


if __name__ == "__main__":
    main()
