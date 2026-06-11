"""Dashboard Streamlit para visualizar el MVP académico SOAR-AI."""

from __future__ import annotations
from typing import Any

import pandas as pd
import streamlit as st

from main import ejecutar_pipeline


COLUMNAS_TABLA_SOC = [
    "id_alerta",
    "tipo_incidente",
    "severidad",
    "hostname",
    "ip_origen",
    "prioridad",
    "requiere_escalamiento",
    "táctica_mitre",
    "técnica_mitre",
    "id_técnica",
]


@st.cache_data(show_spinner="Ejecutando pipeline SOAR-AI...")
def cargar_reportes() -> list[dict[str, Any]]:
    """Ejecuta el pipeline sin salida en consola y conserva el resultado en caché."""
    return ejecutar_pipeline(imprimir=False)


def main() -> None:
    st.set_page_config(
        page_title="SOAR-AI | Dashboard SOC Académico",
        layout="wide",
    )

    st.title("SOAR-AI | Dashboard SOC Académico")
    st.caption(
        "Visualización académica de incidentes simulados. "
        "El sistema analiza, clasifica y recomienda; no ejecuta acciones reales."
    )

    reportes = cargar_reportes()
    df = pd.DataFrame(reportes)

    if df.empty:
        st.warning("No hay reportes disponibles para visualizar.")
        return

    _mostrar_kpis(df)
    _mostrar_graficos(df)
    _mostrar_tabla_soc(df)
    _mostrar_detalle_incidente(reportes, df)


def _mostrar_kpis(df: pd.DataFrame) -> None:
    incidentes_totales = len(df)
    incidentes_criticos = int((df["severidad"] == "Crítica").sum())
    incidentes_escalados = int(df["requiere_escalamiento"].fillna(False).sum())

    tecnicas_mitre = df["id_técnica"].fillna("N/A")
    cobertura_mitre = int((tecnicas_mitre != "N/A").sum())
    porcentaje_mitre = (cobertura_mitre / incidentes_totales) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Incidentes totales", incidentes_totales)
    col2.metric("Incidentes críticos", incidentes_criticos)
    col3.metric("Incidentes escalados", incidentes_escalados)
    col4.metric(
        "Cobertura MITRE ATT&CK",
        f"{cobertura_mitre}/{incidentes_totales}",
        f"{porcentaje_mitre:.1f}%",
    )


def _mostrar_graficos(df: pd.DataFrame) -> None:
    st.subheader("Distribuciones SOC")
    col1, col2 = st.columns(2)

    severidad = (
        df["severidad"]
        .fillna("N/A")
        .value_counts()
        .rename_axis("severidad")
        .reset_index(name="cantidad")
    )
    tecnica = (
        df["técnica_mitre"]
        .fillna("N/A")
        .value_counts()
        .rename_axis("técnica_mitre")
        .reset_index(name="cantidad")
    )

    with col1:
        st.markdown("**Distribución por severidad**")
        st.bar_chart(severidad, x="severidad", y="cantidad")

    with col2:
        st.markdown("**Distribución por técnica MITRE**")
        st.bar_chart(tecnica, x="técnica_mitre", y="cantidad")


def _mostrar_tabla_soc(df: pd.DataFrame) -> None:
    st.subheader("Tabla SOC")
    columnas_disponibles = [
        columna for columna in COLUMNAS_TABLA_SOC if columna in df.columns
    ]
    st.dataframe(
        df[columnas_disponibles],
        width="stretch",
        hide_index=True,
    )


def _mostrar_detalle_incidente(reportes: list[dict[str, Any]], df: pd.DataFrame) -> None:
    st.subheader("Detalle de incidente")

    ids_alerta = df["id_alerta"].astype(str).tolist()
    id_seleccionado = st.selectbox("Seleccionar incidente", ids_alerta)
    reporte = next(
        reporte for reporte in reportes if str(reporte.get("id_alerta")) == id_seleccionado
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Evidencia**")
        _mostrar_lista(reporte.get("evidencia"))

        st.markdown("**Recomendación playbook**")
        _mostrar_lista(reporte.get("recomendación_playbook"))

    with col2:
        st.markdown("**Contexto histórico**")
        _mostrar_contexto_historico(reporte.get("contexto_historico"))

    st.markdown("**Análisis IA Qwen3**")
    _mostrar_analisis_qwen3(reporte.get("analisis_qwen3_real"))


def _mostrar_analisis_qwen3(analisis: Any) -> None:
    if not isinstance(analisis, dict):
        st.info("Análisis Qwen3 no disponible como JSON estructurado.")
        if analisis:
            with st.expander("Ver contenido recibido de Qwen3"):
                st.write(analisis)
        return

    if analisis.get("error"):
        st.warning(
            "Qwen3 no entregó un análisis estructurado válido o no está disponible."
        )
        with st.expander("Ver contenido original de Qwen3"):
            contenido_original = analisis.get("respuesta_original")
            if contenido_original:
                st.code(str(contenido_original), language="text")
            else:
                st.json(analisis, expanded=False)
        return

    claves_reconocidas = {
        "resumen_ejecutivo",
        "hechos_observados",
        "inferencias_razonables",
        "analisis_riesgo",
        "explicacion_mitre",
        "uso_contexto_historico",
        "recomendacion_soc",
        "posibles_falsos_positivos",
        "siguiente_accion_sugerida",
    }
    if not any(clave in analisis for clave in claves_reconocidas):
        st.info("Qwen3 devolvió un JSON, pero no coincide con el esquema esperado.")
        with st.expander("Ver JSON recibido"):
            st.json(analisis, expanded=False)
        return

    st.markdown(
        f"**Resumen ejecutivo:** "
        f"{analisis.get('resumen_ejecutivo', 'No disponible en el reporte')}"
    )

    if "hechos_observados" in analisis:
        st.markdown("**Hechos observados**")
        _mostrar_lista(analisis.get("hechos_observados"))

    if "inferencias_razonables" in analisis:
        st.markdown("**Inferencias razonables**")
        _mostrar_lista(analisis.get("inferencias_razonables"))

    st.markdown(
        f"**Análisis de riesgo:** "
        f"{analisis.get('analisis_riesgo', 'No disponible en el reporte')}"
    )
    st.markdown(
        "**Explicación MITRE ATT&CK:** "
        f"{analisis.get('explicacion_mitre', 'No disponible en el reporte')}"
    )

    if "uso_contexto_historico" in analisis:
        st.markdown(
            "**Uso del contexto histórico:** "
            f"{analisis.get('uso_contexto_historico', 'No disponible en el reporte')}"
        )

    st.markdown("**Recomendación SOC**")
    _mostrar_lista(analisis.get("recomendacion_soc"))

    st.markdown("**Posibles falsos positivos**")
    _mostrar_lista(analisis.get("posibles_falsos_positivos"))

    st.markdown(
        "**Siguiente acción sugerida:** "
        f"{analisis.get('siguiente_accion_sugerida', 'No disponible en el reporte')}"
    )


def _mostrar_contexto_historico(contexto: Any) -> None:
    if not isinstance(contexto, dict) or not contexto:
        st.write("No disponible en el reporte.")
        return

    st.write(
        "**IP origen recurrente:** "
        f"{_formatear_booleano(contexto.get('ip_origen_recurrente'))}"
    )

    usuarios = contexto.get("usuarios_recurrentes") or []
    usuarios_texto = ", ".join(usuarios) if usuarios else "No"
    st.write(f"**Usuarios recurrentes:** {usuarios_texto}")

    st.write(
        "**Hostname recurrente:** "
        f"{_formatear_booleano(contexto.get('hostname_recurrente'))}"
    )
    st.write(
        "**Incidentes relacionados previos:** "
        f"{contexto.get('incidentes_relacionados_previos', 0)}"
    )
    st.write(
        "**Incidentes críticos previos:** "
        f"{contexto.get('incidentes_criticos_previos', 0)}"
    )
    st.write(
        "**Patrón repetido:** "
        f"{_formatear_booleano(contexto.get('patron_repetido'))}"
    )
    st.info(contexto.get("resumen_contexto", "No disponible en el reporte."))


def _formatear_booleano(valor: Any) -> str:
    return "Sí" if bool(valor) else "No"


def _mostrar_lista(valores: Any) -> None:
    if not valores:
        st.write("No disponible en el reporte.")
        return

    if not isinstance(valores, list):
        st.write(valores)
        return

    for valor in valores:
        st.markdown(f"- {valor}")


if __name__ == "__main__":
    main()
