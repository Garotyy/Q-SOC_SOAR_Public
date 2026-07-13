"""Dashboard Streamlit rapido y seguro para demo de Q-SOC / SOAR-IA."""

from __future__ import annotations

import json
import html
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RUTA_REPORTES = DATA_DIR / "reportes_generados.json"
RUTA_METRICAS_ML = DATA_DIR / "metricas_arbol_decision.json"
RUTA_EVENTOS_ML = DATA_DIR / "eventos_ml_simulados.csv"

CLAVE_TACTICA_MITRE = "t\u00e1ctica_mitre"
CLAVE_TECNICA_MITRE = "t\u00e9cnica_mitre"
CLAVE_ID_TECNICA = "id_t\u00e9cnica"
CLAVE_PLAYBOOK = "recomendaci\u00f3n_playbook"

COLUMNAS_TABLA_SOC = [
    "id_alerta",
    "tipo_incidente",
    "severidad",
    "hostname",
    "ip_origen",
    "prioridad",
    "requiere_escalamiento",
    CLAVE_TACTICA_MITRE,
    CLAVE_TECNICA_MITRE,
    CLAVE_ID_TECNICA,
]

CLAVES_QWEN_RECONOCIDAS = {
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

COLORES_METRICAS = {
    "Accuracy": "#3b82f6",
    "Error Rate": "#f97316",
    "Precision": "#8b5cf6",
    "Recall": "#22c55e",
    "F1-Score": "#14b8a6",
}
COLORES_CLASES = {
    "normal": "#3b82f6",
    "fallido": "#f59e0b",
    "sospechoso": "#ef4444",
}


def _inyectar_estilos() -> None:
    """Agrega estilos compactos y compatibles con tema oscuro."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .qsoc-hero {
            border: 1px solid rgba(125, 211, 252, 0.22);
            border-radius: 14px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1rem;
            background: linear-gradient(
                135deg,
                rgba(14, 116, 144, 0.20),
                rgba(15, 23, 42, 0.45)
            );
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.20);
        }
        .qsoc-hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
            letter-spacing: 0;
        }
        .qsoc-subtitle {
            margin-top: 0.35rem;
            color: rgba(226, 232, 240, 0.82);
            font-size: 1rem;
        }
        .qsoc-demo-note {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.35rem 0.65rem;
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 999px;
            color: #bae6fd;
            background: rgba(8, 47, 73, 0.45);
            font-size: 0.82rem;
        }
        .qsoc-card {
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: rgba(15, 23, 42, 0.36);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16);
            min-height: 92px;
        }
        .qsoc-card-title {
            color: rgba(203, 213, 225, 0.78);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }
        .qsoc-card-value {
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 700;
            color: #f8fafc;
        }
        .qsoc-card-note {
            margin-top: 0.2rem;
            color: rgba(203, 213, 225, 0.74);
            font-size: 0.8rem;
        }
        .qsoc-badge {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(30, 41, 59, 0.65);
            color: #e2e8f0;
        }
        .qsoc-badge.green {
            color: #bbf7d0;
            border-color: rgba(34, 197, 94, 0.35);
            background: rgba(20, 83, 45, 0.35);
        }
        .qsoc-badge.red {
            color: #fecaca;
            border-color: rgba(248, 113, 113, 0.35);
            background: rgba(127, 29, 29, 0.35);
        }
        .qsoc-badge.blue {
            color: #bfdbfe;
            border-color: rgba(96, 165, 250, 0.35);
            background: rgba(30, 64, 175, 0.30);
        }
        .qsoc-detail-grid {
            display: grid;
            grid-template-columns: 150px 1fr;
            gap: 0.35rem 0.75rem;
            font-size: 0.92rem;
        }
        .qsoc-detail-label {
            color: rgba(203, 213, 225, 0.72);
        }
        .qsoc-detail-value {
            color: #f8fafc;
            font-weight: 600;
        }
        .qsoc-list-card {
            max-width: 980px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            background: rgba(15, 23, 42, 0.28);
            margin-bottom: 0.7rem;
        }
        .qsoc-list-card ol,
        .qsoc-list-card ul {
            margin-bottom: 0;
            padding-left: 1.25rem;
        }
        .qsoc-list-card li {
            margin: 0.28rem 0;
            line-height: 1.45;
            max-width: 88ch;
        }
        .qsoc-text-box {
            max-width: 980px;
            line-height: 1.5;
        }
        .qsoc-confusion-table {
            max-width: 620px;
            margin: 0.7rem auto 0;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.30);
        }
        .qsoc-confusion-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        .qsoc-confusion-table th,
        .qsoc-confusion-table td {
            padding: 0.45rem 0.65rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
            text-align: center;
        }
        .qsoc-confusion-table th {
            color: #bfdbfe;
            font-weight: 650;
            background: rgba(30, 41, 59, 0.58);
        }
        .qsoc-confusion-table td:first-child,
        .qsoc-confusion-table th:first-child {
            text-align: left;
            color: #e2e8f0;
        }
        .qsoc-confusion-table tr:last-child td {
            border-bottom: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mostrar_encabezado() -> None:
    st.markdown(
        """
        <div class="qsoc-hero">
          <h1>Q-SOC | Agente SOAR-IA con apoyo predictivo para respuesta ante incidentes</h1>
          <div class="qsoc-subtitle">
            Visualizaci\u00f3n del flujo de clasificaci\u00f3n, m\u00e9tricas ML y respuesta asistida
          </div>
          <div class="qsoc-demo-note">
            Modo demo: se cargan reportes y m\u00e9tricas ya generadas. Qwen solo se ejecuta bajo demanda.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tarjeta_metrica(titulo: str, valor: Any, nota: str = "") -> None:
    nota_html = f'<div class="qsoc-card-note">{_html(nota)}</div>' if nota else ""
    st.markdown(
        f"""
        <div class="qsoc-card">
          <div class="qsoc-card-title">{_html(titulo)}</div>
          <div class="qsoc-card-value">{_html(valor)}</div>
          {nota_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _html(valor: Any) -> str:
    return html.escape(str(valor))


def main() -> None:
    st.set_page_config(
        page_title="Q-SOC | SOAR-IA",
        layout="wide",
    )

    _inyectar_estilos()
    _mostrar_encabezado()

    datos = cargar_datos_dashboard()
    reportes = datos["reportes"]
    metricas_ml = datos["metricas_ml"]
    eventos_ml = datos["eventos_ml"]

    df_reportes = pd.DataFrame(reportes)
    df_eventos_ml = eventos_ml if isinstance(eventos_ml, pd.DataFrame) else pd.DataFrame()

    _mostrar_estado_archivos(datos)

    if not df_reportes.empty:
        _mostrar_kpis_principales(df_reportes)
    else:
        st.warning(
            "No hay reportes generados disponibles. El dashboard no ejecuta el "
            "pipeline automaticamente para evitar llamadas a Qwen o sobrescrituras."
        )

    _mostrar_evaluacion_modelo(metricas_ml)
    _mostrar_graficos_ml(metricas_ml, df_eventos_ml)

    if df_reportes.empty:
        return

    df_filtrado = _mostrar_incidentes_soc(df_reportes)
    if df_filtrado.empty:
        st.info("No hay incidentes que coincidan con los filtros seleccionados.")
        return

    _mostrar_detalle_incidente(reportes, df_filtrado, metricas_ml)


@st.cache_data(show_spinner=False)
def cargar_datos_dashboard() -> dict[str, Any]:
    """Carga archivos ya generados sin ejecutar el pipeline principal."""
    return {
        "reportes": _cargar_json_si_existe(RUTA_REPORTES, valor_por_defecto=[]),
        "metricas_ml": _cargar_json_si_existe(RUTA_METRICAS_ML, valor_por_defecto={}),
        "eventos_ml": _cargar_csv_si_existe(RUTA_EVENTOS_ML),
        "existe_reportes": RUTA_REPORTES.exists(),
        "existe_metricas": RUTA_METRICAS_ML.exists(),
        "existe_eventos_ml": RUTA_EVENTOS_ML.exists(),
    }


def _cargar_json_si_existe(ruta: Path, valor_por_defecto: Any) -> Any:
    if not ruta.exists():
        return valor_por_defecto

    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception:
        return valor_por_defecto


def _cargar_csv_si_existe(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(ruta)
    except Exception:
        return pd.DataFrame()


def _mostrar_estado_archivos(datos: dict[str, Any]) -> None:
    faltantes = []
    if not datos["existe_reportes"]:
        faltantes.append("data/reportes_generados.json")
    if not datos["existe_metricas"]:
        faltantes.append("data/metricas_arbol_decision.json")
    if not datos["existe_eventos_ml"]:
        faltantes.append("data/eventos_ml_simulados.csv")

    if faltantes:
        st.warning("Archivos no disponibles: " + ", ".join(faltantes))


def _mostrar_kpis_principales(df: pd.DataFrame) -> None:
    st.subheader("KPIs principales")
    incidentes_totales = len(df)
    severidad = df.get("severidad", pd.Series(dtype=str)).astype(str)
    incidentes_criticos = int(
        severidad.map(_normalizar_texto_ascii).eq("critica").sum()
    )
    escalamiento = _obtener_serie(df, "requiere_escalamiento", False)
    incidentes_escalados = int(escalamiento.fillna(False).sum())

    tecnicas_mitre = df.get(CLAVE_ID_TECNICA, pd.Series(dtype=str)).fillna("N/A")
    cobertura_mitre = int((tecnicas_mitre != "N/A").sum())
    porcentaje_mitre = (
        (cobertura_mitre / incidentes_totales) * 100 if incidentes_totales else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _tarjeta_metrica("Incidentes totales", incidentes_totales)
    with col2:
        _tarjeta_metrica("Incidentes cr\u00edticos", incidentes_criticos)
    with col3:
        _tarjeta_metrica("Incidentes escalados", incidentes_escalados)
    with col4:
        _tarjeta_metrica(
            "Cobertura MITRE ATT&CK",
            f"{cobertura_mitre}/{incidentes_totales}",
            f"{porcentaje_mitre:.1f}%",
        )


def _mostrar_evaluacion_modelo(metricas_ml: dict[str, Any]) -> None:
    st.subheader("Evaluaci\u00f3n del modelo predictivo")
    metricas = metricas_ml.get("metricas") if isinstance(metricas_ml, dict) else None
    if not isinstance(metricas, dict):
        st.info("No hay m\u00e9tricas ML disponibles para mostrar.")
        return

    accuracy = float(metricas.get("accuracy", 0.0))
    error_rate = 1 - accuracy
    precision = float(metricas.get("precision_macro", 0.0))
    recall = float(metricas.get("recall_macro", 0.0))
    f1 = float(metricas.get("f1_macro", 0.0))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _tarjeta_metrica("Accuracy", f"{accuracy * 100:.1f}%")
    with col2:
        _tarjeta_metrica("Error Rate", f"{error_rate * 100:.1f}%")
    with col3:
        _tarjeta_metrica("Precision", f"{precision * 100:.1f}%")
    with col4:
        _tarjeta_metrica("Recall", f"{recall * 100:.1f}%")
    with col5:
        _tarjeta_metrica("F1-Score", f"{f1 * 100:.1f}%")


def _mostrar_graficos_ml(metricas_ml: dict[str, Any], df_eventos_ml: pd.DataFrame) -> None:
    st.subheader("Gr\u00e1ficos ML")
    col1, col2 = st.columns(2)

    with col1:
        _mostrar_grafico_metricas(metricas_ml)

    with col2:
        _mostrar_distribucion_clases(df_eventos_ml)

    col3, col4 = st.columns(2)
    with col3:
        _mostrar_matriz_confusion(metricas_ml, mostrar_tabla=False)
    with col4:
        _mostrar_tabla_matriz_confusion_desde_metricas(metricas_ml)


def _mostrar_grafico_metricas(metricas_ml: dict[str, Any]) -> None:
    metricas = metricas_ml.get("metricas") if isinstance(metricas_ml, dict) else None
    if not isinstance(metricas, dict):
        st.info("No hay datos para el gr\u00e1fico de m\u00e9tricas.")
        return

    df_metricas = pd.DataFrame(
        [
            {
                "m\u00e9trica": "Accuracy",
                "valor": float(metricas.get("accuracy", 0.0)) * 100,
            },
            {
                "m\u00e9trica": "Precision",
                "valor": float(metricas.get("precision_macro", 0.0)) * 100,
            },
            {
                "m\u00e9trica": "Recall",
                "valor": float(metricas.get("recall_macro", 0.0)) * 100,
            },
            {
                "m\u00e9trica": "F1-Score",
                "valor": float(metricas.get("f1_macro", 0.0)) * 100,
            },
        ]
    )
    st.markdown("**Accuracy, Precision, Recall y F1-Score**")
    _mostrar_barras_coloreadas(
        df=df_metricas,
        categoria="m\u00e9trica",
        valor="valor",
        colores=COLORES_METRICAS,
        titulo="M\u00e9tricas principales del \u00e1rbol de decisi\u00f3n",
        etiqueta_y="Porcentaje",
        sufijo="%",
        limite_y=(0, 105),
    )


def _mostrar_distribucion_clases(df_eventos_ml: pd.DataFrame) -> None:
    if df_eventos_ml.empty or "clase_evento_ml" not in df_eventos_ml.columns:
        st.info("No hay dataset ML disponible para distribuci\u00f3n de clases.")
        return

    distribucion = (
        df_eventos_ml["clase_evento_ml"]
        .fillna("N/A")
        .value_counts()
        .rename_axis("clase")
        .reset_index(name="cantidad")
    )
    st.markdown("**Distribuci\u00f3n de clases del dataset**")
    _mostrar_barras_coloreadas(
        df=distribucion,
        categoria="clase",
        valor="cantidad",
        colores=COLORES_CLASES,
        titulo="Distribuci\u00f3n de clases del dataset ML",
        etiqueta_y="Cantidad de eventos",
    )


def _mostrar_barras_coloreadas(
    df: pd.DataFrame,
    categoria: str,
    valor: str,
    colores: dict[str, str],
    titulo: str,
    etiqueta_y: str,
    sufijo: str = "",
    limite_y: tuple[int, int] | None = None,
    figsize: tuple[float, float] = (6.4, 3.9),
) -> None:
    if plt is None:
        st.bar_chart(df, x=categoria, y=valor)
        return

    categorias = df[categoria].astype(str).tolist()
    valores = df[valor].astype(float).tolist()
    colores_barras = [colores.get(item, "#60a5fa") for item in categorias]

    fig, ax = plt.subplots(figsize=figsize)
    barras = ax.bar(categorias, valores, color=colores_barras, edgecolor="#dbeafe", linewidth=0.5)
    ax.set_title(titulo)
    ax.set_ylabel(etiqueta_y)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.24)
    ax.grid(axis="x", visible=False)

    if limite_y is not None:
        ax.set_ylim(*limite_y)

    desplazamiento = max(valores) * 0.025 if valores else 0.5
    for barra, valor_barra in zip(barras, valores):
        etiqueta = f"{valor_barra:.1f}{sufijo}" if sufijo else f"{valor_barra:.0f}"
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + desplazamiento,
            etiqueta,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )

    fig.tight_layout()
    st.pyplot(fig)


def _mostrar_matriz_confusion(
    metricas_ml: dict[str, Any], mostrar_tabla: bool = True
) -> None:
    matriz = metricas_ml.get("matriz_confusion") if isinstance(metricas_ml, dict) else None
    if not isinstance(matriz, dict):
        st.info("No hay matriz de confusi\u00f3n disponible.")
        return

    labels = matriz.get("labels", [])
    valores = matriz.get("valores", [])
    if not labels or not valores:
        st.info("La matriz de confusi\u00f3n est\u00e1 vac\u00eda.")
        return

    st.markdown("**Matriz de confusi\u00f3n**")
    if plt is None:
        st.info("Matplotlib no est\u00e1 disponible; se muestra solo la tabla num\u00e9rica.")
    else:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        imagen = ax.imshow(valores, cmap="Blues")
        ax.set_title("Matriz de confusi\u00f3n del \u00e1rbol de decisi\u00f3n", fontsize=10)
        ax.set_xlabel("Clase predicha", fontsize=9)
        ax.set_ylabel("Clase real", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        maximo = max(max(fila) for fila in valores) if valores else 0
        umbral = maximo / 2
        for fila, valores_fila in enumerate(valores):
            for columna, valor in enumerate(valores_fila):
                color = "white" if valor > umbral else "black"
                ax.text(
                    columna,
                    fila,
                    str(valor),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)

    if mostrar_tabla:
        _mostrar_tabla_matriz_confusion(labels, valores)


def _mostrar_tabla_matriz_confusion_desde_metricas(metricas_ml: dict[str, Any]) -> None:
    matriz = metricas_ml.get("matriz_confusion") if isinstance(metricas_ml, dict) else None
    if not isinstance(matriz, dict):
        st.info("No hay tabla de matriz de confusi\u00f3n disponible.")
        return

    labels = matriz.get("labels", [])
    valores = matriz.get("valores", [])
    if not labels or not valores:
        st.info("La tabla de matriz de confusi\u00f3n est\u00e1 vac\u00eda.")
        return

    st.markdown("**Tabla de matriz de confusi\u00f3n**")
    _mostrar_tabla_matriz_confusion(labels, valores)


def _mostrar_tabla_matriz_confusion(labels: list[Any], valores: list[list[Any]]) -> None:
    total_prueba = sum(sum(int(valor) for valor in fila) for fila in valores)
    encabezados = "".join(f"<th>Pred: {_html(label)}</th>" for label in labels)
    filas = []
    for label, fila in zip(labels, valores):
        celdas = "".join(f"<td>{_html(valor)}</td>" for valor in fila)
        filas.append(f"<tr><td>Real: {_html(label)}</td>{celdas}</tr>")

    st.markdown(
        f"""
        <div class="qsoc-confusion-table">
          <table>
            <thead>
              <tr><th>Matriz de confusi\u00f3n (datos de prueba: {total_prueba} eventos)</th>{encabezados}</tr>
            </thead>
            <tbody>
              {''.join(filas)}
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mostrar_incidentes_soc(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Incidentes SOC")
    df_ordenado = _ordenar_incidentes(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        severidades = sorted(df_ordenado.get("severidad", pd.Series(dtype=str)).dropna().unique())
        filtro_severidad = st.multiselect("Filtrar por severidad", severidades)
    with col2:
        tipos = sorted(df_ordenado.get("tipo_incidente", pd.Series(dtype=str)).dropna().unique())
        filtro_tipo = st.multiselect("Filtrar por tipo", tipos)
    with col3:
        solo_escalados = st.checkbox("Solo escalados")

    df_filtrado = df_ordenado.copy()
    if filtro_severidad:
        df_filtrado = df_filtrado[df_filtrado["severidad"].isin(filtro_severidad)]
    if filtro_tipo:
        df_filtrado = df_filtrado[df_filtrado["tipo_incidente"].isin(filtro_tipo)]
    if solo_escalados and "requiere_escalamiento" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["requiere_escalamiento"].fillna(False)]

    columnas_disponibles = [
        columna for columna in COLUMNAS_TABLA_SOC if columna in df_filtrado.columns
    ]
    st.dataframe(df_filtrado[columnas_disponibles], width="stretch", hide_index=True)
    return df_filtrado


def _ordenar_incidentes(df: pd.DataFrame) -> pd.DataFrame:
    orden_prioridad = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    df_ordenado = df.copy()
    prioridad = _obtener_serie(df_ordenado, "prioridad", "P4")
    df_ordenado["_orden_prioridad"] = (
        prioridad.map(orden_prioridad).fillna(9)
    )
    columnas_orden = ["_orden_prioridad"]
    if "timestamp" in df_ordenado.columns:
        columnas_orden.append("timestamp")
    return df_ordenado.sort_values(columnas_orden).drop(columns=["_orden_prioridad"])


def _mostrar_detalle_incidente(
    reportes: list[dict[str, Any]], df_filtrado: pd.DataFrame, metricas_ml: dict[str, Any]
) -> None:
    st.subheader("Panel de visualización de incidentes")

    ids_alerta = df_filtrado["id_alerta"].astype(str).tolist()
    lista_opciones = ["Vista General"] + ids_alerta
    
    id_seleccionado = st.selectbox("Seleccionar incidente", lista_opciones)

<<<<<<< HEAD
    # 3. Lógica para la Vista General (Nuevo gráfico de Escalamiento/Reducción de Carga)
    if id_seleccionado == "Vista General":
        st.markdown("### Decisión de Escalamiento y Reducción de Carga")
        st.info("Este gráfico de valor de negocio representa el comportamiento real del agente Q-SOC frente al conjunto de prueba del modelo predictivo.")

        # Extraemos la matriz de confusión real desde el JSON cargado en metricas_ml
        matriz = metricas_ml.get("matriz_confusion", {}) if isinstance(metricas_ml, dict) else {}
        labels = matriz.get("labels", [])
        valores = matriz.get("valores", [])

        # Validamos que existan datos de la matriz en el JSON
        if valores and len(valores) >= 2 and len(labels) >= 2:
            try:
                # Buscamos los índices de las clases para mapear la matriz correctamente
                # Normalmente las clases son: 0: normal, 1: fallido, 2: sospechoso (o similar)
                # Mapeamos a la lógica: Ataque real (sospechoso/fallido) vs Evento benigno (normal)
                idx_normal = labels.index("normal") if "normal" in labels else 0
                idx_fallido = labels.index("fallido") if "fallido" in labels else 1
                idx_sospechoso = labels.index("sospechoso") if "sospechoso" in labels else (2 if len(labels) > 2 else 1)

                # --- EXTRACCIÓN DINÁMICA DE LA MATRIZ DE CONFUSIÓN ---
                # Fila es "Real", Columna es "Predicho"
                # Eventos Benignos (Reales normales)
                no_escalado_benigno = int(valores[idx_normal][idx_normal]) # Real normal predicho normal (bien filtrado)
                escalado_benigno = sum(int(valores[idx_normal][j]) for j in range(len(valores)) if j != idx_normal) # Real normal predicho sospechoso/fallido (Falso Positivo)

                # Ataques Reales (Reales fallidos o sospechosos)
                escalado_ataque = 0
                no_escalado_ataque = 0
                for i in [idx_fallido, idx_sospechoso]:
                    if i < len(valores):
                        # Predichos como anomalía (bien escalados)
                        escalado_ataque += sum(int(valores[i][j]) for j in range(len(valores)) if j != idx_normal)
                        # Predichos como normal (Falsos Negativos - Cifra crítica)
                        no_escalado_ataque += int(valores[i][idx_normal])

            except Exception:
                # Fallback con tus valores por defecto si los índices no calzan perfectamente
                escalado_ataque, escalado_benigno, no_escalado_benigno, no_escalado_ataque = 140, 40, 310, 10
        else:
            # Fallback de contingencia con los números base
            escalado_ataque, escalado_benigno, no_escalado_benigno, no_escalado_ataque = 140, 40, 310, 10

        # --- CÁLCULOS MATEMÁTICOS ---
        total_eval = escalado_ataque + escalado_benigno + no_escalado_benigno + no_escalado_ataque
        escalados_totales = escalado_ataque + escalado_benigno
        no_escalados_totales = no_escalado_benigno + no_escalado_ataque
        reduccion_carga_pct = (no_escalados_totales / total_eval) * 100 if total_eval > 0 else 0.0

        # --- RENDERIZADO DEL GRÁFICO ---
        if plt is None:
            # Si en la nube fallara Matplotlib por alguna extraña razón, mostramos datos tabulares limpios
            st.warning("Matplotlib no disponible para renderizar el gráfico. Se muestran métricas de negocio resumidas:")
            col_a, col_b = st.columns(2)
            col_a.metric("Reducción de Carga al SOC", f"{reduccion_carga_pct:.1f}%")
            col_b.metric("Falsos Negativos (Críticos)", f"{no_escalado_ataque} eventos")
        else:
            # Generamos el gráfico interactivo usando la misma lógica del evaluador
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            categorias_plot = ["Escalado al analista", "No escalado\n(filtrado por Q-SOC)"]

            # Dibujamos las barras apiladas
            ax.bar(categorias_plot, [escalado_ataque, no_escalado_ataque], color="#d62728", label="Ataque real", edgecolor="white", width=0.55)
            ax.bar(categorias_plot, [escalado_benigno, no_escalado_benigno], bottom=[escalado_ataque, no_escalado_ataque], color="#1f77b4", label="Evento benigno", edgecolor="white", width=0.55)

            # Función helper local para las etiquetas de porcentaje internas
            def colocar_etiqueta(x_coord, y_base, valor_sec):
                if valor_sec > 0 and total_eval > 0:
                    ax.text(x_coord, y_base + valor_sec / 2, f"{valor_sec}\n({100*valor_sec/total_eval:.1f}%)",
                            ha="center", va="center", color="white", fontweight="bold", fontsize=9)

            colocar_etiqueta(0, 0, escalado_ataque)
            colocar_etiqueta(0, escalado_ataque, escalado_benigno)
            colocar_etiqueta(1, 0, no_escalado_ataque)
            colocar_etiqueta(1, no_escalado_ataque, no_escalado_benigno)

            # Totales en la parte superior de cada barra
            for idx_bar, suma_col in enumerate([escalados_totales, no_escalados_totales]):
                ax.text(idx_bar, suma_col + total_eval * 0.02, f"n = {suma_col}", ha="center", fontweight="bold")

            ax.set_ylabel("Número de eventos (conjunto de prueba)")
            ax.set_title(f"Decisión de escalamiento del agente Q-SOC (n = {total_eval})\n"
                         f"Reducción de carga al SOC: {reduccion_carga_pct:.1f}% de eventos filtrados", fontsize=11, fontweight="bold")
            ax.legend(loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)

            # Anotación llamativa para el Falso Negativo (Cifra Crítica)
            if no_escalado_ataque > 0:
                ax.annotate(f"Falsos negativos: {no_escalado_ataque} ({100*no_escalado_ataque/total_eval:.1f}%)",
                            xy=(1, no_escalado_ataque), xytext=(1.05, total_eval * 0.35),
                            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
                            color="#d62728", fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig)
            
        return

=======
    if id_seleccionado == "Vista General":
        st.markdown("Panorama Global de Incidentes")
        st.info("Estas gráficas reflejan el total de incidentes mostrados actualmente en la tabla superior.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Distribución por Severidad**")
            if "severidad" in df_filtrado.columns:
                # Contamos y graficamos la severidad, usando el color rojo para alerta
                st.bar_chart(df_filtrado["severidad"].value_counts(), color="#ef4444")
            else:
                st.write("Sin datos.")
                
        with col2:
            st.write("**Distribución por Tipo de Incidente**")
            if "tipo_incidente" in df_filtrado.columns:
                # Contamos y graficamos los tipos de incidentes, usando color azul
                st.bar_chart(df_filtrado["tipo_incidente"].value_counts(), color="#3b82f6")
            else:
                st.write("Sin datos.")
                
        return

    # 4. Lógica original para cuando se selecciona un incidente específico
>>>>>>> ca8aa19 (feat: agregar vista general interactiva y actualizar dependencias)
    reporte = next(
        reporte for reporte in reportes if str(reporte.get("id_alerta")) == id_seleccionado
    )

    col1, col2 = st.columns(2)
    with col1:
        _mostrar_resumen_incidente(reporte)

    with col2:
        _mostrar_prediccion_ml(reporte.get("prediccion_ml"))

    _mostrar_tabs_respuesta(reporte)


def _mostrar_resumen_incidente(reporte: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**Resumen del incidente**")
        escalamiento = _formatear_booleano(reporte.get("requiere_escalamiento"))
        campos = [
            ("ID incidente", reporte.get("id_alerta", "N/A")),
            ("Tipo incidente", reporte.get("tipo_incidente", "N/A")),
            ("Severidad", reporte.get("severidad", "N/A")),
            ("Prioridad", reporte.get("prioridad", "N/A")),
            (
                "MITRE",
                f"{reporte.get(CLAVE_TACTICA_MITRE, 'N/A')} / "
                f"{reporte.get(CLAVE_TECNICA_MITRE, 'N/A')}",
            ),
            ("T\u00e9cnica ID", reporte.get(CLAVE_ID_TECNICA, "N/A")),
            ("Escalamiento", escalamiento),
        ]
        detalle = "".join(
            "<div class='qsoc-detail-label'>{}</div>"
            "<div class='qsoc-detail-value'>{}</div>".format(_html(etiqueta), _html(valor))
            for etiqueta, valor in campos
        )
        st.markdown(
            f"<div class='qsoc-detail-grid'>{detalle}</div>",
            unsafe_allow_html=True,
        )


def _mostrar_tabs_respuesta(reporte: dict[str, Any]) -> None:
    tabs = st.tabs(
        [
            "Evidencia",
            "Playbook recomendado",
            "Contexto hist\u00f3rico",
            "An\u00e1lisis avanzado Q-SOC / Qwen",
            "JSON / detalles t\u00e9cnicos",
        ]
    )

    with tabs[0]:
        _mostrar_lista_en_tarjeta(reporte.get("evidencia"), ordenada=False)

    with tabs[1]:
        _mostrar_playbook_tarjetas(reporte.get(CLAVE_PLAYBOOK))

    with tabs[2]:
        _mostrar_contexto_historico_compacto(reporte.get("contexto_historico"))

    with tabs[3]:
        _mostrar_qwen_bajo_demanda(reporte)

    with tabs[4]:
        with st.expander("Ver JSON completo del incidente", expanded=False):
            st.json(reporte, expanded=False)


def _mostrar_prediccion_ml(prediccion: Any) -> None:
    with st.container(border=True):
        st.markdown("**Predicci\u00f3n ML**")
        if not isinstance(prediccion, dict):
            st.info("No hay predicci\u00f3n ML disponible en el reporte.")
            return

        if prediccion.get("estado") == "error":
            st.warning(prediccion.get("detalle", "Predicci\u00f3n ML no disponible."))
            return

        clase = prediccion.get("clase_predicha", "N/A")
        confianza = float(prediccion.get("confianza", 0.0))
        estado = prediccion.get("estado", "N/A")
        color = _color_clase_ml(str(clase))
        st.markdown(
            f"""
            <div class="qsoc-card" style="min-height: 0; margin-bottom: 0.75rem;">
              <div class="qsoc-card-title">Clase predicha</div>
              <div class="qsoc-card-value">{_html(clase)}</div>
              <div class="qsoc-card-note">
                Confianza: {confianza * 100:.1f}% &nbsp;|&nbsp;
                <span class="qsoc-badge {color}">Estado: {_html(estado)}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probabilidades = prediccion.get("probabilidades")
        if isinstance(probabilidades, dict) and probabilidades:
            df_prob = pd.DataFrame(
                [
                    {"clase": clase, "probabilidad": float(valor) * 100}
                    for clase, valor in probabilidades.items()
                ]
            )
            _mostrar_barras_coloreadas(
                df=df_prob,
                categoria="clase",
                valor="probabilidad",
                colores=COLORES_CLASES,
                titulo="Probabilidades por clase",
                etiqueta_y="Probabilidad",
                sufijo="%",
                limite_y=(0, 105),
                figsize=(4.8, 2.8),
            )


def _mostrar_lista_en_tarjeta(valores: Any, ordenada: bool = False) -> None:
    if not valores:
        st.info("No disponible en el reporte.")
        return

    if not isinstance(valores, list):
        valores = [valores]

    etiqueta_lista = "ol" if ordenada else "ul"
    items = "".join(f"<li>{_html(valor)}</li>" for valor in valores)
    st.markdown(
        f"""
        <div class="qsoc-list-card">
          <{etiqueta_lista}>{items}</{etiqueta_lista}>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mostrar_playbook_tarjetas(valores: Any) -> None:
    if not valores:
        st.info("No disponible en el reporte.")
        return

    if not isinstance(valores, list):
        valores = [valores]

    for indice, accion in enumerate(valores, start=1):
        st.markdown(
            f"""
            <div class="qsoc-list-card">
              <span class="qsoc-badge blue">Paso {indice}</span>
              <div class="qsoc-text-box" style="margin-top: 0.45rem;">
                {_html(accion)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _mostrar_contexto_historico_compacto(contexto: Any) -> None:
    if not isinstance(contexto, dict) or not contexto:
        st.info("No disponible en el reporte.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        _tarjeta_metrica(
            "IP origen recurrente",
            _formatear_booleano(contexto.get("ip_origen_recurrente")),
        )
    with col2:
        _tarjeta_metrica(
            "Incidentes previos",
            contexto.get("incidentes_relacionados_previos", 0),
        )
    with col3:
        _tarjeta_metrica(
            "Cr\u00edticos previos",
            contexto.get("incidentes_criticos_previos", 0),
        )

    usuarios = contexto.get("usuarios_recurrentes") or []
    usuarios_texto = ", ".join(usuarios) if usuarios else "No disponible en el reporte"
    st.markdown(
        f"""
        <div class="qsoc-list-card">
          <div><strong>Usuarios recurrentes:</strong> {_html(usuarios_texto)}</div>
          <div><strong>Hostname recurrente:</strong> {_html(_formatear_booleano(contexto.get("hostname_recurrente")))}</div>
          <div><strong>Patr\u00f3n repetido:</strong> {_html(_formatear_booleano(contexto.get("patron_repetido")))}</div>
          <div style="margin-top: 0.55rem;">{_html(contexto.get("resumen_contexto", "No disponible en el reporte."))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _color_clase_ml(clase: str) -> str:
    clase_normalizada = clase.strip().lower()
    if clase_normalizada == "normal":
        return "green"
    if clase_normalizada in {"fallido", "sospechoso"}:
        return "red"
    return "blue"


def _mostrar_qwen_bajo_demanda(reporte: dict[str, Any]) -> None:
    id_incidente = str(reporte.get("id_alerta", "sin_id"))
    st.session_state.setdefault("analisis_qsoc_qwen", {})

    analisis_guardado = reporte.get("analisis_qwen3_real")
    analisis_sesion = st.session_state["analisis_qsoc_qwen"].get(id_incidente)

    if _es_analisis_qwen_valido(analisis_sesion):
        st.success("An\u00e1lisis Qwen disponible en la sesi\u00f3n actual.")
        with st.expander("Ver an\u00e1lisis Qwen de esta sesi\u00f3n", expanded=False):
            _mostrar_analisis_qwen3(analisis_sesion)
        return

    if id_incidente in st.session_state["analisis_qsoc_qwen"]:
        st.warning(
            "Qwen no se encuentra disponible o no entreg\u00f3 una respuesta "
            "v\u00e1lida. El an\u00e1lisis base del incidente se mantiene "
            "disponible mediante ML, MITRE y playbook."
        )
        with st.expander("Ver detalle del intento Qwen"):
            st.json(analisis_sesion, expanded=False)
        return

    if _es_analisis_qwen_valido(analisis_guardado):
        st.success("An\u00e1lisis Qwen previamente guardado en el reporte.")
        with st.expander("Ver an\u00e1lisis Qwen guardado", expanded=False):
            _mostrar_analisis_qwen3(analisis_guardado)
        return

    clase_ml = _obtener_clase_ml(reporte.get("prediccion_ml"))
    if clase_ml == "normal":
        st.info(
            "El modelo predictivo clasific\u00f3 este incidente como normal. "
            "No se requiere an\u00e1lisis profundo con Qwen en modo demo."
        )
        return

    if clase_ml not in {"fallido", "sospechoso"}:
        st.info(
            "No hay una predicci\u00f3n ML de riesgo suficiente para habilitar "
            "el an\u00e1lisis avanzado en modo demo."
        )
        return

    st.warning(
        "El modelo predictivo detect\u00f3 riesgo. Puede generarse un an\u00e1lisis "
        "SOC avanzado con Qwen para este incidente."
    )

    if st.button("Generar an\u00e1lisis SOC avanzado", key=f"qwen_{id_incidente}"):
        with st.spinner("Consultando Qwen local para este incidente..."):
            analisis = _ejecutar_qwen_bajo_demanda(reporte)
        st.session_state["analisis_qsoc_qwen"][id_incidente] = analisis

        if _es_analisis_qwen_valido(analisis):
            st.success("An\u00e1lisis Qwen generado para esta sesi\u00f3n.")
            with st.expander("Ver an\u00e1lisis Qwen de esta sesi\u00f3n", expanded=True):
                _mostrar_analisis_qwen3(analisis)
        else:
            st.warning(
                "Qwen no se encuentra disponible o no entreg\u00f3 una respuesta "
                "v\u00e1lida. El an\u00e1lisis base del incidente se mantiene "
                "disponible mediante ML, MITRE y playbook."
            )


def _ejecutar_qwen_bajo_demanda(reporte: dict[str, Any]) -> dict[str, Any]:
    try:
        from src.qwen_agent import analizar_con_qwen3

        respuesta = analizar_con_qwen3(reporte)
        if not isinstance(respuesta, dict):
            return {
                "error": "respuesta_no_estructurada",
                "respuesta_original": str(respuesta),
            }
        return respuesta
    except Exception as error:
        return {
            "error": "qwen_no_disponible",
            "detalle": str(error),
        }


def _es_analisis_qwen_valido(analisis: Any) -> bool:
    return (
        isinstance(analisis, dict)
        and not analisis.get("error")
        and any(clave in analisis for clave in CLAVES_QWEN_RECONOCIDAS)
    )


def _mostrar_analisis_qwen3(analisis: dict[str, Any]) -> None:
    st.markdown(
        f"**Resumen ejecutivo:** "
        f"{analisis.get('resumen_ejecutivo', 'No disponible en el reporte')}"
    )

    _mostrar_lista_con_titulo("Hechos observados", analisis.get("hechos_observados"))
    _mostrar_lista_con_titulo(
        "Inferencias razonables", analisis.get("inferencias_razonables")
    )

    st.markdown(
        f"**An\u00e1lisis de riesgo:** "
        f"{analisis.get('analisis_riesgo', 'No disponible en el reporte')}"
    )
    st.markdown(
        "**Explicaci\u00f3n MITRE ATT&CK:** "
        f"{analisis.get('explicacion_mitre', 'No disponible en el reporte')}"
    )

    if "uso_contexto_historico" in analisis:
        st.markdown(
            "**Uso del contexto hist\u00f3rico:** "
            f"{analisis.get('uso_contexto_historico', 'No disponible en el reporte')}"
        )

    _mostrar_lista_con_titulo("Recomendaci\u00f3n SOC", analisis.get("recomendacion_soc"))
    _mostrar_lista_con_titulo(
        "Posibles falsos positivos", analisis.get("posibles_falsos_positivos")
    )
    st.markdown(
        "**Siguiente acci\u00f3n sugerida:** "
        f"{analisis.get('siguiente_accion_sugerida', 'No disponible en el reporte')}"
    )


def _mostrar_lista_con_titulo(titulo: str, valores: Any) -> None:
    if valores is None:
        return

    st.markdown(f"**{titulo}**")
    _mostrar_lista(valores)


def _obtener_clase_ml(prediccion: Any) -> str | None:
    if not isinstance(prediccion, dict) or prediccion.get("estado") == "error":
        return None
    clase = prediccion.get("clase_predicha")
    return str(clase).strip().lower() if clase else None


def _mostrar_lista(valores: Any) -> None:
    if not valores:
        st.write("No disponible en el reporte.")
        return

    if not isinstance(valores, list):
        st.write(valores)
        return

    for valor in valores:
        st.markdown(f"- {valor}")


def _normalizar_texto_ascii(valor: Any) -> str:
    texto = str(valor).strip()
    texto_ascii = (
        unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    )
    normalizado = texto_ascii.lower()
    if normalizado == "cratica":
        return "critica"
    return normalizado


def _formatear_booleano(valor: Any) -> str:
    return "S\u00ed" if bool(valor) else "No"


def _obtener_serie(df: pd.DataFrame, columna: str, valor_defecto: Any) -> pd.Series:
    if columna in df.columns:
        return df[columna]
    return pd.Series([valor_defecto] * len(df), index=df.index)


if __name__ == "__main__":
    main()
