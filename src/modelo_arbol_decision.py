"""Entrenamiento aislado de un arbol de decision para SOAR-AI.

Este modulo no participa todavia en el pipeline principal. Entrena un modelo
pequeno para clasificar eventos simulados como normal, fallido o sospechoso y
guarda metricas de evaluacion para revision academica.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.generador_dataset_ml import RUTA_DATASET_ML, generar_dataset_ml
from src.utils import DATA_DIR


RANDOM_STATE = 42
RUTA_METRICAS = DATA_DIR / "metricas_arbol_decision.json"
CLASES_OBJETIVO = ["normal", "fallido", "sospechoso"]
COLUMNA_OBJETIVO = "clase_evento_ml"
COLUMNAS_CATEGORICAS = [
    "tipo_evento",
    "servicio",
    "usuario",
    "hostname",
    "severidad_inicial",
]
COLUMNAS_NUMERICAS = [
    "hora_evento",
    "es_horario_laboral",
    "ip_origen_es_privada",
    "usuario_privilegiado",
    "host_critico",
    "conteo_eventos_ip_ventana",
    "conteo_usuarios_distintos_ip",
    "conteo_hosts_distintos_ip",
    "conteo_fallidos_ventana",
    "servicio_sensible",
]
COLUMNAS_EXCLUIDAS = {
    "id_evento",
    "timestamp",
    "ip_origen",
    "ip_destino",
    "resultado",
    COLUMNA_OBJETIVO,
}


def entrenar_arbol_decision(
    ruta_dataset: str | Path = RUTA_DATASET_ML,
    ruta_metricas: str | Path = RUTA_METRICAS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Entrena el arbol de decision y guarda metricas reproducibles."""
    ruta_dataset = Path(ruta_dataset)
    if not ruta_dataset.exists():
        generar_dataset_ml(ruta_salida=ruta_dataset)

    datos = pd.read_csv(ruta_dataset)
    X, y, columnas_entrenamiento = _separar_features_objetivo(datos)
    _validar_columnas_entrenamiento(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                _crear_one_hot_encoder(),
                COLUMNAS_CATEGORICAS,
            ),
            ("numericas", "passthrough", COLUMNAS_NUMERICAS),
        ]
    )

    modelo = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=25,
        min_samples_split=50,
        random_state=random_state,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ]
    )
    pipeline.fit(X_train, y_train)

    predicciones = pipeline.predict(X_test)
    metricas = _calcular_metricas(
        y_test=y_test,
        predicciones=predicciones,
        columnas_entrenamiento=columnas_entrenamiento,
        columnas_categoricas=COLUMNAS_CATEGORICAS,
        columnas_numericas=COLUMNAS_NUMERICAS,
        total_eventos=len(datos),
        total_entrenamiento=len(X_train),
        total_prueba=len(X_test),
    )
    _guardar_metricas(metricas, ruta_metricas)
    return metricas


def _separar_features_objetivo(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if COLUMNA_OBJETIVO not in datos.columns:
        raise ValueError(f"El dataset no contiene la columna objetivo {COLUMNA_OBJETIVO}.")

    columnas_entrenamiento = [
        columna for columna in datos.columns if columna not in COLUMNAS_EXCLUIDAS
    ]
    X = datos[columnas_entrenamiento].copy()
    y = datos[COLUMNA_OBJETIVO].copy()
    return X, y, columnas_entrenamiento


def _validar_columnas_entrenamiento(datos: pd.DataFrame) -> None:
    """Verifica que el dataset tenga las columnas esperadas por el pipeline ML."""
    columnas_requeridas = set(COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS)
    faltantes = sorted(columnas_requeridas.difference(datos.columns))
    if faltantes:
        raise ValueError(
            "El dataset no contiene columnas requeridas para entrenamiento: "
            + ", ".join(faltantes)
        )


def _crear_one_hot_encoder() -> OneHotEncoder:
    """Crea un encoder compatible con versiones nuevas y anteriores de sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _calcular_metricas(
    y_test: pd.Series,
    predicciones: Any,
    columnas_entrenamiento: list[str],
    columnas_categoricas: list[str],
    columnas_numericas: list[str],
    total_eventos: int,
    total_entrenamiento: int,
    total_prueba: int,
) -> dict[str, Any]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        predicciones,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test,
        predicciones,
        average="weighted",
        zero_division=0,
    )

    return {
        "modelo": "DecisionTreeClassifier",
        "objetivo": COLUMNA_OBJETIVO,
        "clases": CLASES_OBJETIVO,
        "total_eventos": total_eventos,
        "total_entrenamiento": total_entrenamiento,
        "total_prueba": total_prueba,
        "parametros_modelo": {
            "max_depth": 4,
            "min_samples_leaf": 25,
            "min_samples_split": 50,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        },
        "features_usadas": columnas_entrenamiento,
        "features_categoricas": columnas_categoricas,
        "features_numericas": columnas_numericas,
        "features_excluidas": sorted(COLUMNAS_EXCLUIDAS),
        "metricas": {
            "accuracy": accuracy_score(y_test, predicciones),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
        },
        "matriz_confusion": {
            "labels": CLASES_OBJETIVO,
            "valores": confusion_matrix(
                y_test,
                predicciones,
                labels=CLASES_OBJETIVO,
            ).tolist(),
        },
        "classification_report": classification_report(
            y_test,
            predicciones,
            labels=CLASES_OBJETIVO,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_texto": classification_report(
            y_test,
            predicciones,
            labels=CLASES_OBJETIVO,
            zero_division=0,
        ),
    }


def _guardar_metricas(metricas: dict[str, Any], ruta_metricas: str | Path) -> None:
    ruta = Path(ruta_metricas)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    resultado = entrenar_arbol_decision()
    print(f"Metricas guardadas en: {RUTA_METRICAS}")
    print(f"Accuracy: {resultado['metricas']['accuracy']:.4f}")
    print(f"F1 macro: {resultado['metricas']['f1_macro']:.4f}")
