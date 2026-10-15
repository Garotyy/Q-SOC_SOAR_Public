"""Entrenamiento de Random Forest sobre el Dataset Real CICIDS2017 (Kali) para SOAR-AI.

Este módulo entrena un modelo de ensamble (Random Forest) utilizando la telemetría
procesada de Kali Linux/CICIDS2017. Guarda tanto el pipeline completo
(preprocesamiento + clasificador) como las métricas de evaluación.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
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
from sklearn.ensemble import RandomForestClassifier

from src.utils import DATA_DIR


RANDOM_STATE = 42
# Apuntamos directamente al dataset homologado generado desde Kali
RUTA_DATASET_KALI = DATA_DIR / "dataset_kali_homologado.csv"
RUTA_METRICAS = DATA_DIR / "metricas_random_forest.json"
RUTA_MODELO = DATA_DIR / "modelo_random_forest.joblib"

CLASES_OBJETIVO = ["normal", "fallido", "sospechoso"]
COLUMNA_OBJETIVO = "clase_evento_ml"

COLUMNAS_CATEGORICAS = [
    "usuario",
    "servicio", 
]

COLUMNAS_NUMERICAS = [
    "hora_evento",
    "es_horario_laboral",
    "ip_origen_es_privada",
    "usuario_privilegiado",
    "conteo_eventos_ip_ventana",
    "conteo_usuarios_distintos_ip",
    "conteo_hosts_distintos_ip",
    "conteo_fallidos_ventana",
    "host_critico",      
    "servicio_sensible",
]

COLUMNAS_EXCLUIDAS = {
    "id_evento",
    "timestamp",
    "ip_origen",
    "ip_destino",
    "resultado",
    COLUMNA_OBJETIVO,
    "tipo_evento",        
    "severidad_inicial",   
    "hostname",            
}


def entrenar_random_forest(
    ruta_dataset: str | Path = RUTA_DATASET_KALI,
    ruta_metricas: str | Path = RUTA_METRICAS,
    random_state: int = RANDOM_STATE,
    ruta_modelo: str | Path = RUTA_MODELO,
) -> dict[str, Any]:
    """Entrena el Random Forest y persiste el modelo y las métricas."""
    ruta_dataset = Path(ruta_dataset)
    if not ruta_dataset.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {ruta_dataset}.\n"
            "Ejecuta primero: python src\\procesar_kali_dataset.py"
        )

    print(f"Cargando datos reales desde: {ruta_dataset.name}...")
    datos = pd.read_csv(ruta_dataset)
    
    X, y, columnas_entrenamiento = _separar_features_objetivo(datos)
    _validar_columnas_entrenamiento(X)

    print(f"Dividiendo dataset de {len(datos)} registros (Train: 75%, Test: 25%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    print("Entrenando ensamble Random Forest (100 árboles de decisión)...")
    pipeline = crear_pipeline_random_forest(random_state=random_state)
    pipeline.fit(X_train, y_train)
    _guardar_modelo(pipeline, ruta_modelo)

    print("Evaluando modelo sobre conjunto de prueba...")
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


def crear_pipeline_random_forest(random_state: int = RANDOM_STATE) -> Pipeline:
    """Crea el pipeline reutilizable de preprocesamiento y Random Forest ML."""
    # NOTA: Eliminamos el ColumnTransformer tradicional o nos aseguramos de que
    # no fuerce arrays densos. Al usar matrices dispersas, el pipeline vuela.
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                _crear_one_hot_encoder(),
                COLUMNAS_CATEGORICAS,
            ),
            ("numericas", "passthrough", COLUMNAS_NUMERICAS),
        ],
        sparse_threshold=0.3, # Permite que el preprocesador mantenga todo disperso
    )

    modelo = RandomForestClassifier(
        n_estimators=100,          
        max_depth=3,              
        min_samples_leaf=500,
        min_samples_split=12,
        random_state=random_state,
        class_weight="balanced",   
        n_jobs=-1                  
    )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ]
    )


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
    """Crea un encoder que genera matrices dispersas para optimizar RAM."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

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
    clases_presentes = sorted(list(set(y_test) | set(predicciones)))
    
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
        "modelo": "RandomForestClassifier (CICIDS2017 Kali)",
        "objetivo": COLUMNA_OBJETIVO,
        "clases": clases_presentes,
        "total_eventos": total_eventos,
        "total_entrenamiento": total_entrenamiento,
        "total_prueba": total_prueba,
        "parametros_modelo": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "min_samples_split": 15,
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
            "labels": clases_presentes,
            "valores": confusion_matrix(
                y_test,
                predicciones,
                labels=clases_presentes,
            ).tolist(),
        },
        "classification_report": classification_report(
            y_test,
            predicciones,
            labels=clases_presentes,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_texto": classification_report(
            y_test,
            predicciones,
            labels=clases_presentes,
            zero_division=0,
        ),
    }


def _guardar_metricas(metricas: dict[str, Any], ruta_metricas: str | Path) -> None:
    ruta = Path(ruta_metricas)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False)


def _guardar_modelo(pipeline: Pipeline, ruta_modelo: str | Path) -> None:
    """Guarda el pipeline ajustado para inferir sin volver a entrenar."""
    ruta = Path(ruta_modelo)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta)
    print(f"Modelo entrenado guardado en: {ruta}")


if __name__ == "__main__":
    resultado = entrenar_random_forest()
    print(f"Modelo disponible en: {RUTA_MODELO}")
    print(f"\nMétricas guardadas con éxito en: {RUTA_METRICAS}")
    print(f"Accuracy del Bosque (Data Real Kali): {resultado['metricas']['accuracy'] * 100:.2f}%")
    print(f"F1 macro del Bosque (Data Real Kali): {resultado['metricas']['f1_macro'] * 100:.2f}%")
