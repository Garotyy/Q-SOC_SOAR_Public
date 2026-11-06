# Completar la comparación de tres modelos

Esta ampliación evalúa el árbol de decisión base con las mismas doce variables originales de red, registros e índices utilizados por Random Forest alternativo y XGBoost. Reutiliza sus resultados y predicciones; solo entrena el árbol. No altera el modelo integrado, los reportes ni el dashboard, y no consulta a Qwen.

```bat
python -m src.completar_comparacion_red --experimento "outputs\comparacion_representaciones\20261104T202052_028236Z"
```

Se necesitan las dos entregas de comparación anteriores, los tres CSV originales en data, el CSV homologado sin cambios, la carpeta del experimento anterior y resultados_completos_qsoc.json en Descargas. No hacen falta dependencias adicionales. Si la referencia inicial está en otra ubicación, utilizar --referencia-inicial "RUTA_COMPLETA".

Se verifican huellas de fuentes y capturas, reconstrucción de cohorte, índices y etiquetas de prueba, versión de scikit-learn y coincidencia de la configuración del árbol con su primera evaluación. Se reutiliza exclusivamente su estimador; no el preprocesador de incidentes, porque ahora las entradas son numéricas de red. La configuración base permanece con profundidad máxima 3, hoja mínima 20, división mínima 50, ponderación balanced y semilla 42. No se ajustan parámetros para mejorar el resultado de prueba.

Genera resultados_tres_modelos_red.json, comparacion_tres_modelos.csv y predicciones_tres_modelos.csv.gz en una carpeta nueva bajo outputs/comparacion_final_red. También informa métricas para flujos con un vector de entrada presente en entrenamiento y para los que tienen un vector ausente. El indicador se calcula con hashes de 64 bits de las doce mediciones convertidas a float32, sin etiqueta.

Esta separación descriptiva no equivale a una validación externa ni demuestra independencia de conexiones o escenarios. El F1 macro conserva las tres clases; revisar los soportes antes de interpretar los subconjuntos. Los experimentos sobre esta partición ya examinada son exploratorios. La elección de configuraciones o umbrales futuros requiere validación separada y confirmación con datos reservados.

Los tiempos son mediciones de una ejecución, no tiempos del analista ni del flujo completo de Q-SOC. No se selecciona ni integra automáticamente ningún modelo. Enviar resultados_tres_modelos_red.json para elaborar la comparación final y su justificación.
