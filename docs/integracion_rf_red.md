# Integración del Random Forest seleccionado en Q-SOC

La ampliación conecta el RF de doce variables originales con el predictor y el flujo de reportes de Q-SOC. Conserva la ruta contextual para incidentes que no contienen mediciones de red. No atribuye las métricas del clasificador de flujos a los incidentes sin esas mediciones.

## Instalación

Desde la raíz de SOAR-IA, con el entorno activo y el ZIP extraído:

```bat
python instalar_integracion_red.py
python -m src.activar_rf_red --experimento "outputs\comparacion_final_red\20261104T213643_357988Z"
```

El instalador verifica las versiones revisadas de main.py, src/ml_predictor.py, src/detector.py y src/clasificador.py antes de escribir. Respalda esos archivos y .gitignore cuando corresponda. Si alguno cambió, se detiene antes de modificar el proyecto. Puede previsualizarse con --comprobar. La carpeta de respaldo queda en respaldos_integracion_red; contiene los originales con sus rutas relativas. Para deshacer la instalación pueden recuperarse esos originales. Los archivos nuevos quedan enumerados en archivos.json del respaldo.

La activación requiere las entregas de comparación anteriores, los tres CSV originales en data, el CSV homologado sin cambios y resultados_completos_qsoc.json en Descargas. Reconstruye la misma cohorte, verifica sus capturas e índices y entrena únicamente RF con los parámetros seleccionados. Guarda modelo_random_forest_red.joblib y su ficha JSON, por separado del modelo contextual. Antes de aceptarlo comprueba que el artefacto serializado reproduce, fila por fila, las predicciones y la matriz de referencia. Si ya existe el artefacto de esta misma referencia, lo reutiliza.

## Qué se valida

Todos los flujos de prueba pasan por el predictor persistido y la política del lote. Se selecciona una muestra explícita de 90 casos por predicción, sin usar la etiqueta real, para ejecutar detectar_incidentes, clasificación, severidad, escalamiento, reporte y contexto mediante main.ejecutar_pipeline. Se comprueba que esas predicciones y decisiones coinciden con el lote. La validación no consulta al LLM y conserva los reportes históricos.

Los resultados se escriben en una carpeta nueva bajo outputs/integracion_red. resumen_integracion_red.json contiene totales, métricas reproducidas, huellas y alcance; predicciones_red.csv.gz conserva todas las predicciones; cola_revision_red.csv.gz conserva las revisiones sugeridas; reportes_red.json es la muestra detallada. No se afirma que se generó el flujo contextual completo para los 416.633 registros ni se confunden flujos con incidentes SOC consolidados.

## Panel y nuevas entradas

```bat
streamlit run dashboard_red.py
```

El panel original dashboard.py mantiene sus fuentes. El panel de red muestra los resultados nuevos, distingue evaluación histórica de estadísticas de una nueva entrada y permite solicitar el análisis avanzado con Q-SOC bajo demanda. El servicio generativo conserva su configuración actual. Las respuestas solicitadas desde el panel permanecen en la sesión.

Un CSV diferente puede procesarse con:

```bat
python -m src.entrada_red --csv "RUTA_DEL_CSV"
```

Este comando no carga Label y no calcula exactitud sin etiquetas independientes de evaluación. Reconoce opcionalmente Source IP/ip_origen, Destination IP/ip_destino, Timestamp/timestamp, hostname, usuario y severidad. La clasificación depende exclusivamente de las doce mediciones. IP, usuario y fecha no se inventan cuando faltan. No se generan técnicas MITRE específicas a partir de la clase normal/fallido/sospechoso.

Los incidentes JSON enriquecidos pueden incluir telemetria_red con un objeto de mediciones o flujos_red con una lista de mediciones. Al agrupar alertas se conservan sus flujos por separado. La clase del incidente se resume por la mayor prioridad sospechoso > fallido > normal; su probabilidad corresponde al flujo elegido, no a una confianza calibrada del incidente completo. Esta agregación requiere validación específica y no hereda automáticamente el F1 del clasificador de flujos.

## Reglas y límites

Una medición ausente o inválida no se sustituye por cero ni se convierte en normal: queda como error y requiere revisión. Se conserva el recorte no negativo de Flow Duration y la conversión float32 empleados en el experimento. Una clase normal no suprime criterios independientes de revisión, incluidos severidad alta/crítica y recurrencia de origen cuando esa información existe. Se corrige la normalización de mayúsculas y acentos al decidir el análisis avanzado. En lotes sin IP no puede validarse correlación de identidad; una coincidencia por falta de datos no se trata como historial.

La política propone revisión humana, sin ejecutar contención, bloqueos ni cambios de credenciales. La severidad de un flujo sin contexto se deja como No determinada. Las predicciones ML son indicadores para revisión y no demuestran por sí solas fuerza bruta, exfiltración o técnicas MITRE.

La reproducción del test confirma la conexión técnica del modelo seleccionado. No es una nueva validación externa, no demuestra generalización temporal y no mide una reducción operativa del cuello de botella. La comparación con tiempos humanos históricos mantiene su población y no se traslada al panel de red.

Las pruebas automatizadas usan casos técnicos pequeños y no constituyen resultados científicos. Se comprueban etiquetas ignoradas en inferencia, calidad de entrada, rutas contextual/red, reglas independientes, conservación de fuentes, persistencia, reportes y panel. Los códigos experimentales anteriores conservan sus huellas históricas; sus guardas pueden rechazar una repetición si el predictor ya fue modificado por esta instalación. Para reproducir íntegramente una versión histórica se conserva su revisión de Git o el respaldo original.
