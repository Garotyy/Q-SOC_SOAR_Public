# Prueba de representación de entrada para Q-SOC

La comparación anterior mostró una mejora al cambiar la configuración de Random Forest, pero persistieron 43.964 falsas alarmas y 3.348 ataques omitidos. El diagnóstico confirmó siete columnas constantes, solo 1.487 vectores de entrada diferentes y 313 vectores con etiquetas distintas. Estos hallazgos justifican comprobar si conservar más información de los flujos originales mejora el filtrado.

La temperatura de Qwen ya está configurada en 0,0 en el código recibido. Ese parámetro modifica la generación del LLM, no las predicciones de Random Forest o XGBoost. No elimina por sí mismo las alucinaciones ni corrige los falsos positivos del clasificador.

## Ejecución

Con el entorno y directorio del proyecto ya activos:

```bat
python -m src.comparar_representaciones
```

Se requiere el módulo comparar_modelos de la entrega anterior, el CSV homologado existente, los tres CSV originales de lunes, martes y miércoles en data, y la referencia resultados_completos_qsoc.json en Descargas. Se utiliza el entorno actual; no hace falta instalar otra dependencia.

Si la referencia estuviera en otra ubicación:

```bat
python -m src.comparar_representaciones --referencia "RUTA_COMPLETA_DEL_JSON"
```

## Control de la comparación

El programa lee las capturas originales por bloques. Reproduce la limpieza del homologador sobre todas las columnas, incluso las que no serán seleccionadas. Reconstruye las doce entradas homologadas y la etiqueta únicamente para verificar la identidad de cada fila. Comprueba los posibles órdenes de los tres archivos y acepta solo un orden cuya secuencia coincida con el CSV utilizado en el experimento anterior.

Si cambió el tamaño, los valores homologados, las etiquetas o los índices de la partición, se detiene antes de entrenar. La igualdad de filas se verifica con hashes de 64 bits de las entradas y la etiqueta, no mediante identificadores únicos de eventos; por tanto, sigue existiendo la posibilidad teórica de colisiones y no se afirma identidad de eventos a partir de las variables por sí solas. También se conservan hashes de los archivos originales y la procedencia de cada muestra.

Se comprueban la huella del CSV homologado, los módulos originales y las versiones de scikit-learn y XGBoost. La partición se mantiene estratificada 75/25 con semilla 42 y se verifican sus hashes contra la ejecución de referencia.

La variable que cambia es la representación de entrada. Se mantienen RF alternativo con profundidad 12, hoja mínima 5 y división mínima 15, y XGBoost con la configuración prefijada en la primera prueba. Los pesos se calculan exclusivamente a partir del entrenamiento y las métricas de prueba se mantienen sin ponderar.

Las doce variables de red son Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Mean, Bwd Packet Length Mean, Flow Bytes/s, Flow Packets/s, SYN Flag Count y ACK Flag Count. No se utilizan Label, el archivo o el día de captura como entradas del clasificador. Flow Duration conserva el tratamiento no negativo del homologador anterior.

Los resultados de referencia de ambas configuraciones se toman del JSON anterior después de las verificaciones. Se entrenan solamente sus versiones con variables de red. Se comparan F1 macro, métricas de entrenamiento/prueba, FP, FN, precisión de alertas y filtrado.

## Interpretación

El propósito es comprobar si recuperar información mejora la clasificación sobre los mismos registros. No se presupone un ganador ni se ajustan parámetros o umbrales utilizando el conjunto de prueba. Esta sigue siendo una ampliación exploratoria sobre una partición que ya ha sido examinada; una validación externa o temporal seguirá siendo necesaria para estudiar generalización.

Una reducción de FP debe contrastarse con los ataques omitidos, el comportamiento por clase y el costo computacional. El resultado no debe aceptarse solo porque aumente el filtrado.

Los 313 vectores con etiquetas diferentes son una evidencia de ambigüedad en la representación multiclase. El diagnóstico agregado no demuestra que todos esos grupos mezclen específicamente tráfico benigno y ataques, ni atribuye cada falso positivo a esa ambigüedad. Esta prueba permitirá contrastar el efecto de cambiar la representación conservando los estimadores.

## Alcance de la integración

El clasificador con variables de red requiere telemetría de flujo real. Los actuales incidentes consolidados no contienen automáticamente todas esas mediciones; no sería correcto rellenarlas con valores inventados y asumir que el desempeño se mantiene.

Esta prueba no sustituye el archivo modelo_random_forest.joblib, no modifica el predictor ni los reportes oficiales, no llama a Qwen y no actualiza automáticamente el dashboard. Si se decide integrar un modelo de red, habrá que definir su punto de entrada de telemetría y volver a evaluar el flujo integrado. Una mejora sobre flujos no constituye por sí sola una reducción medida del cuello de botella en un SOC operativo.

## Entregables de la ejecución

Cada ejecución crea una carpeta nueva bajo outputs/comparacion_representaciones. Enviar resultados_representacion.json al terminar. El archivo incluye la verificación de cohorte, la configuración y las métricas de ambas representaciones. También se generan comparacion_representaciones.csv, procedencia_muestras.csv.gz y predicciones_variables_red.csv.gz.

La entrega incluye pruebas con datos artificiales para verificar el funcionamiento. Sus métricas no representan resultados de CICIDS2017 y no deben incorporarse a la tesis.
