# Auditoría descriptiva de errores de Q-SOC

Ejecutar después de comparar_representaciones, conservando los archivos de su carpeta de salida y los tres CSV originales en data:

```bat
python -m src.auditar_errores_red --experimento "outputs\comparacion_representaciones\20261104T202052_028236Z"
```

El auditor no entrena, no cambia el predictor, no modifica los resultados previos ni consulta a Qwen. Crea una carpeta nueva en outputs/auditoria_errores_red. Comprueba los índices de prueba y reproduce las matrices desde las predicciones guardadas; verifica las huellas de las capturas originales antes de leer sus variables.

Genera auditoria_errores_red.json con errores por etiqueta original, captura y puerto; distribuciones de las doce variables para aciertos y errores de cada clase; casos que falla un modelo y recupera el otro; y proporción de vectores de prueba presentes en entrenamiento. La igualdad de vectores se estima con hashes de 64 bits sobre las entradas float32. No demuestra que sean el mismo evento ni que exista fuga de información.

También conserva los casos mal clasificados, con variables originales y procedencia, en casos_mal_clasificados.csv.gz. Enviar el JSON para revisar el diagnóstico. Si se necesitan ejemplos específicos después, utilizar ese archivo de casos.

Esta es una auditoría exploratoria: los patrones observados orientan hipótesis para validar por separado. No deben utilizarse estos mismos datos de prueba para ajustar umbrales y presentar después la evaluación como independiente. Las probabilidades y el modelo experimental no se guardaron en la ejecución anterior; este auditor no inventa puntuaciones de confianza ni calcula calibración.

Random Forest sigue como candidato principal según el resultado agregado. La etiqueta fallido requiere atención específica: XGBoost omitió menos ejemplos de esa clase en esta partición, aunque tuvo más falsas alarmas y omisiones totales. Esto justifica comprobar cuáles fallan ambos modelos antes de proponer combinaciones o cambios.
