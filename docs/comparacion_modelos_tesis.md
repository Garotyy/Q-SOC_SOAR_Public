# Comparación de clasificadores de Q-SOC

Este experimento añade evidencia para comparar Árbol de Decisión, Random Forest y XGBoost. Permite analizar qué errores cambian, cuánto cuesta entrenar y predecir, y qué limitaciones persisten. No presupone que XGBoost vaya a superar a Random Forest.

## Puesta en marcha en Windows

Desde Anaconda Prompt, dentro de SOAR-IA y con `prototipo-soar-ai` activo:

```bat
python -m pip install -r requirements-comparacion.txt
python -m src.comparar_modelos --incluir-rf-alternativo --auditar-incidentes
```

El experimento utiliza `data/dataset_kali_homologado.csv`, que ya está en el computador. No debe regenerarse para esta comparación: cambiar su orden o contenido podría cambiar la partición utilizada anteriormente. Se procesa el CSV completo, sin muestreo de filas.

Cada ejecución crea `outputs/comparacion_modelos/FECHA_HORA/`. La consola muestra la ruta exacta. Para revisar los resultados, enviar `resultados_completos.json` y `comparacion_modelos.csv` de esa carpeta.

La ejecución no llama a `main.py` ni a Qwen. El modelo, las métricas y los reportes oficiales permanecen en sus rutas actuales. El experimento no cambia el dashboard ni selecciona automáticamente el modelo que se integrará.

La instalación añade XGBoost al entorno. Conviene registrar las versiones finales que quedan documentadas en `protocolo.json`. El número de hilos se limita por defecto a cuatro; puede ajustarse con `--hilos 2`.

## Qué se compara

| Nombre de la ejecución | Configuración |
| --- | --- |
| arbol_decision | Estimador creado por el módulo existente: profundidad 3, hoja mínima 20, división mínima 50, ponderación balanced y semilla 42. Se sustituye su preprocesamiento por el común de 12 entradas de Random Forest. |
| random_forest_actual | Estimador creado por el código actual: 100 árboles, profundidad 3, hoja mínima 500, división mínima 12, ponderación balanced y semilla 42. |
| random_forest_alternativo | Variante opcional: mismo bosque, profundidad 12, hoja mínima 5 y división mínima 15. Corresponde a los valores declarados por el JSON histórico; se evalúa como experimento separado, sin atribuirle los resultados anteriores. |
| xgboost | Boosting multiclase con 200 rondas, profundidad 3, learning_rate 0,1, subsample 0,8, colsample_bytree 0,8, regularización L2 de 1, método hist y semilla 42. Usa pesos por muestra derivados exclusivamente del entrenamiento. |

Estas son configuraciones prefijadas. No equivalen a una búsqueda exhaustiva ni a una comparación con idéntico presupuesto computacional. Las rondas de boosting no equivalen al número de árboles de Random Forest en un problema multiclase. Los resultados permitirán comparar las configuraciones ensayadas, no afirmar que un algoritmo es siempre superior.

La variante de RF modifica conjuntamente tres restricciones de crecimiento. Si mejora, podrá discutirse el efecto de esa configuración de capacidad, pero no atribuir toda la diferencia a un único parámetro.

## Protocolo común

1. Seleccionar exactamente las dos variables categóricas y diez numéricas utilizadas por el pipeline actual de Random Forest. Excluir identificadores, etiqueta objetivo, tipo_evento, hostname y severidad_inicial.
2. Aplicar una única separación estratificada 75/25 con semilla 42, sin cambiar el orden del CSV. Guardar los índices y sus hashes para comprobar identidad de muestras.
3. Ajustar el codificador OneHotEncoder exclusivamente con entrenamiento. Usar el mismo preprocesamiento ajustado y matrices float32 en todos los modelos. El vocabulario de las tres etiquetas se fija de antemano.
4. Mantener ponderación balanced para Árbol y RF. Aplicar la misma fórmula a las muestras de entrenamiento de XGBoost. Calcular las métricas sin ponderar sobre el conjunto de prueba.
5. Mantener el conjunto de prueba fuera del ajuste de parámetros, del preprocesamiento y del early stopping. Los parámetros se registran antes de entrenar.
6. Registrar métricas por clase, F1 macro y ponderado, matriz de confusión, errores del filtro, tiempos, parámetros reales, versiones, código y huella del CSV.

La partición ya fue utilizada en la evaluación anterior de la tesis. Esta ampliación es una comparación exploratoria sobre ese mismo conjunto, no una prueba confirmatoria con datos externos nunca examinados. Si se ajustan parámetros después de ver los resultados, deberá utilizarse una validación interna del 75 % de entrenamiento y posteriormente una evaluación independiente. No se debe optimizar repetidamente sobre el 25 % de prueba.

## Hallazgos verificados al revisar el código recibido

### Parámetros declarados y parámetros ejecutados

`src/random_forest.py` crea el bosque con `max_depth=3`, `min_samples_leaf=500` y `min_samples_split=12`. Sin embargo, `_calcular_metricas` escribe literalmente `12`, `5` y `15` en el bloque de parámetros, independientemente del estimador ajustado. El JSON recibido contiene esos valores declarados.

Esta discrepancia no demuestra por sí sola que las cifras de la matriz sean incorrectas. Impide reconstruir el experimento utilizando únicamente los parámetros escritos en el JSON. La comparación nueva usa `get_params()` para registrar la configuración efectiva y contrasta su matriz con la referencia anterior.

El Árbol de Decisión también presenta valores declarados que difieren de su constructor. La comparación utiliza los valores del estimador real y no copia sus metadatos estáticos.

### Variables que realmente recibe el clasificador

Las entradas son:

- Categóricas: usuario y servicio.
- Numéricas: hora_evento, es_horario_laboral, ip_origen_es_privada, usuario_privilegiado, conteo_eventos_ip_ventana, conteo_usuarios_distintos_ip, conteo_hosts_distintos_ip, conteo_fallidos_ventana, host_critico y servicio_sensible.

El apartado D.2 del informe recibido enumera doce variables originales de CICIDS2017. Esa descripción no coincide con el pipeline del código recibido y tendrá que corregirse al actualizar la metodología y el anexo.

En `procesar_kali_dataset.py`, siete de las doce entradas se fijan por construcción: usuario = network_system; hora_evento = 12; es_horario_laboral = 1; ip_origen_es_privada = 1; usuario_privilegiado = 0; conteo_usuarios_distintos_ip = 1; conteo_hosts_distintos_ip = 1. La auditoría comprobará qué columnas son efectivamente constantes en el CSV utilizado.

Las restantes se construyen a partir del puerto de destino, la duración del flujo y una tasa de paquetes hacia atrás. En particular:

- conteo_eventos_ip_ventana representa `int(log1p(Flow Duration))`, no un conteo observado de eventos por IP.
- conteo_fallidos_ventana procede de una tasa de paquetes reversos, truncada a 100, no de intentos de autenticación fallidos observados.
- host_critico se asigna según el puerto de destino, no según un inventario de activos.

Estas diferencias tienen relevancia para la integración: el adaptador de incidentes sí utiliza recuentos de eventos e intentos de autenticación. Un mismo nombre de variable no garantiza la misma definición ni distribución. Cambiar de algoritmo conservando esta representación no corrige automáticamente esa diferencia semántica.

### Prevención de fuga de etiquetas

El homologador deriva `tipo_evento` y `severidad_inicial` directamente de las etiquetas. El antiguo pipeline del Árbol incluía esas columnas; por eso no se ejecuta ese entrenamiento antiguo para compararlo con CICIDS2017. Solo se reutiliza su estimador, al que se entrega el mismo preprocesamiento sin esas columnas que reciben RF y XGBoost.

Esta observación describe un riesgo que aparecería al aplicar ese pipeline antiguo directamente al CSV homologado; no demuestra que la evaluación histórica del árbol sobre sus datos simulados tuviera exactamente la misma fuga.

## Archivos que produce el experimento

| Archivo | Uso |
| --- | --- |
| resultados_completos.json | Resumen completo para revisar la ejecución y preparar la tesis. |
| comparacion_modelos.csv | Tabla comparativa con métricas, FP, FN, filtrado y tiempos. |
| metricas_MODELO.json | Métricas de entrenamiento y prueba, matriz y parámetros efectivos de cada configuración. |
| protocolo.json | Semilla, partición, huellas, entorno y configuración previa al entrenamiento. |
| indices_particion.npz | Posiciones del CSV empleadas en entrenamiento y prueba. |
| predicciones_prueba.csv.gz | Predicciones de todos los modelos sobre las mismas filas de prueba. |
| muestra_falso_negativo_MODELO.csv | Hasta 50 errores de omisión con sus variables de entrada. Es una muestra, no el total de errores. |
| muestra_falso_positivo_MODELO.csv | Hasta 50 falsas alertas con sus variables de entrada. |
| diagnostico.json | Constantes, vectores ambiguos, comparación pareada y contraste con el JSON histórico. |
| predicciones_incidentes_exploratorias.csv | Diagnóstico opcional de los incidentes reconstruidos, sin Qwen ni modificación de reportes. |
| figuras/ | Gráficos separados de F1, FP, FN y matrices, en PNG y PDF. |

Opcionalmente, `--guardar-modelos` conserva los objetos experimentales en la carpeta de la ejecución. Se guardan junto con el preprocesador y el codificador de etiquetas; no son reemplazos directos del archivo productivo esperado por `ml_predictor.py`.

La auditoría de vectores idénticos se interpreta con cautela: distintos flujos pueden terminar con las mismas variables después de la homologación. Esto no prueba identidad de eventos ni fuga entre particiones. Si un mismo vector tiene etiquetas diferentes, evidencia una limitación de la representación para separarlos de manera determinista. La cota empírica de accuracy que se calcula con esos grupos no es una métrica de generalización ni una estimación del F1 máximo.

## Cómo transformar los resultados en análisis crítico

Primero, verificar si el RF actual reproduce la matriz anterior y si la población coincide. Si difiere, revisar el hash y el orden del CSV, las versiones y la configuración antes de mezclar cifras antiguas y nuevas.

Después, analizar las configuraciones en conjunto:

- ¿Aumenta F1 macro y mejora la clase sospechoso, o solo aumenta accuracy por la clase normal?
- ¿Disminuyen los ataques omitidos? ¿Cuántas falsas alertas nuevas aparecen como contrapartida?
- ¿Qué ataques que detectaba RF deja de detectar la alternativa, aunque su total global mejore?
- ¿El RF alternativo mejora con respecto al actual? ¿La diferencia aparece también al compararlo con XGBoost?
- ¿Las métricas de entrenamiento y prueba muestran resultados bajos en ambos subconjuntos o una brecha de generalización? Una métrica aislada del 100 % no diagnostica sobreajuste.
- ¿Persisten errores concentrados en las mismas combinaciones de variables? ¿Los vectores ambiguos o las constantes limitan la información disponible?
- ¿Las entradas de los incidentes presentan categorías no vistas o valores fuera del rango de entrenamiento? Una mejora sobre flujos no valida automáticamente el uso sobre incidentes.

En Resultados se incorporan los valores y ejemplos observados. En Discusión se contrastan los cambios con la expectativa de reducir sobrecarga y se explican los compromisos entre detección y falsas alarmas. La conclusión debe basarse en las configuraciones realmente ensayadas y mantener las limitaciones de integración.

La comparación se realiza después de la validación inicial. Debe describirse como una evaluación complementaria y no como si XGBoost hubiera formado parte desde el principio del prototipo, de sus 27 reportes o de la prueba de tiempo.

## Apartados del informe que necesitarán actualización

- 2.3.1: fundamento del Árbol, Random Forest y XGBoost y justificación de la comparación.
- 3.3.1 y 3.4: variables reales, protocolo comparativo, tratamiento del desbalance y alcance exploratorio.
- 4.2: tabla y figuras comparativas; conservar la versión anterior como referencia identificada.
- 5.1: análisis de los errores recuperados e introducidos y del efecto de las configuraciones.
- 5.2: diferenciar el resultado sobre flujos del comportamiento del adaptador de incidentes.
- 5.5, 6 y Anexo D: limitaciones, conclusiones respaldadas por las nuevas mediciones y parámetros/variables efectivos.

El objetivo específico actual menciona integrar Random Forest. La comparación complementaria puede documentarse conservando ese prototipo de referencia. Si después se decide integrar XGBoost, será necesario revisar el objetivo y volver a evaluar el flujo que cambie; no trasladar automáticamente los resultados previos a la nueva versión.

## Referencias técnicas para reproducir la implementación

- XGBoost, interfaz scikit-learn: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
- XGBoost, parámetros: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost, Python API y sample_weight: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- scikit-learn, ponderación de muestras: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html

Las pruebas del paquete usan entradas artificiales únicamente para verificar el código. Sus métricas no deben utilizarse como resultados de CICIDS2017 ni incorporarse a la tesis.
