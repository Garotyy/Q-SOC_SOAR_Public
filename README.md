# SOAR-AI

SOAR-AI es un MVP académico de un sistema SOAR semi-automatizado con preparación para IA agéntica. El prototipo analiza alertas simuladas, detecta posibles incidentes, clasifica actividad SSH sospechosa, mapea MITRE ATT&CK, calcula severidad, recomienda playbooks, simula escalamiento SOC e intenta enriquecer cada reporte con Qwen3 ejecutado localmente mediante Ollama.

El sistema no ejecuta acciones reales sobre red. Todas las respuestas son recomendaciones o simulaciones para un entorno académico.

## Caso MVP

El primer caso implementado es la detección de ataques de fuerza bruta SSH.

El diseño queda preparado para agregar detectores futuros de:

- phishing
- malware
- exfiltración
- lateral movement
- privilege escalation
- ransomware
- suspicious PowerShell
- anomalías de red
- otros incidentes SOC

## Estructura

```text
SOAR-AI/
+-- data/
|   +-- alertas_simuladas.json
|   +-- reportes_generados.json
+-- src/
|   +-- __init__.py
|   +-- detector.py
|   +-- clasificador.py
|   +-- mitre_mapper.py
|   +-- severity_engine.py
|   +-- playbooks.py
|   +-- escalamiento.py
|   +-- report_generator.py
|   +-- utils.py
|   +-- qwen_agent.py
|   +-- qwen_agent_placeholder.py
+-- notebooks/
|   +-- prueba_mvp.ipynb
+-- main.py
+-- requirements.txt
+-- README.md
```

## Instalación

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

El proyecto usa `requests` para comunicarse con Ollama. La lógica principal sigue funcionando aunque Ollama no esté activo.

## Ejecución

```bash
python main.py
```

El pipeline imprime un reporte en consola y guarda la salida estructurada en:

```text
data/reportes_generados.json
```

## Qwen3 local con Ollama

La integración real está en `src/qwen_agent.py` y llama a:

```text
http://localhost:11434/api/generate
```

Modelo esperado:

```text
qwen3:0.6b
```

Preparación local sugerida:

```bash
ollama pull qwen3:0.6b
ollama serve
```

Por cada reporte generado, `main.py` intenta llamar a Qwen3 real después de construir el reporte estructurado. Si Ollama responde, la salida se guarda en:

```text
analisis_qwen3_real
```

Si Ollama está apagado, el modelo no existe o la llamada falla, el sistema guarda:

```text
analisis_qwen3_real: null
```

El campo `analisis_qwen3_placeholder` se mantiene siempre como fallback académico.

## Reglas principales

### Clasificación

- `SSH Brute Force`: actividad SSH con cinco o más intentos fallidos, o múltiples usuarios atacados.
- `Unauthorized Access Attempt`: intentos fallidos SSH aislados.
- `Suspicious Login Activity`: login exitoso después de intentos fallidos de bajo volumen.
- `Unknown Event`: evento recibido sin detector activo en el MVP.

### MITRE ATT&CK

Para `SSH Brute Force`:

- Tactic: `Credential Access`
- Technique: `Brute Force`
- Technique ID: `T1110`
- Sub-technique: `Password Guessing`
- Sub-technique ID: `T1110.001`

### Severidad

- `Baja`: menos de 5 intentos fallidos.
- `Media`: entre 5 y 10 intentos fallidos.
- `Alta`: más de 10 intentos fallidos.
- `Crítica`: más de 20 intentos fallidos, múltiples usuarios atacados o login exitoso posterior.

## Extensión sugerida

Para agregar nuevos incidentes, se recomienda:

1. Crear un detector nuevo o extender `src/detector.py`.
2. Añadir reglas de clasificación en `src/clasificador.py`.
3. Registrar el mapeo MITRE en `src/mitre_mapper.py`.
4. Añadir acciones reutilizables en `src/playbooks.py`.
5. Ajustar severidad y escalamiento si el nuevo caso requiere criterios propios.
