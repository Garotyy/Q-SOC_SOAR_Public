import json
import os

# Rutas de los archivos usando rutas relativas seguras
ruta_original = '../data/alertas_simuladas.json'
ruta_nueva = '../data/dataset_examen.json'

print(f"Buscando tus alertas en: {ruta_original}...")

try:
    with open(ruta_original, 'r', encoding='utf-8') as f:
        logs_crudos = json.load(f)
except FileNotFoundError:
    print("❌ ERROR: No se encontró 'alertas_simuladas.json' en la carpeta 'data'.")
    print("Por favor, asegúrate de haber movido tu archivo JSON a la carpeta 'data'.")
    exit()

dataset_examen = []

# Convertimos cada log plano en una pregunta de examen con su respuesta correcta
for log in logs_crudos:
    accion = "ADVERTIR"
    resultado = str(log.get("resultado", "")).lower()
    
    if resultado == "exitoso" and log.get("tipo_evento") == "ssh_authentication":
        accion = "ESCALAR_INCIDENTE"
    elif resultado == "fallido":
        accion = "BLOQUEAR"
    elif resultado == "permitido":
        accion = "PERMITIR"

    caso = {
        "escenario": f"Prueba de {log.get('id_alerta', 'Alerta')}",
        "log_entrada": log,
        "contexto_historico": {"notas": "Generado automáticamente"},
        "salida_esperada": {"accion": accion}
    }
    dataset_examen.append(caso)

# Guardamos el nuevo archivo listo para evaluar
with open(ruta_nueva, 'w', encoding='utf-8') as f:
    json.dump(dataset_examen, f, indent=4)

print(f"✅ ¡Éxito total! Se creó el examen '{ruta_nueva}' con {len(dataset_examen)} casos.")