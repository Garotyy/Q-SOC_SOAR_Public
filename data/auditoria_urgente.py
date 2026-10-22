import sys
import joblib
import json
from pathlib import Path

# Asegurar codificación utf-8 en la salida de consola en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Directorio base del script para rutas relativas robustas
BASE_DIR = Path(__file__).resolve().parent

print("1. VERIFICACIÓN DEL MODELO")
try:
    modelo_path = BASE_DIR / "modelo_random_forest.joblib"
    modelo = joblib.load(modelo_path) 
    print(f"Tipo de modelo cargado: {type(modelo)}")
    print(f"Cantidad de variables de entrada (features): {getattr(modelo, 'n_features_in_', 'No definido')}")
except Exception as e:
    print(f"Error cargando el modelo: {e}")

# 2. Auditar las alucinaciones en los JSON
print("\n--- 2. AUDITORÍA DE FIDELIDAD DE QWEN ---")
try:
    json_path = BASE_DIR / "reportes_generados.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    conteo = 0
    for item in data:
        qwen = item.get("analisis_qwen3_real", {})
        # Filtrar alertas que tienen análisis real completado (no omitidas)
        if qwen and qwen.get("estado") != "omitido":
            ans = qwen.get("respuesta", qwen)
            print(f"\nID: {item.get('id_alerta')}")
            print(f"  Evidencia cruda: {item.get('evidencia')}")
            print(f"  Hechos Qwen: {ans.get('hechos_observados')}")
            print(f"  Explicación MITRE Qwen: {ans.get('explicacion_mitre')}")
            print(f"  Riesgo Qwen: {ans.get('analisis_riesgo')}")
            conteo += 1
            
    print(f"\nTotal de reportes auditados con éxito: {conteo}")
except Exception as e:
    print(f"Error leyendo el JSON: {e}")
