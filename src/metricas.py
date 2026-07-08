import json
import ollama
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

def evaluar_con_metricas_avanzadas(ruta_json, modelo='qwen2.5:3b'):
    # Leemos el JSON (NO usamos pandas aquí)
    with open(ruta_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    print(f"🚀 Iniciando evaluación de {len(dataset)} casos con {modelo}...")
    
    y_esperado = []
    y_predicho = []
    
    for i, caso in enumerate(dataset):
        prompt = f"""
        Analiza este evento de ciberseguridad y responde SOLO en JSON estricto sin saludos ni texto adicional.
        Log: {json.dumps(caso['log_entrada'])}
        Contexto: {json.dumps(caso['contexto_historico'])}
        
        Eres un agente SOC autónomo y estricto. NO tienes permitido pedir ayuda humana.
        DEBES tomar una decisión final eligiendo UNA de estas 4 acciones:
        - PERMITIR
        - ADVERTIR
        - BLOQUEAR
        - ESCALAR_INCIDENTE
        
        Estructura obligatoria: {{"accion": "VALOR"}}
        """
        
        try:
            respuesta = ollama.generate(model=modelo, prompt=prompt, format='json', options={'temperature': 0.0})
            texto_ia = respuesta['response'].strip()
            
            # Limpieza: Si Qwen agrega las molestas comillas de markdown (```json), las quitamos
            if texto_ia.startswith("```json"):
                texto_ia = texto_ia.replace("```json", "").replace("```", "").strip()
            elif texto_ia.startswith("```"):
                texto_ia = texto_ia.replace("```", "").strip()
                
            ia_json = json.loads(texto_ia)
            
            accion_esperada = caso['salida_esperada']['accion']
            # Convertimos a mayúsculas por si Qwen responde "bloquear" en minúscula
            accion_ia = str(ia_json.get('accion', 'ERROR')).strip().upper()
            
            if "REVISI" in accion_ia or "HUMAN" in accion_ia:
                accion_ia = "REVISION_HUMANA"
            elif "ESCALAR" in accion_ia:
                accion_ia = "ESCALAR_INCIDENTE"
            
            y_esperado.append(accion_esperada)
            y_predicho.append(accion_ia)
            
        except Exception as e:
            print(f"\n❌ Falla en caso {i+1}: {e}")
            # Parche de seguridad: Solo intentamos imprimir si la variable se creó
            if 'respuesta' in locals():
                print(f"Respuesta cruda: {respuesta.get('response', 'NADA')}\n")
            else:
                print("El modelo no generó respuesta (Fallo de conexión o timeout).\n")
            
            y_esperado.append(caso['salida_esperada']['accion'])
            y_predicho.append('ERROR_FORMATO')
            
    print("\n" + "="*50)
    print("REPORTE DE MÉTRICAS CIENTÍFICAS DEL SOAR")
    print("="*50)
    
    accuracy = accuracy_score(y_esperado, y_predicho)
    error_rate = 1 - accuracy
    print(f"Accuracy (Exactitud):  {accuracy * 100:.2f}%")
    print(f"Error Rate (Tasa Err): {error_rate * 100:.2f}%\n")
    
    precision = precision_score(y_esperado, y_predicho, average='macro', zero_division=0)
    recall = recall_score(y_esperado, y_predicho, average='macro', zero_division=0)
    f1 = f1_score(y_esperado, y_predicho, average='macro', zero_division=0)
    
    print(f"🎯 Precision: {precision * 100:.2f}%")
    print(f"🔍 Recall:    {recall * 100:.2f}%")
    print(f"⚖️ F1-Score:  {f1 * 100:.2f}%\n")
    
    print("MATRIZ DE CONFUSIÓN:")
    etiquetas_unicas = sorted(list(set(y_esperado + y_predicho)))
    matriz = confusion_matrix(y_esperado, y_predicho, labels=etiquetas_unicas)
    
    print(f"{'':<18} " + " ".join([f"Pred:{e[:4]:<5}" for e in etiquetas_unicas]))
    for i, fila in enumerate(matriz):
        print(f"Real:{etiquetas_unicas[i][:12]:<13} " + " ".join([f"{val:<10}" for val in fila]))
        
    print("\nREPORTE DETALLADO POR CLASE:")
    print(classification_report(y_esperado, y_predicho, zero_division=0))

if __name__ == "__main__":
    ruta_archivo = '../data/dataset_examen.json'
    print(f"Buscando el dataset en: {ruta_archivo}")
    evaluar_con_metricas_avanzadas(ruta_archivo, modelo='qwen2.5:3b')