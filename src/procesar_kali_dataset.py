"""Módulo Adaptador para procesar el dataset real CICIDS2017 (Kali).

Toma los flujos de red crudos, limpia las anomalías matemáticas, mapea las
etiquetas a ['normal', 'fallido', 'sospechoso'] y formatea los datos para el
Random Forest del Agente Q-SOC.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Inyectar raíz para evitar ModuleNotFoundError
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configura la ruta exacta de tu CSV extraído de Kali
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUTA_SALIDA_LIMPIA = DATA_DIR / "dataset_kali_homologado.csv"

def transformar_y_limpiar_dataset():
    archivos_kali = list(DATA_DIR.glob("*pcap_ISCX.csv"))
    if not archivos_kali:
        print("❌ Error: No se encontraron archivos CSV del dataset en la carpeta data.")
        return
        
    print(f"🔍 Leyendo {len(archivos_kali)} archivos de dataset de Kali...")
    lista_dfs = [pd.read_csv(archivo, skipinitialspace=True) for archivo in archivos_kali]
    df = pd.concat(lista_dfs, ignore_index=True)
    
    print(f"📊 Registros iniciales cargados: {df.shape[0]}")

    # 1. Limpieza matemática estricta preliminar (Infinitos y NaNs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 2. Homologación Estratégica de Etiquetas (Labels)
    print("\n🏷️ Mapeando etiquetas reales a clases Q-SOC...")
    
    def mapear_etiqueta_real(label):
        label_upper = str(label).strip().upper()
        if label_upper == "BENIGN":
            return "normal"
        elif "PATATOR" in label_upper or "PORT" in label_upper:
            # Fuerza bruta o escaneos (Muchos intentos fallidos)
            return "fallido"
        else:
            # DoS, Infiltraciones, Exploits activos
            return "sospechoso"

    df['clase_evento_ml'] = df['Label'].apply(mapear_etiqueta_real)
    
    # 3. Mapeo de Atributos Críticos de Red
    df_homologado = pd.DataFrame()
    
    # Traducimos puertos a "servicios" comunes para mantener consistencia
    def determinar_servicio(port):
        try:
            port = int(port)
            if port == 22: return "ssh"
            elif port in [80, 443]: return "http"
            elif port == 21: return "ftp"
            elif port in [139, 445]: return "smb"
            else: return "otros"
        except Exception:
            return "otros"

    print("⚡ Traduciendo telemetría de red cruda a variables contextuales...")
    df_homologado['servicio'] = df['Destination Port'].apply(determinar_servicio)
    df_homologado['tipo_evento'] = df['Label'].apply(lambda x: "Network Flow" if x == "BENIGN" else f"Attack: {x}")
    df_homologado['usuario'] = "network_system"
    df_homologado['hostname'] = df['Destination Port'].apply(lambda p: f"srv-port-{p}")
    df_homologado['severidad_inicial'] = df['clase_evento_ml'].apply(lambda c: "Alta" if c == "sospechoso" else ("Media" if c == "fallido" else "Baja"))
    
    # Variables numéricas derivadas fijas
    df_homologado['hora_evento'] = 12  
    df_homologado['es_horario_laboral'] = 1
    df_homologado['ip_origen_es_privada'] = 1 
    df_homologado['usuario_privilegiado'] = 0
    df_homologado['host_critico'] = df['Destination Port'].apply(lambda p: 1 if p in [22, 21, 445] else 0)
    
    # Sanitización de la columna Flow Duration (Evita desbordamientos por negativos/infinitos)
    duraciones_limpias = pd.to_numeric(df['Flow Duration'], errors='coerce').fillna(0).clip(lower=0)
    df_homologado['conteo_eventos_ip_ventana'] = duraciones_limpias.apply(lambda x: int(np.log1p(x)) if np.isfinite(np.log1p(x)) else 0)    
    df_homologado['conteo_usuarios_distintos_ip'] = 1
    df_homologado['conteo_hosts_distintos_ip'] = 1
    
    # Búsqueda dinámica y flexible de la columna Backward Packets (CICIDS2017)
    col_bwd = [c for c in df.columns if ('bwd' in c.lower() or 'backward' in c.lower()) and 'packet' in c.lower() and '/' in c.lower()]
    
    if col_bwd:
        columna_objetivo_bwd = col_bwd[0]
        print(f"📌 Columna de paquetes reversos detectada: '{columna_objetivo_bwd}'")
        pkts_back_limpios = pd.to_numeric(df[columna_objetivo_bwd], errors='coerce').fillna(0).clip(lower=0)
    else:
        print("⚠️ Advertencia: No se encontró métrica exacta de paquetes hacia atrás. Aplicando fallback.")
        pkts_back_limpios = pd.Series(0, index=df.index)
        
    df_homologado['conteo_fallidos_ventana'] = pkts_back_limpios.apply(lambda x: int(min(x, 100)) if np.isfinite(x) else 0)
    df_homologado['servicio_sensible'] = df_homologado['servicio'].apply(lambda s: 1 if s in ["ssh", "ftp", "smb"] else 0)
    
    # Aseguramos la columna objetivo al final
    df_homologado['clase_evento_ml'] = df['clase_evento_ml']

    # 4. Guardado final del dataset limpio y estructurado
    df_homologado.to_csv(RUTA_SALIDA_LIMPIA, index=False)
    print(f"\n✅ ¡Dataset de Kali procesado con éxito!")
    print(f"💾 Guardado en: {RUTA_SALIDA_LIMPIA}")
    print("\n📊 Distribución final de clases obtenida desde CICIDS2017:")
    print(df_homologado['clase_evento_ml'].value_counts())

if __name__ == "__main__":
    transformar_y_limpiar_dataset()