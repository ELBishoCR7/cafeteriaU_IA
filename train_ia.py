import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def entrenar_cerebro():
    print("⏳ Cargando y procesando datos...")
    
    # 1. Cargar datos
    try:
        df = pd.read_csv('dataset_convertido.csv')
    except FileNotFoundError:
        print("❌ Error: No se encuentra 'dataset_convertido.csv'. Asegúrate de que esté en la misma carpeta.")
        return

    # 2. Limpieza y Conversión de Fechas (Manejo robusto de formatos mixtos)
    # Intentamos inferir el formato. Si hay fechas como '30-02', las convertirá en NaT (Not a Time)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
    
    # Eliminar filas donde la fecha o la cantidad sean inválidas
    df = df.dropna(subset=['date', 'quantity'])
    
    # Asegurar que 'quantity' es numérico
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)

    # 3. Feature Engineering (Variables para la IA)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Codificar el nombre del producto (Texto -> Número)
    le = LabelEncoder()
    df['item_code'] = le.fit_transform(df['item_name'])

    # --- MÓDULO 1: PREDICCIÓN DE DEMANDA (CORREGIDO) ---
    print("🧠 Entrenando modelo de predicción de cantidad...")
    
    # X = Features (Día, Mes, Año, Producto ID)
    X = df[['day_of_week', 'month', 'day', 'item_code']]
    
    # Y = Target (CORREGIDO: Ahora apuntamos a 'quantity', NO a 'item_type')
    y = df['quantity'] 

    # Usamos RandomForestRegressor (Regresor = predice números, no categorías)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- MÓDULO 2: DETECCIÓN DE TENDENCIAS ---
    print("📈 Analizando tendencias de mercado...")
    trend_info = {}
    
    if not df.empty:
        latest_date = df['date'].max()
        start_date = latest_date - pd.Timedelta(days=30) # Últimos 30 días
        recent_data = df[df['date'] >= start_date]

        for item in df['item_name'].unique():
            item_data = recent_data[recent_data['item_name'] == item].groupby('date')['quantity'].sum().reset_index()
            
            status = "Estable ➖"
            slope = 0
            
            if len(item_data) > 1:
                # Regresión lineal simple para ver la pendiente (slope)
                x_vals = np.arange(len(item_data))
                y_vals = item_data['quantity'].values
                slope = np.polyfit(x_vals, y_vals, 1)[0]
                
                if slope > 0.1: status = "Subiendo 🔥"
                elif slope < -0.1: status = "Bajando 📉"
            
            trend_info[item] = {'slope': round(slope, 3), 'status': status}

    # --- MÓDULO 3: HORAS PICO ---
    print("⏰ Calculando horas pico...")
    # Sumar ventas por 'time_of_sale'
    if 'time_of_sale' in df.columns:
        peak_hours = df.groupby('time_of_sale')['quantity'].sum().sort_values(ascending=False).to_dict()
    else:
        peak_hours = {}
        print("⚠️ Advertencia: No se encontró columna 'time_of_sale'.")

    # 4. Guardar todo en un solo archivo .pkl
    artifacts = {
        'model': model,      # El cerebro que predice números
        'encoder': le,       # El traductor de nombres a códigos
        'trends': trend_info,# Datos de qué sube/baja
        'peak_hours': peak_hours # Datos de horas pico
    }
    
    joblib.dump(artifacts, 'cerebro_completo.pkl')
    print("✅ ¡Éxito! Archivo 'cerebro_completo.pkl' generado correctamente.")
    print(f"   - Modelo entrenado con {len(df)} registros.")

if __name__ == "__main__":
    entrenar_cerebro()