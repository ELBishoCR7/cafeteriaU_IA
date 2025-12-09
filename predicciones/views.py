from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import pandas as pd
import os
from django.conf import settings
from datetime import datetime, timedelta

# --- 1. CARGAR EL CEREBRO COMPLETO AL INICIAR ---
# Nota: Cambié el nombre a 'cerebro_completo.pkl' como en el entrenamiento nuevo.
MODEL_PATH = os.path.join(settings.BASE_DIR, 'cerebro_completo.pkl')

# Variables globales
model = None
encoder = None
trends = None
peak_hours = None

try:
    artifacts = joblib.load(MODEL_PATH)
    # Extraemos todas las partes del cerebro con seguridad (.get por si falta alguna)
    model = artifacts.get('model')
    encoder = artifacts.get('encoder')
    trends = artifacts.get('trends', {})
    peak_hours = artifacts.get('peak_hours', {})
    print("✅ IA (Cerebro Completo) cargada correctamente.")
except Exception as e:
    print(f"❌ Error cargando IA: {e}")
    print("Asegúrate de haber corrido 'python train_ia.py' primero.")


# --- 2. VISTA DE PREDICCIÓN AVANZADA (Ventas + Stock + Fecha Agotamiento) ---
class PrediccionAvanzadaView(APIView):
    # Estas dos líneas evitan el error CSRF Forbidden 403
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        if model is None:
            return Response({'error': 'El modelo IA no está disponible'}, status=500)

        # Obtener datos
        data = request.data
        item_name = data.get('producto')
        # Si no envían stock, asumimos 0 (solo predice ventas)
        stock_actual = float(data.get('stock_actual', 0)) 
        # Si no envían fecha, usamos hoy
        fecha_str = data.get('fecha', datetime.now().strftime('%Y-%m-%d'))

        if not item_name:
            return Response({'error': 'Falta el nombre del producto'}, status=400)

        try:
            # Validar si el producto existe
            try:
                item_code = encoder.transform([item_name])[0]
            except ValueError:
                return Response({'error': f'El producto "{item_name}" no es conocido por la IA'}, status=400)

            # Preparar fechas
            fecha_inicio = pd.to_datetime(fecha_str)

            # --- LÓGICA DE PROYECCIÓN SEMANAL ---
            ventas_semana = []
            stock_temp = stock_actual
            fecha_agotamiento = None
            
            # Predecimos los próximos 7 días (Mañana -> Semana que viene)
            for i in range(1, 8): 
                fecha_futura = fecha_inicio + timedelta(days=i)
                
                # Features: [day_of_week, month, day, item_code]
                features = [[fecha_futura.dayofweek, fecha_futura.month, fecha_futura.day, item_code]]
                
                # Predicción (max(0) evita números negativos)
                prediccion = max(0, model.predict(features)[0])
                
                ventas_semana.append({
                    'fecha': fecha_futura.strftime('%Y-%m-%d'),
                    'prediccion': round(prediccion, 2)
                })

                # Cálculo de ruptura de stock
                stock_temp -= prediccion
                # Si el stock baja de 0 y es la primera vez que pasa, guardamos la fecha
                if stock_temp <= 0 and fecha_agotamiento is None:
                    fecha_agotamiento = fecha_futura.strftime('%Y-%m-%d')

            # Totales
            total_semana = sum(d['prediccion'] for d in ventas_semana)

            # Generar Mensaje de Alerta
            if fecha_agotamiento:
                mensaje = f"🚨 URGENTE: Tu stock se agotará el {fecha_agotamiento}."
            elif stock_temp < (stock_actual * 0.2): # Si queda menos del 20%
                mensaje = "⚠️ Advertencia: El stock terminará la semana muy bajo."
            else:
                mensaje = "✅ Stock saludable para esta semana."

            return Response({
                'producto': item_name,
                'venta_mañana': ventas_semana[0]['prediccion'],
                'venta_total_semana': round(total_semana, 2),
                'inventario_necesario_semana': round(total_semana * 1.1, 0), # Buffer del 10%
                'fecha_estimada_agotamiento': fecha_agotamiento or "Stock suficiente para >7 días",
                'alerta_ia': mensaje,
                'detalle_dias': ventas_semana
            }, status=200)

        except Exception as e:
            return Response({'error': str(e)}, status=500)


# --- 3. VISTA DE DASHBOARD INTELIGENTE (Tendencias + Horas Pico) ---
class DashboardInteligenteView(APIView):
    # API pública sin CSRF
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        if not trends:
            return Response({'error': 'Datos de tendencias no disponibles. Re-entrena el modelo.'}, status=500)

        try:
            # Filtrar productos que suben y bajan
            subiendo = [k for k, v in trends.items() if "Subiendo" in v['status']]
            bajando = [k for k, v in trends.items() if "Bajando" in v['status']]
            
            # Ordenar horas pico de mayor a menor venta
            # peak_hours es un diccionario {'Night': 500, 'Morning': 200...}
            horas_ordenadas = dict(sorted(peak_hours.items(), key=lambda item: item[1], reverse=True))
            top_hora = list(horas_ordenadas.keys())[0] if horas_ordenadas else "N/A"

            return Response({
                'resumen_semanal': {
                    'productos_estrella_ascenso': subiendo[:5], # Top 5 subiendo
                    'productos_en_riesgo': bajando[:5],         # Top 5 bajando
                    'hora_pico_absoluta': top_hora,
                },
                'recomendaciones_operativas': {
                    'mensaje_staff': f"Refuerza personal durante: {top_hora}.",
                    'detalle_horas': horas_ordenadas
                },
                'analisis_completo_tendencias': trends
            }, status=200)
        
        except Exception as e:
            return Response({'error': str(e)}, status=500)