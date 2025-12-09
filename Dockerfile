# Usar una imagen oficial de Python ligera
FROM python:3.10-slim

# Evitar que Python genere archivos .pyc y permitir logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto al contenedor
COPY . .

# ENTRENAR LA IA AL CONSTRUIR (Truco importante)
# Esto asegura que el archivo .pkl se genere con las versiones de linux del contenedor
RUN python train_ia.py

# Ejecutar migraciones (para crear tablas básicas de Django si usas SQLite temporal)
RUN python manage.py migrate

# Comando para arrancar el servidor usando Gunicorn
# Reemplaza 'nombre_de_tu_proyecto' por el nombre real de la carpeta que tiene el wsgi.py
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 ms_ia.wsgi:application