from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # This line includes the URLs from your 'predicciones' app
    path('', include('predicciones.urls')),
]