from django.urls import path
from .views import PrediccionAvanzadaView, DashboardInteligenteView

urlpatterns = [
    path('api/prediccion-ventas/', PrediccionAvanzadaView.as_view()),
    path('api/dashboard/', DashboardInteligenteView.as_view()),
]