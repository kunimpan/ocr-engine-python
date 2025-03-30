from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from ocr_transcript import views
#from .views import display_image, download_json

urlpatterns = [
    path('', views.index, name='index'),
    path('technician/', views.technician, name='technician'),
    path('high_school_tesseract/', views.high_school_tesseract, name='high_school_tesseract'),
    path('technician_tesseract/', views.technician_tesseract, name='technician_tesseract'),
    path('high_school/', views.high_school, name='high_school'),
    #path("download-json/", download_json, name="download_json"),
    path('download-json/', views.download_json, name='download_json'),
    path('display-image/', views.display_image, name='display_image'),
    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)