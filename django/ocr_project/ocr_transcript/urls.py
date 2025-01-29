from django.urls import path
from ocr_transcript import views
from .views import download_json

urlpatterns = [
    path('', views.index, name='upload_image'),
    path("download-json/", download_json, name="download_json"),
]