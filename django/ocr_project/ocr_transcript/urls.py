from django.urls import path
from ocr_transcript import views

urlpatterns = [
    path('', views.index, name='upload_image'),
]