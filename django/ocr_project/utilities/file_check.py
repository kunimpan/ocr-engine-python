import pytesseract
import cv2
from matplotlib import table
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import magic
from django.shortcuts import render
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile

def check_file_type(uploaded_file):
    """ ตรวจสอบประเภทของไฟล์ """
    mime = magic.Magic(mime=True)
    file_data = uploaded_file.read()
    mime_type = mime.from_buffer(file_data)

    # รีเซ็ตตัวชี้ไฟล์ (ถ้าไม่ทำ อาจเกิดปัญหาตอนเซฟไฟล์)
    uploaded_file.seek(0)

    if mime_type.startswith("image"):
        return "Image"
    elif mime_type == "application/pdf":
        return "PDF"
    return "Unknown"


