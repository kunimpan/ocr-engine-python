from pathlib import Path
from pdf2image import convert_from_bytes  # ใช้ในการแปลง PDF เป็นรูปภาพ
from PIL import Image
import io
import base64

def process_file(uploaded_file):
    poppler_path = r"C:\python\poppler-24.08.0\Library\bin"

    """
    ฟังก์ชันตรวจสอบไฟล์ว่ามีนามสกุลเป็น PDF หรือไฟล์รูปภาพ
    - ถ้าเป็น PDF: อ่านข้อมูลเป็น bytes แล้วใช้ pdf2image แปลง PDF เป็นรูป (list ของ PIL Image)
    - ถ้าเป็นรูปภาพ: เปิดไฟล์ด้วย PIL และ return เป็น list ของรูปภาพ (เพื่อความสอดคล้องกัน)
    """
    # ตรวจสอบนามสกุลไฟล์ด้วย pathlib
    file_path = Path(uploaded_file.name)
    ext = file_path.suffix.lower()  # เช่น ".pdf", ".jpg" เป็นต้น

    if ext == ".pdf":
        # อ่านข้อมูลไฟล์เป็น bytes
        pdf_bytes = uploaded_file.read()
        # แปลง PDF เป็นรูปภาพ (จะได้เป็น list ของ PIL Image)
        images = convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
        return images

    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        # ถ้าเป็นไฟล์รูปภาพ ให้เปิดด้วย PIL และ return เป็น list
        image = Image.open(uploaded_file)
        return [image]

    else:
        raise ValueError("รองรับเฉพาะไฟล์ PDF หรือไฟล์รูปภาพเท่านั้น.")
    
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str