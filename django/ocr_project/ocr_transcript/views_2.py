from email.mime import image
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from utilities.helper import detect_text_group_in_cell, detect_text_in_group, get_x_position, get_y_position, load_image_for_ocr, predict_text_in_cell, separate_table_and_studentInfo, detect_text_group, table_cell_detection
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import json
from django.http import HttpResponse

print(pytesseract.get_tesseract_version())
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Create your views here.
def index(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_path = Path(settings.MEDIA_ROOT) / image.name
        uploaded_file_url = fs.url(filename)
        
        binary_img = load_image_for_ocr(cv2.imread(file_path))
        table_img, studentInfo_img = separate_table_and_studentInfo(binary_img)
        text_group_personal_img = detect_text_group(studentInfo_img)
        #print(len(text_group_personal_img))

        custom_config = r'--oem 3 --psm 7'  # ใช้ OCR Engine Mode และ Page Segmentation Mode ที่เหมาะสม
        studentInfo_text_box = []
        for idx, text_img in enumerate(text_group_personal_img):
            text = pytesseract.image_to_string(text_img, config=custom_config, lang='tha')
            text_cleaned = text.replace("\n", "")  # ลบ \n ออก
            studentInfo_text_box.append(text_cleaned)

        table_cell_contours = table_cell_detection(table_img)
        sorted_contours = sorted(table_cell_contours, key=get_y_position) # จัดเรียง contours ตามตำแหน่ง y (แนวตั้ง) เพื่อแยกแถว

        # แยกแถว
        rows = []
        current_row = []
        previous_y = -1
        for ctr in sorted_contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if(w < 30 and h < 30):
                continue
            if previous_y == -1 or abs(y - previous_y) < 10:  # Same row threshold
                current_row.append((x, y, w, h))
            else:
                # จัดเรียงคอลัมน์ในแถวปัจจุบันตามตำแหน่ง x
                rows.append(sorted(current_row, key=get_x_position))
                current_row = [(x, y, w, h)]
            previous_y = y

        # เพิ่มแถวสุดท้าย
        if current_row:
            rows.append(sorted(current_row, key=get_x_position))

        cell_images = [] #เก็บรูปเซลของตาราง
        for row_index, row in enumerate(rows[1:2]):
            for col_index, cell in enumerate(row):
                x, y, w, h = cell
                cell_img = table_img[y+4:y+h-4, x+4:x+w-4]
                cell_images.append(cell_img)

        cell_text_group_images = [] #เก็บรูปกลุ่มข้อความในเซลตาราง
        for idx_cell, cell_img in enumerate(cell_images):
            kernel = np.ones((5, 15), np.uint8)
            lines_cell = cv2.dilate(cell_img, kernel, iterations=1)
            if idx_cell == 0 or idx_cell == 5:
                text_group_cells, calculate_line_stats = detect_text_group_in_cell(lines_cell, cell_img, 1)
            else:
                text_group_cells = detect_text_group_in_cell(lines_cell, cell_img, 2, calculate_line_stats)
            cell_text_group_images.append(text_group_cells)

        cell_text_group_sub_images = [] #เก็บรูปกลุ่มข้อความย่อยในเซลตาราง
        for idx_cell, cell in enumerate(cell_text_group_images):
            sub_text_cell_images = []
            for idx_text, text_img in enumerate(cell):
                sub_text_cell_images.append(detect_text_in_group(text_img))
            cell_text_group_sub_images.append(sub_text_cell_images)

        cell_subject_code = cell_text_group_sub_images[0]
        cell_subject_name = cell_text_group_sub_images[1]
        cell_credit = cell_text_group_sub_images[2]
        cell_academic_results = cell_text_group_sub_images[3]
        cell_subject_code_2 = cell_text_group_sub_images[5]
        cell_subject_name_2 = cell_text_group_sub_images[6]
        cell_credit_2 = cell_text_group_sub_images[7]
        cell_academic_results_2 = cell_text_group_sub_images[8]

        global text_box_subject_code
        text_box_subject_code = predict_text_in_cell(cell_subject_code)
        global text_box_subject_name 
        text_box_subject_name = predict_text_in_cell(cell_subject_name)
        global text_box_credit 
        text_box_credit = predict_text_in_cell(cell_credit)
        global text_box_academic_results 
        text_box_academic_results = predict_text_in_cell(cell_academic_results)
        text_box_subject_code_2 = predict_text_in_cell(cell_subject_code_2)
        text_box_subject_name_2 = predict_text_in_cell(cell_subject_name_2)
        text_box_credit_2 = predict_text_in_cell(cell_credit_2)
        text_box_academic_results_2 = predict_text_in_cell(cell_academic_results_2)

        combined_list = list(zip(text_box_subject_code, text_box_subject_name, text_box_credit, text_box_academic_results))

        return render(request, 'index.html', {
            'uploaded_file_url':uploaded_file_url,
            'studentInfo':studentInfo_text_box[2:],
            #'subject_code':text_box_subject_code,
            #'subject_name':text_box_subject_name,
            #'credit':text_box_credit,
            #'academic_results':text_box_academic_results,
            #'subject_code_2':text_box_subject_code_2,
            #'subject_name_2':text_box_subject_name_2,
            #'credit_2':text_box_credit_2,
            #'academic_results_2':text_box_academic_results_2,
            'combined_list':combined_list
        })
    
    else:
        return render(request, 'index.html')
    
def download_json(request):
    # ข้อมูล JSON ที่ต้องการให้ดาวน์โหลด
    data = {
        "subject_code": text_box_subject_code,
        "subject_name": text_box_subject_name,
        "credit": text_box_credit,
        "academic_results": text_box_academic_results 
    }

    # แปลงข้อมูลเป็น JSON
    json_data = json.dumps(data, indent=4, ensure_ascii=False)  # ensure_ascii=False รองรับภาษาไทย

    # สร้าง Response และตั้งค่าให้เป็นไฟล์แนบ (Attachment)
    response = HttpResponse(json_data, content_type="application/json")
    response["Content-Disposition"] = 'attachment; filename="data.json"'

    return response
    


