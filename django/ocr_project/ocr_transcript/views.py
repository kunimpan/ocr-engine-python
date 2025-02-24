from email.mime import image
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from utilities.helper import cells_detect, create_grid_image, crop_image, detect_sub_text_in_group, detect_sub_text_in_group_stud, detect_text_group_in_cell, find_text_student_info_fh, find_text_student_info_sh, hough_line_transform, persective_transformation, predict_text, predict_text_stud, show_information, split_grade_table_and_students
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

        image = cv2.imread(file_path)
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=100, sigmaSpace=100) # จำกัด noise
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_TSR_DJ")
        output_folder.mkdir(exist_ok=True)

        binary_gaussian = cv2.adaptiveThreshold(
            gray_img, 
            maxValue=255, 
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV, 
            blockSize=51, 
            C=21 #21
        )

        cv2.imwrite(f"{output_folder}/original.png", image)
        cv2.imwrite(f"{output_folder}/denoised.png", denoised)
        cv2.imwrite(f"{output_folder}/gray.png", gray_img)
        cv2.imwrite(f"{output_folder}/binary_g.png", binary_gaussian)

        table_img, student_info_img, student_info_fh_img, student_info_sh_img, table_original_img = split_grade_table_and_students(binary_gaussian, denoised)
        table_persective_img, table_original_persective_img = persective_transformation(binary_gaussian, denoised)
        grid_img = create_grid_image(table_persective_img)
        line_mask, table_without_lines, table_without_lines_2 = hough_line_transform(table_persective_img, table_original_persective_img, grid_img)
        cell_contours = cells_detect(grid_img)

        cv2.imwrite(f"{output_folder}/table_img.png", table_img)
        cv2.imwrite(f"{output_folder}/student_info_img.png", student_info_img)
        cv2.imwrite(f"{output_folder}/table_original_img.png", table_original_img)
        cv2.imwrite(f"{output_folder}/table_persective_img.png", table_persective_img)
        cv2.imwrite(f"{output_folder}/table_original_persective_img.png", table_original_persective_img)
        cv2.imwrite(f"{output_folder}/student_info_fh_img.png", student_info_fh_img)
        cv2.imwrite(f"{output_folder}/student_info_sh_img.png", student_info_sh_img)
        cv2.imwrite(f"{output_folder}/grid_img.png", grid_img)

        cell_images = crop_image(cell_contours, table_without_lines_2)

        cell_subject_code_img = cell_images[0]
        cell_subject_name_img = cell_images[1]
        cell_credit_img = cell_images[2]
        cell_academic_results_img = cell_images[3]
        cell_subject_code_img_2 = cell_images[5]
        cell_subject_name_img_2 = cell_images[6]
        cell_credit_img_2 = cell_images[7]
        cell_academic_results_img_2 = cell_images[8]

        # ตารางครึ่งแรก
        text_subject_code_images, calculate_line_stats_1, subject_code_img = detect_text_group_in_cell(cell_subject_code_img, 1)
        text_subject_name_images, subject_name_img = detect_text_group_in_cell(cell_subject_name_img, 2, calculate_line_stats_1)
        text_credit_images, credit_img = detect_text_group_in_cell(cell_credit_img, 2, calculate_line_stats_1)
        text_academic_results_images, academic_results_img = detect_text_group_in_cell(cell_academic_results_img, 2, calculate_line_stats_1)

        # ตารางครึ่งหลัง
        text_subject_code_images_2, calculate_line_stats_2, subject_code_img_2 = detect_text_group_in_cell(cell_subject_code_img_2, 1)
        text_subject_name_images_2, subject_name_img_2 = detect_text_group_in_cell(cell_subject_name_img_2, 2, calculate_line_stats_2)
        text_credit_images_2, credit_img_2 = detect_text_group_in_cell(cell_credit_img_2, 2, calculate_line_stats_2)
        text_academic_results_images_2, academic_results_img_2 = detect_text_group_in_cell(cell_academic_results_img_2, 2, calculate_line_stats_2)

        cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_img.jpg", subject_code_img)
        cv2.imwrite(f"{output_folder}/cell_images/cca_subject_name_img.jpg", subject_name_img)
        cv2.imwrite(f"{output_folder}/cell_images/cca_credit_img.jpg", credit_img)
        cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_img.jpg", academic_results_img)
        cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_img_2.jpg", subject_code_img_2)
        cv2.imwrite(f"{output_folder}/cell_images/cca_subject_name_img_2.jpg", subject_name_img_2)
        cv2.imwrite(f"{output_folder}/cell_images/cca_credit_img_2.jpg", credit_img_2)
        cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_img_2.jpg", academic_results_img_2)

        text_group_subject_code = detect_sub_text_in_group(text_subject_code_images)
        text_group_subject_name = detect_sub_text_in_group(text_subject_name_images)
        text_group_credit = detect_sub_text_in_group(text_credit_images)
        text_group_academic_results = detect_sub_text_in_group(text_academic_results_images)
        text_group_subject_code_2 = detect_sub_text_in_group(text_subject_code_images_2)
        text_group_subject_name_2 = detect_sub_text_in_group(text_subject_name_images_2)
        text_group_credit_2 = detect_sub_text_in_group(text_credit_images_2)
        text_group_academic_results_2 = detect_sub_text_in_group(text_academic_results_images_2)

        global text_box_subject_code_all
        global text_box_subject_name_all
        global text_box_credit_all
        global text_box_academic_results_all

        text_box_subject_code = predict_text(text_group_subject_code, 1)
        text_box_subject_name = predict_text(text_group_subject_name)
        text_box_credit = predict_text(text_group_credit, 3)
        text_box_academic_results = predict_text(text_group_academic_results, 4)
        text_box_subject_code_2 = predict_text(text_group_subject_code_2, 1)
        text_box_subject_name_2 = predict_text(text_group_subject_name_2)
        text_box_credit_2 = predict_text(text_group_credit_2, 3)
        text_box_academic_results_2 = predict_text(text_group_academic_results_2, 4)

        text_box_subject_code_all = text_box_subject_code + text_box_subject_code_2
        text_box_subject_name_all = text_box_subject_name + text_box_subject_name_2
        text_box_credit_all = text_box_credit + text_box_credit_2
        text_box_academic_results_all = text_box_academic_results + text_box_academic_results_2

        combined_list = list(zip(text_box_subject_code_all, text_box_subject_name_all, text_box_credit_all, text_box_academic_results_all))

        text_stud_fh_images = find_text_student_info_fh(student_info_fh_img)
        text_stud_sh_images = find_text_student_info_sh(student_info_sh_img)

        indices_fh = [3, -2, -1]
        indices_sh = [-3, -1]
        student_name, field_of_study, field_of_work = [text_stud_fh_images[i] for i in indices_fh]
        cgpa, graduation_date = [text_stud_sh_images[i] for i in indices_sh]

        text_group_student_name = detect_sub_text_in_group_stud(student_name)
        text_group_field_of_study = detect_sub_text_in_group_stud(field_of_study)
        text_group_field_of_work = detect_sub_text_in_group_stud(field_of_work)

        text_group_cgpa = detect_sub_text_in_group_stud(cgpa)
        text_group_graduation_date = detect_sub_text_in_group_stud(graduation_date)

        text_box_student_name = predict_text_stud(text_group_student_name[3:], 1)
        text_box_field_of_study = predict_text_stud(text_group_field_of_study[1:], 1)
        text_box_field_of_work = predict_text_stud(text_group_field_of_work[1:], 1)

        text_box_cgpa = predict_text_stud(text_group_cgpa[1:], 1)
        text_box_graduation_date = predict_text_stud(text_group_graduation_date[1:], 1)


        return render(request, 'index.html', {
            'uploaded_file_url':uploaded_file_url,
            'student_name':text_box_student_name,
            'field_of_study':text_box_field_of_study,
            'field_of_work':text_box_field_of_work,
            'cgpa':text_box_cgpa,
            'graduation_date':text_box_graduation_date,
            #'studentInfo':studentInfo_text_box[2:],
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

        '''
        print("จำนวนรหัสวิชา:" ,len(text_box_subject_code))
        show_information(text_box_subject_code)

        print("จำนวนชื่อวิชา:" ,len(text_box_subject_code))
        show_information(text_box_subject_name)

        print("จำนวนหน่วยกิต:" ,len(text_box_credit))
        show_information(text_box_credit)

        print("จำนวนผลการเรียน:" ,len(text_box_academic_results))
        show_information(text_box_academic_results)

        print("จำนวนรหัสวิชา 2:" ,len(text_box_subject_code_2))
        show_information(text_box_subject_code_2)

        print("จำนวนชื่อวิชา 2:" ,len(text_box_subject_code_2))
        show_information(text_box_subject_name_2)

        print("จำนวนหน่วยกิต 2:" ,len(text_box_credit_2))
        show_information(text_box_credit_2)

        print("จำนวนผลการเรียน 2:" ,len(text_box_academic_results_2))
        show_information(text_box_academic_results_2)
        '''

    else:
        return render(request, 'index.html')






def download_json(request):
    
    # ข้อมูล JSON ที่ต้องการให้ดาวน์โหลด
    data = {
        "subject_code": text_box_subject_code_all,
        "subject_name": text_box_subject_name_all,
        "credit": text_box_credit_all,
        "academic_results": text_box_academic_results_all 
    }

    # แปลงข้อมูลเป็น JSON
    json_data = json.dumps(data, indent=4, ensure_ascii=False)  # ensure_ascii=False รองรับภาษาไทย

    # สร้าง Response และตั้งค่าให้เป็นไฟล์แนบ (Attachment)
    response = HttpResponse(json_data, content_type="application/json")
    response["Content-Disposition"] = 'attachment; filename="data.json"'
    

    return response


    


