from django.contrib import messages
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import json
from django.http import FileResponse, HttpResponse, JsonResponse
import utilities.file_check as fc
import utilities.function_hs_tsr as hs_tsr
import utilities.function_hs as hs
import utilities.function_tn_tsr as tn_tsr
import utilities.function_tn as tn
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import gc

print(pytesseract.get_tesseract_version())
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Create your views here.
def index(request):
    return render(request, 'index.html')


def high_school_tesseract(request):
    if request.method == 'POST' and 'file_uploads' in request.FILES:
        uploaded_files = request.FILES.getlist('file_uploads')  # รับไฟล์ทั้งหมด

        media_path = Path(settings.MEDIA_ROOT)  # ใช้ pathlib.Path
        media_path.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ media/ ถ้ายังไม่มี

        saved_images = []
        pillow_images = []
        for idx, file in enumerate(uploaded_files):
            file_type = fc.check_file_type(file)

            if file_type == "Image":
                # บันทึกเฉพาะไฟล์รูปภาพลงโฟลเดอร์ media/
                image_path = media_path / file.name
                with image_path.open('wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                img = Image.open(file)
                pillow_images.append(img)
                        
                image_url = settings.MEDIA_URL + file.name
                saved_images.append(image_url)
                messages.success(request, f"อัปโหลด {file.name} สำเร็จ!")

            elif file_type == "PDF":
                messages.warning(request, f"อัปโหลด {file.name} สำเร็จ แต่ยังไม่ได้บันทึก (PDF)")

            else:
                messages.error(request, f"{file.name} ไม่ใช่ไฟล์ที่รองรับ!") 

        output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_HS_TSR_DJ/front")
        output_folder.mkdir(exist_ok=True)

        print("pillow_images:", len(pillow_images))

        if len(saved_images) > 0:
            # f คือ front
            if len(saved_images) == 1:
                f_image = pillow_images[0]
            else:
                f_image = pillow_images[1]
            #f_image = Image.open(f"..{saved_images[0]}")
            f_new_size = (1660, 2347)
            f_resized_pil = f_image.resize(f_new_size, Image.LANCZOS)
            f_img_rgb = np.array(f_resized_pil)
            f_img_cv = cv2.cvtColor(f_img_rgb, cv2.COLOR_RGB2BGR)
            f_denoised = cv2.bilateralFilter(f_img_cv, d=9, sigmaColor=75, sigmaSpace=75) # จำกัด noise
            f_gray_img = cv2.cvtColor(f_denoised, cv2.COLOR_BGR2GRAY)
        
            f_binary_gaussian = cv2.adaptiveThreshold(
                f_gray_img, 
                maxValue=255, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV, 
                blockSize=51, #51
                C=9 #21 #15
            )

            # สร้าง kernel สำหรับ morphological operation
            kernel = np.ones((3, 3), np.uint8)
            f_dilated = cv2.dilate(f_binary_gaussian, kernel, iterations=1)
            f_closed_dummy = cv2.morphologyEx(f_binary_gaussian, cv2.MORPH_CLOSE, kernel, iterations=1)

            cv2.imwrite(f"{output_folder}/img_cv.png", f_img_cv)
            cv2.imwrite(f"{output_folder}/denoised.png", f_denoised)
            cv2.imwrite(f"{output_folder}/gray.png", f_gray_img)
            cv2.imwrite(f"{output_folder}/binary_g.png", f_binary_gaussian)
            cv2.imwrite(f"{output_folder}/dilated.png", f_dilated)
            cv2.imwrite(f"{output_folder}/closed_dummy.png", f_closed_dummy)

            table_img, table_dummy_img, table_original_img, student_info_img, student_info_fh_img, student_info_sh_img = hs_tsr.split_grade_table_and_students(f_binary_gaussian, f_denoised, f_dilated)
            table_persective_img, table_original_persective_img, table_dummy_persective_img = hs_tsr.persective_transformation(f_binary_gaussian, f_denoised, f_dilated)

            cv2.imwrite(f"{output_folder}/table_img.png", table_img)
            cv2.imwrite(f"{output_folder}/table_dummy_img.png", table_dummy_img)
            cv2.imwrite(f"{output_folder}/table_original_img.png", table_original_img)
            cv2.imwrite(f"{output_folder}/student_info_img.png", student_info_img)
            cv2.imwrite(f"{output_folder}/student_info_fh_img.png", student_info_fh_img)
            cv2.imwrite(f"{output_folder}/student_info_sh_img.png", student_info_sh_img)

            cv2.imwrite(f"{output_folder}/table_persective_img.png", table_persective_img)
            cv2.imwrite(f"{output_folder}/table_original_persective_img.png", table_original_persective_img)
            cv2.imwrite(f"{output_folder}/table_dummy_persective_img.png", table_dummy_persective_img)

            # หา column ของตาราง
            cell_column_images = hs_tsr.find_table_columns(table_dummy_persective_img, table_persective_img)

            # แบ่ง column ตาราง
            cell_subject_code_name_img = hs_tsr.crop_top_bottom(cell_column_images[0])
            cell_credit_img = hs_tsr.crop_top_bottom(cell_column_images[1])
            cell_academic_results_img = hs_tsr.crop_top_bottom(cell_column_images[2])

            cell_subject_code_name_img_2 = hs_tsr.crop_top_bottom(cell_column_images[3])
            cell_credit_img_2 = hs_tsr.crop_top_bottom(cell_column_images[4])
            cell_academic_results_img_2 = hs_tsr.crop_top_bottom(cell_column_images[5])

            cell_subject_code_name_img_3 = hs_tsr.crop_top_bottom(cell_column_images[6])
            cell_credit_img_3 = hs_tsr.crop_top_bottom(cell_column_images[7])
            cell_academic_results_img_3 = hs_tsr.crop_top_bottom(cell_column_images[8])

            # จับกลุ่มข้อความของ cell ตาราง
            text_academic_results_images, calculate_line_stats_1, academic_results_img = hs_tsr.detect_text_in_cell(cell_academic_results_img, 1)
            text_subject_code_name_images, subject_code_name_img = hs_tsr.detect_text_in_cell(cell_subject_code_name_img, 2, calculate_line_stats_1)
            text_credit_images, credit_img = hs_tsr.detect_text_in_cell(cell_credit_img, 2, calculate_line_stats_1)

            text_academic_results_images_2, calculate_line_stats_2, academic_results_img_2 = hs_tsr.detect_text_in_cell(cell_academic_results_img_2, 1)
            text_subject_code_name_images_2, subject_code_name_img_2 = hs_tsr.detect_text_in_cell(cell_subject_code_name_img_2, 2, calculate_line_stats_2)
            text_credit_images_2, credit_img_2 = hs_tsr.detect_text_in_cell(cell_credit_img_2, 2, calculate_line_stats_2)

            text_academic_results_images_3, calculate_line_stats_3, academic_results_img_3 = hs_tsr.detect_text_in_cell(cell_academic_results_img_3, 1)
            text_subject_code_name_images_3, subject_code_name_img_3 = hs_tsr.detect_text_in_cell(cell_subject_code_name_img_3, 2, calculate_line_stats_3)
            text_credit_images_3, credit_img_3 = hs_tsr.detect_text_in_cell(cell_credit_img_3, 2, calculate_line_stats_3)

            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results.jpg", academic_results_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_name.jpg", subject_code_name_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit.jpg", credit_img)

            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_2.jpg", academic_results_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_name_2.jpg", subject_code_name_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_2.jpg", credit_img_2)

            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_3.jpg", academic_results_img_3)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_name_3.jpg", subject_code_name_img_3)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_3.jpg", credit_img_3)

            # จับข้อความย่อยในกลุ่มข้อความ
            text_group_subject_code_name = hs_tsr.detect_sub_text_in_group(text_subject_code_name_images)
            text_group_subject_code, text_group_subject_name = hs_tsr.separate_subject_code_and_name(text_group_subject_code_name)
            text_group_credit = hs_tsr.detect_sub_text_in_group(text_credit_images)
            text_group_academic_results = hs_tsr.detect_sub_text_in_group(text_academic_results_images)

            text_group_subject_code_name_2 = detect_sub_text_in_group(text_subject_code_name_images_2)
            text_group_subject_code_2, text_group_subject_name_2 = hs_tsr.separate_subject_code_and_name(text_group_subject_code_name_2)
            text_group_credit_2 = hs_tsr.detect_sub_text_in_group(text_credit_images_2)
            text_group_academic_results_2 = hs_tsr.detect_sub_text_in_group(text_academic_results_images_2)

            text_group_subject_code_name_3 = detect_sub_text_in_group(text_subject_code_name_images_3)
            text_group_subject_code_3, text_group_subject_name_3 = hs_tsr.separate_subject_code_and_name(text_group_subject_code_name_3)
            text_group_credit_3 = hs_tsr.detect_sub_text_in_group(text_credit_images_3)
            text_group_academic_results_3 = hs_tsr.detect_sub_text_in_group(text_academic_results_images_3)

            # ทำนาย
            
            text_box_subject_code = hs_tsr.predict_text(text_group_subject_code, 1)
            text_box_subject_name = hs_tsr.predict_text(text_group_subject_name, 2)
            text_box_credit = hs_tsr.predict_text(text_group_credit, 3)
            text_box_academic_results = hs_tsr.predict_text(text_group_academic_results, 4)

            text_box_subject_code_2 = hs_tsr.predict_text(text_group_subject_code_2, 1)
            text_box_subject_name_2 = hs_tsr.predict_text(text_group_subject_name_2, 2)
            text_box_credit_2 = hs_tsr.predict_text(text_group_credit_2, 3)
            text_box_academic_results_2 = hs_tsr.predict_text(text_group_academic_results_2, 4)

            text_box_subject_code_3 = hs_tsr.predict_text(text_group_subject_code_3, 1)
            text_box_subject_name_3 = hs_tsr.predict_text(text_group_subject_name_3, 2)
            text_box_credit_3 = hs_tsr.predict_text(text_group_credit_3, 3)
            text_box_academic_results_3 = hs_tsr.predict_text(text_group_academic_results_3, 4)
            
            
            ## ข้อมูลนักศึกษา
            # หาชื่อกับนามสกุล
            name_coordinate, lastname_coordinate = hs_tsr.find_text_student_info_fh(student_info_fh_img)
            text_name_img, text_lastname_img = hs_tsr.find_text_student_info_sh(student_info_sh_img, name_coordinate, lastname_coordinate)

            ### จับกลุ่มข้อความย่อย
            text_group_student_name = hs_tsr.detect_sub_text_in_group_stud(text_name_img)
            text_group_student_lastname = hs_tsr.detect_sub_text_in_group_stud(text_lastname_img)

            # ทำนาย
            text_box_student_name = hs_tsr.predict_text_stud(text_group_student_name[1:])
            text_box_student_lastname = hs_tsr.predict_text_stud(text_group_student_lastname[1:])
            text_student_fullname = text_box_student_name + text_box_student_lastname
            


            if len(pillow_images) == 2:
                output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_HS_TSR_DJ/back")
                output_folder.mkdir(exist_ok=True)

                ## หน้าหลัง
                b_image = pillow_images[0]
                b_new_size = (1660, 2347)
                b_resized_pil = b_image.resize(b_new_size, Image.LANCZOS)
                b_img_rgb = np.array(b_resized_pil)
                b_img_cv = cv2.cvtColor(b_img_rgb, cv2.COLOR_RGB2BGR)
                b_denoised = cv2.bilateralFilter(b_img_cv, d=9, sigmaColor=75, sigmaSpace=75) # จำกัด noise
                b_gray_img = cv2.cvtColor(b_denoised, cv2.COLOR_BGR2GRAY)
        
                b_binary_gaussian = cv2.adaptiveThreshold(
                    b_gray_img, 
                    maxValue=255, 
                    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    thresholdType=cv2.THRESH_BINARY_INV, 
                    blockSize=51, #51
                    C=9 #21 #15
                )

                # สร้าง kernel สำหรับ morphological operation
                kernel = np.ones((3, 3), np.uint8)
                b_dilated = cv2.dilate(b_binary_gaussian, kernel, iterations=1)
                b_closed_dummy = cv2.morphologyEx(b_binary_gaussian, cv2.MORPH_CLOSE, kernel, iterations=1)

                cv2.imwrite(f"{output_folder}/img_cv.png", b_img_cv)
                cv2.imwrite(f"{output_folder}/denoised.png", b_denoised)
                cv2.imwrite(f"{output_folder}/gray.png", b_gray_img)
                cv2.imwrite(f"{output_folder}/binary_g.png", b_binary_gaussian)
                cv2.imwrite(f"{output_folder}/dilated.png", b_dilated)
                cv2.imwrite(f"{output_folder}/closed_dummy.png", b_closed_dummy)

                ## หาตาราง
                table_img, table_dummy_img, table_original_im = hs_tsr.fine_table(b_binary_gaussian, b_denoised, b_dilated)
                table_persective_img, table_original_persective_img, table_dummy_persective_img = hs_tsr.persective_transformation(b_binary_gaussian, b_denoised, b_dilated)

                cv2.imwrite(f"{output_folder}/table_img.png", table_img)
                cv2.imwrite(f"{output_folder}/table_dummy_img.png", table_dummy_img)

                cv2.imwrite(f"{output_folder}/table_persective_img.png", table_persective_img)
                cv2.imwrite(f"{output_folder}/table_original_persective_img.png", table_original_persective_img)
                cv2.imwrite(f"{output_folder}/table_dummy_persective_img.png", table_dummy_persective_img)

                # หา column ของตาราง
                cell_column_images, cell_column_dummy_images = hs_tsr.find_table_columns_b(table_dummy_persective_img, table_persective_img)

                # แบ่ง column ตาราง
                cell_1 = cell_column_images[0]
                cell_2 = hs_tsr.crop_top_bottom(cell_column_images[1], 20, 40)
                
                department_credits_img, department_academic_results_img, sum_department_credits_img, gpa_img = hs_tsr.find_table_subject_group(cell_2)

                # หากลุ่มข้อความของตาราง subject_group
                texts_department_credits = hs_tsr.detect_text_subject_group(department_credits_img)
                texts_department_academic_results = hs_tsr.detect_text_subject_group(department_academic_results_img)

                texts_sum_department_credits = hs_tsr.detect_text_subject_group(sum_department_credits_img)
                texts_gpa = hs_tsr.detect_text_subject_group(gpa_img)

                ## ทำนาย
                text_box_department_credits = hs_tsr.predict_text_subject_group(texts_department_credits)
                text_box_department_academic_results = hs_tsr.predict_text_subject_group(texts_department_academic_results)
                text_box_sum_department_credits = hs_tsr.predict_text_subject_group(texts_sum_department_credits)
                text_box_gpa = hs_tsr.predict_text_subject_group(texts_gpa)

                print(text_box_department_credits)
                print(text_box_department_academic_results)
                print(text_box_sum_department_credits)
                print(text_box_gpa)

                ### จบ if

            # JSON
            if(len(pillow_images) == 2):
                # มีทั้งหน้าทั้งหลัง
                # รวมข้อมูล
                text_box_subject_code_all = text_box_subject_code + text_box_subject_code_2 + text_box_subject_code_3
                text_box_subject_name_all = text_box_subject_name + text_box_subject_name_2 + text_box_subject_name_3
                text_box_credit_all = text_box_credit + text_box_credit_2 + text_box_credit_3
                text_box_academic_results_all = text_box_academic_results + text_box_academic_results_2 + text_box_academic_results_3

                #ลบช่องว่าง
                subject_codes = [scode.replace(" ", "") for scode in text_box_subject_code_all]
                subject_names = [sname.strip() for sname in text_box_subject_name_all]
                credits = [credit.replace(" ", "") for credit in text_box_credit_all] 
                grades = [grade.replace(" ", "") for grade in text_box_academic_results_all]
                
                student_name = text_box_student_name.replace(" ", "")
                student_lastname = text_box_student_lastname.replace(" ", "")

                department_credits = [departcredit.replace(" ", "") for departcredit in text_box_department_credits]
                department_academic_results = [ac.replace(" ", "") for ac in text_box_department_academic_results]
                sum_department_credits = [sum.replace(" ", "") for sum in text_box_sum_department_credits]
                gpa = [gpa.replace(" ", "") for gpa in text_box_gpa]
            
                combined_list = list(zip(subject_codes, subject_names, credits, grades))

                enrolled_subjects = []
                for scode, sname, credit, grade in zip(subject_codes, subject_names, credits, grades):
                    enrolled_subjects.append({
                        "subject_code": scode,
                        "subject_name": sname,
                        "credit": credit,
                        "grade": grade,
                    })
            
                # สร้างโครงสร้าง JSON ในรูปแบบของ dictionary
                data = {
                    "student_info": {
                        "name": student_name,
                        "lastname": student_lastname,
                        "math_credits": department_credits[1],
                        "math_grade": department_academic_results[1],
                        "science_credits": department_credits[2],
                        "science_grade": department_academic_results[2],
                        "gpa": gpa[0],
                    },
                    "enrolled_subjects": enrolled_subjects
                }

                # แปลง Python dictionary เป็น JSON string
                json_path = media_path / "transcript.json"
                with json_path.open("w", encoding="utf-8") as json_file:
                    json.dump([data], json_file, indent=4, ensure_ascii=False)

                return render(request, 'high_school_tesseract.html', {
                    'saved_images': saved_images,
                    'student_name':student_name,
                    'student_lastname':student_lastname,
                    "math_credits": department_credits[1],
                    "math_grade": department_academic_results[1],
                    "science_credits": department_credits[2],
                    "science_grade": department_academic_results[2],
                    "gpa": gpa[0],                    
                    'combined_list':combined_list,
                    'json_url': settings.MEDIA_URL + "transcript.json"
                })
            
            else:
                # มีแค่ภาพด้านหน้า
                # รวมข้อมูล
                text_box_subject_code_all = text_box_subject_code + text_box_subject_code_2 + text_box_subject_code_3
                text_box_subject_name_all = text_box_subject_name + text_box_subject_name_2 + text_box_subject_name_3
                text_box_credit_all = text_box_credit + text_box_credit_2 + text_box_credit_3
                text_box_academic_results_all = text_box_academic_results + text_box_academic_results_2 + text_box_academic_results_3

                #ลบช่องว่าง
                subject_codes = [scode.replace(" ", "") for scode in text_box_subject_code_all]
                subject_names = [sname.strip() for sname in text_box_subject_name_all]
                credits = [credit.replace(" ", "") for credit in text_box_credit_all] 
                grades = [grade.replace(" ", "") for grade in text_box_academic_results_all]
                
                student_name = text_box_student_name.replace(" ", "")
                student_lastname = text_box_student_lastname.replace(" ", "")

                combined_list = list(zip(subject_codes, subject_names, credits, grades))

                enrolled_subjects = []
                for scode, sname, credit, grade in zip(subject_codes, subject_names, credits, grades):
                    enrolled_subjects.append({
                        "subject_code": scode,
                        "subject_name": sname,
                        "credit": credit,
                        "grade": grade,
                    })
            
                # สร้างโครงสร้าง JSON ในรูปแบบของ dictionary
                data = {
                    "student_info": {
                        "name": student_name,
                        "lastname": student_lastname,
                    },
                    "enrolled_subjects": enrolled_subjects
                }

                # แปลง Python dictionary เป็น JSON string
                json_path = media_path / "transcript.json"
                with json_path.open("w", encoding="utf-8") as json_file:
                    json.dump([data], json_file, indent=4, ensure_ascii=False)

                return render(request, 'high_school_tesseract.html', {
                    'saved_images': saved_images,
                    'student_name':student_name,
                    'student_lastname':student_lastname,               
                    'combined_list':combined_list,
                    'json_url': settings.MEDIA_URL + "transcript.json"
                })             
            
    else:
        return render(request, 'high_school_tesseract.html')

def high_school(request):
    if request.method == 'POST' and 'file_uploads' in request.FILES:
        uploaded_files = request.FILES.getlist('file_uploads')  # รับไฟล์ทั้งหมด

        media_path = Path(settings.MEDIA_ROOT)  # ใช้ pathlib.Path
        media_path.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ media/ ถ้ายังไม่มี

        saved_images = []
        pillow_images = []
        for idx, file in enumerate(uploaded_files):
            file_type = fc.check_file_type(file)

            if file_type == "Image":
                # บันทึกเฉพาะไฟล์รูปภาพลงโฟลเดอร์ media/
                image_path = media_path / file.name
                with image_path.open('wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                img = Image.open(file)
                pillow_images.append(img)
                        
                image_url = settings.MEDIA_URL + file.name
                saved_images.append(image_url)
                messages.success(request, f"อัปโหลด {file.name} สำเร็จ!")

            elif file_type == "PDF":
                messages.warning(request, f"อัปโหลด {file.name} สำเร็จ แต่ยังไม่ได้บันทึก (PDF)")

            else:
                messages.error(request, f"{file.name} ไม่ใช่ไฟล์ที่รองรับ!") 

        output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_HS_DJ/front")
        output_folder.mkdir(exist_ok=True)

        print("pillow_images:", len(pillow_images))

        if len(saved_images) > 0:
            # f คือ front
            if len(saved_images) == 1:
                f_image = pillow_images[0]
            else:
                f_image = pillow_images[1]
            #f_image = Image.open(f"..{saved_images[0]}")
            f_new_size = (1660, 2347)
            f_resized_pil = f_image.resize(f_new_size, Image.LANCZOS)
            f_img_rgb = np.array(f_resized_pil)
            f_img_cv = cv2.cvtColor(f_img_rgb, cv2.COLOR_RGB2BGR)
            f_denoised = cv2.bilateralFilter(f_img_cv, d=9, sigmaColor=75, sigmaSpace=75) # จำกัด noise
            f_gray_img = cv2.cvtColor(f_denoised, cv2.COLOR_BGR2GRAY)
        
            f_binary_gaussian = cv2.adaptiveThreshold(
                f_gray_img, 
                maxValue=255, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV, 
                blockSize=51, #51
                C=9 #21 #15
            )

            # สร้าง kernel สำหรับ morphological operation
            kernel = np.ones((3, 3), np.uint8)
            f_dilated = cv2.dilate(f_binary_gaussian, kernel, iterations=1)
            f_closed_dummy = cv2.morphologyEx(f_binary_gaussian, cv2.MORPH_CLOSE, kernel, iterations=1)

            cv2.imwrite(f"{output_folder}/img_cv.png", f_img_cv)
            cv2.imwrite(f"{output_folder}/denoised.png", f_denoised)
            cv2.imwrite(f"{output_folder}/gray.png", f_gray_img)
            cv2.imwrite(f"{output_folder}/binary_g.png", f_binary_gaussian)
            cv2.imwrite(f"{output_folder}/dilated.png", f_dilated)
            cv2.imwrite(f"{output_folder}/closed_dummy.png", f_closed_dummy)

            table_img, table_dummy_img, table_original_img, student_info_img, student_info_fh_img, student_info_sh_img = hs.split_grade_table_and_students(f_binary_gaussian, f_denoised, f_dilated)
            table_persective_img, table_original_persective_img, table_dummy_persective_img = hs.persective_transformation(f_binary_gaussian, f_denoised, f_dilated)

            cv2.imwrite(f"{output_folder}/table_img.png", table_img)
            cv2.imwrite(f"{output_folder}/table_dummy_img.png", table_dummy_img)
            cv2.imwrite(f"{output_folder}/table_original_img.png", table_original_img)
            cv2.imwrite(f"{output_folder}/student_info_img.png", student_info_img)
            cv2.imwrite(f"{output_folder}/student_info_fh_img.png", student_info_fh_img)
            cv2.imwrite(f"{output_folder}/student_info_sh_img.png", student_info_sh_img)

            cv2.imwrite(f"{output_folder}/table_persective_img.png", table_persective_img)
            cv2.imwrite(f"{output_folder}/table_original_persective_img.png", table_original_persective_img)
            cv2.imwrite(f"{output_folder}/table_dummy_persective_img.png", table_dummy_persective_img)

            # หา column ของตาราง
            cell_column_images = hs.find_table_columns(table_dummy_persective_img, table_persective_img)

            # แบ่ง column ตาราง
            cell_subject_code_name_img = hs.crop_top_bottom(cell_column_images[0])
            cell_credit_img = hs.crop_top_bottom(cell_column_images[1])
            cell_academic_results_img = hs.crop_top_bottom(cell_column_images[2])

            cell_subject_code_name_img_2 = hs.crop_top_bottom(cell_column_images[3])
            cell_credit_img_2 = hs.crop_top_bottom(cell_column_images[4])
            cell_academic_results_img_2 = hs.crop_top_bottom(cell_column_images[5])

            cell_subject_code_name_img_3 = hs.crop_top_bottom(cell_column_images[6])
            cell_credit_img_3 = hs.crop_top_bottom(cell_column_images[7])
            cell_academic_results_img_3 = hs.crop_top_bottom(cell_column_images[8])

            # จับกลุ่มข้อความของ cell ตาราง
            text_academic_results_images, calculate_line_stats_1, academic_results_img = hs.detect_text_in_cell(cell_academic_results_img, 1)
            text_subject_code_name_images, subject_code_name_img = hs.detect_text_in_cell(cell_subject_code_name_img, 2, calculate_line_stats_1)
            text_credit_images, credit_img = hs.detect_text_in_cell(cell_credit_img, 2, calculate_line_stats_1)

            text_academic_results_images_2, calculate_line_stats_2, academic_results_img_2 = hs.detect_text_in_cell(cell_academic_results_img_2, 1)
            text_subject_code_name_images_2, subject_code_name_img_2 = hs.detect_text_in_cell(cell_subject_code_name_img_2, 2, calculate_line_stats_2)
            text_credit_images_2, credit_img_2 = hs.detect_text_in_cell(cell_credit_img_2, 2, calculate_line_stats_2)

            text_academic_results_images_3, calculate_line_stats_3, academic_results_img_3 = hs.detect_text_in_cell(cell_academic_results_img_3, 1)
            text_subject_code_name_images_3, subject_code_name_img_3 = hs.detect_text_in_cell(cell_subject_code_name_img_3, 2, calculate_line_stats_3)
            text_credit_images_3, credit_img_3 = hs.detect_text_in_cell(cell_credit_img_3, 2, calculate_line_stats_3)

            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results.jpg", academic_results_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_name.jpg", subject_code_name_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit.jpg", credit_img)

            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_2.jpg", academic_results_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_name_2.jpg", subject_code_name_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_2.jpg", credit_img_2)

            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_3.jpg", academic_results_img_3)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_name_3.jpg", subject_code_name_img_3)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_3.jpg", credit_img_3)

            # จับข้อความย่อยในกลุ่มข้อความ
            text_group_subject_code_name = hs.detect_sub_text_in_group(text_subject_code_name_images)
            text_group_subject_code, text_group_subject_name = hs.separate_subject_code_and_name(text_group_subject_code_name)
            text_group_credit = hs.detect_sub_text_in_group(text_credit_images)
            text_group_academic_results = hs.detect_sub_text_in_group(text_academic_results_images)

            text_group_subject_code_name_2 = detect_sub_text_in_group(text_subject_code_name_images_2)
            text_group_subject_code_2, text_group_subject_name_2 = hs.separate_subject_code_and_name(text_group_subject_code_name_2)
            text_group_credit_2 = hs.detect_sub_text_in_group(text_credit_images_2)
            text_group_academic_results_2 = hs.detect_sub_text_in_group(text_academic_results_images_2)

            text_group_subject_code_name_3 = detect_sub_text_in_group(text_subject_code_name_images_3)
            text_group_subject_code_3, text_group_subject_name_3 = hs.separate_subject_code_and_name(text_group_subject_code_name_3)
            text_group_credit_3 = hs.detect_sub_text_in_group(text_credit_images_3)
            text_group_academic_results_3 = hs.detect_sub_text_in_group(text_academic_results_images_3)

            # จับตัวอักษร 1 ระดับ
            text_group_char_subject_code = hs.detect_one_level_of_char(text_group_subject_code[:])
            text_group_char_subject_code_2 = hs.detect_one_level_of_char(text_group_subject_code_2[:])
            text_group_char_subject_code_3 = hs.detect_one_level_of_char(text_group_subject_code_3[:])

            text_group_char_credit = hs.detect_one_level_of_char(text_group_credit[:])
            text_group_char_credit_2 = hs.detect_one_level_of_char(text_group_credit_2[:])
            text_group_char_credit_3 = hs.detect_one_level_of_char(text_group_credit_3[:])

            text_group_char_academic_results = hs.detect_one_level_of_char(text_group_academic_results[:])
            text_group_char_academic_results_2 = hs.detect_one_level_of_char(text_group_academic_results_2[:])
            text_group_char_academic_results_3 = hs.detect_one_level_of_char(text_group_academic_results_3[:])


            
            # โหลดโมเดล ตัวอักษร 1 ระดับ
            model_path = Path("C:/Users/Impan/Documents/ocr-engine-python/models")
            model_path_char_subject_code_hs = f"{model_path}/char_subject_code_hs_model.h5"
            model_path_char_credit = f"{model_path}/char_credit_model.h5"
            model_path_char_academic_results_hs = f"{model_path}/char_academic_results_hs_model.h5"

            model_char_subject_code_hs = load_model(model_path_char_subject_code_hs)
            model_char_credit= load_model(model_path_char_credit)
            model_char_academic_results_hs= load_model(model_path_char_academic_results_hs)

            # การทำนายตัวอักษร 1 ระดับ
            text_box_subject_code = hs.predict_text_one_level(text_group_char_subject_code, 0, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)
            text_box_subject_code_2 = hs.predict_text_one_level(text_group_char_subject_code_2, 0, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)
            text_box_subject_code_3 = hs.predict_text_one_level(text_group_char_subject_code_3, 0, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)

            text_box_credit = hs.predict_text_one_level(text_group_char_credit, 1, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)
            text_box_credit_2 = hs.predict_text_one_level(text_group_char_credit_2, 1, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)
            text_box_credit_3 = hs.predict_text_one_level(text_group_char_credit_3, 1, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)

            text_box_academic_results = hs.predict_text_one_level(text_group_char_academic_results, 2, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)
            text_box_academic_results_2 = hs.predict_text_one_level(text_group_char_academic_results_2, 2, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)
            text_box_academic_results_3 = hs.predict_text_one_level(text_group_char_academic_results_3, 2, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs)

            # ลบโมเดลทั้งหมดและคืนหน่วยความจำ
            del model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs
            tf.keras.backend.clear_session()  # เคลียร์ session ของ TensorFlow
            gc.collect()

            
            # จับตัวอักษรหลายระดับ
            text_group_char_subject_name = hs.detect_char(text_group_subject_name[:])
            text_group_char_subject_name_2 = hs.detect_char(text_group_subject_name_2[:])
            text_group_char_subject_name_3 = hs.detect_char(text_group_subject_name_3[:])

            # โหลดโมเดลหลายระดับ
            model_path_char_level_0 = f"{model_path}/char_level_0_model.h5"
            model_path_char_level_1 = f"{model_path}/char_level_1_model.h5"
            model_path_char_level_2 = f"{model_path}/char_level_2_model.h5"

            model_char_level_0 = load_model(model_path_char_level_0)
            model_char_level_1 = load_model(model_path_char_level_1)
            model_char_level_2 = load_model(model_path_char_level_2)

            text_box_subject_name = hs.predict_text_multi_level(text_group_char_subject_name[:], model_char_level_0, model_char_level_1, model_char_level_2)
            text_box_subject_name_2 = hs.predict_text_multi_level(text_group_char_subject_name_2[:], model_char_level_0, model_char_level_1, model_char_level_2)
            text_box_subject_name_3 = hs.predict_text_multi_level(text_group_char_subject_name_3[:], model_char_level_0, model_char_level_1, model_char_level_2)

            # ลบโมเดลทั้งหมดและคืนหน่วยความจำ
            del model_char_level_0, model_char_level_1, model_char_level_2
            tf.keras.backend.clear_session()  # เคลียร์ session ของ TensorFlow
            gc.collect()

            # หาชื่อกับนามสกุล
            name_coordinate, lastname_coordinate = hs.find_text_student_info_fh(student_info_fh_img)
            text_name_img, text_lastname_img = hs.find_text_student_info_sh(student_info_sh_img, name_coordinate, lastname_coordinate)

            # จับกลุ่มข้อความย่อย
            text_group_student_name = hs.detect_sub_text_in_group_stud(text_name_img)
            text_group_student_lastname = hs.detect_sub_text_in_group_stud(text_lastname_img)

            # จับตัวอักษรหลายระดับ
            text_group_char_student_name = hs.detect_char_stud(text_group_student_name[1:])
            text_group_char_student_lastname = hs.detect_char_stud(text_group_student_lastname[1:])
            
            # ทำนายตัวอักษรหลายระดับ
            # โหลดโมเดลหลายระดับ
            model_char_level_0 = load_model(model_path_char_level_0)
            model_char_level_1 = load_model(model_path_char_level_1)
            model_char_level_2 = load_model(model_path_char_level_2)

            text_box_student_name = hs.predict_text_multi_level_stud(text_group_char_student_name[:], model_char_level_0, model_char_level_1, model_char_level_2)
            text_box_student_lastname = hs.predict_text_multi_level_stud(text_group_char_student_lastname[:], model_char_level_0, model_char_level_1, model_char_level_2)

            # ลบโมเดลทั้งหมดและคืนหน่วยความจำ
            del model_char_level_0, model_char_level_1, model_char_level_2
            tf.keras.backend.clear_session()  # เคลียร์ session ของ TensorFlow
            gc.collect()
            
            if len(pillow_images) == 2:
                output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_HS_DJ/back")
                output_folder.mkdir(exist_ok=True)

                ## หน้าหลัง
                b_image = pillow_images[0]
                b_new_size = (1660, 2347)
                b_resized_pil = b_image.resize(b_new_size, Image.LANCZOS)
                b_img_rgb = np.array(b_resized_pil)
                b_img_cv = cv2.cvtColor(b_img_rgb, cv2.COLOR_RGB2BGR)
                b_denoised = cv2.bilateralFilter(b_img_cv, d=9, sigmaColor=75, sigmaSpace=75) # จำกัด noise
                b_gray_img = cv2.cvtColor(b_denoised, cv2.COLOR_BGR2GRAY)
        
                b_binary_gaussian = cv2.adaptiveThreshold(
                    b_gray_img, 
                    maxValue=255, 
                    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    thresholdType=cv2.THRESH_BINARY_INV, 
                    blockSize=51, #51
                    C=9 #21 #15
                )

                # สร้าง kernel สำหรับ morphological operation
                kernel = np.ones((3, 3), np.uint8)
                b_dilated = cv2.dilate(b_binary_gaussian, kernel, iterations=1)
                b_closed_dummy = cv2.morphologyEx(b_binary_gaussian, cv2.MORPH_CLOSE, kernel, iterations=1)

                cv2.imwrite(f"{output_folder}/img_cv.png", b_img_cv)
                cv2.imwrite(f"{output_folder}/denoised.png", b_denoised)
                cv2.imwrite(f"{output_folder}/gray.png", b_gray_img)
                cv2.imwrite(f"{output_folder}/binary_g.png", b_binary_gaussian)
                cv2.imwrite(f"{output_folder}/dilated.png", b_dilated)
                cv2.imwrite(f"{output_folder}/closed_dummy.png", b_closed_dummy)

                ## หาตาราง
                table_img, table_dummy_img, table_original_im = hs.fine_table(b_binary_gaussian, b_denoised, b_dilated)
                table_persective_img, table_original_persective_img, table_dummy_persective_img = hs.persective_transformation(b_binary_gaussian, b_denoised, b_dilated)

                cv2.imwrite(f"{output_folder}/table_img.png", table_img)
                cv2.imwrite(f"{output_folder}/table_dummy_img.png", table_dummy_img)

                cv2.imwrite(f"{output_folder}/table_persective_img.png", table_persective_img)
                cv2.imwrite(f"{output_folder}/table_original_persective_img.png", table_original_persective_img)
                cv2.imwrite(f"{output_folder}/table_dummy_persective_img.png", table_dummy_persective_img)

                # หา column ของตาราง
                cell_column_images, cell_column_dummy_images = hs.find_table_columns_b(table_dummy_persective_img, table_persective_img)

                # แบ่ง column ตาราง
                cell_1 = cell_column_images[0]
                cell_2 = hs.crop_top_bottom(cell_column_images[1], 20, 40)
                
                department_credits_img, department_academic_results_img, sum_department_credits_img, gpa_img = hs.find_table_subject_group(cell_2)

                # หากลุ่มข้อความของตาราง subject_group
                texts_department_credits = hs.detect_text_subject_group(department_credits_img)
                texts_department_academic_results = hs.detect_text_subject_group(department_academic_results_img)

                texts_sum_department_credits = hs.detect_text_subject_group(sum_department_credits_img)
                texts_gpa = hs.detect_text_subject_group(gpa_img)

                # จับตัวอักษร 1 ระดับ
                text_group_char_department_credits = hs.detect_one_level_of_char_back(texts_department_credits[:])
                text_group_char_department_academic_results = hs.detect_one_level_of_char_back(texts_department_academic_results[:])

                text_group_char_sum_department_credits = hs.detect_one_level_of_char_back(texts_sum_department_credits[:])
                text_group_char_gpa = hs.detect_one_level_of_char_back(texts_gpa[:])

                # ทำนายตัวอักษร 1 ระดับ
                model_char_credit= load_model(model_path_char_credit)

                text_box_department_credits = hs.predict_text_one_level_b(text_group_char_department_credits, 1, model_char_credit)
                text_box_department_academic_results = hs.predict_text_one_level_b(text_group_char_department_academic_results, 1, model_char_credit)
                text_box_sum_department_credits = hs.predict_text_one_level_b(text_group_char_sum_department_credits, 1, model_char_credit)
                text_box_gpa = hs.predict_text_one_level_b(text_group_char_gpa, 1, model_char_credit)

                # ลบโมเดลทั้งหมดและคืนหน่วยความจำ
                del model_char_credit
                tf.keras.backend.clear_session()  # เคลียร์ session ของ TensorFlow
                gc.collect()

                print(text_box_department_credits)
                print(text_box_department_academic_results)
                print(text_box_sum_department_credits)
                print(text_box_gpa)

            # JSON
            if(len(pillow_images) == 2):
                # มีทั้งหน้าทั้งหลัง
                # รวมข้อมูล
                text_box_subject_code_all = text_box_subject_code + text_box_subject_code_2 + text_box_subject_code_3
                text_box_subject_name_all = text_box_subject_name + text_box_subject_name_2 + text_box_subject_name_3
                text_box_credit_all = text_box_credit + text_box_credit_2 + text_box_credit_3
                text_box_academic_results_all = text_box_academic_results + text_box_academic_results_2 + text_box_academic_results_3

                #ลบช่องว่าง
                subject_codes = [scode.replace(" ", "") for scode in text_box_subject_code_all]
                subject_names = [sname.strip() for sname in text_box_subject_name_all]
                credits = [credit.replace(" ", "") for credit in text_box_credit_all] 
                grades = [grade.replace(" ", "") for grade in text_box_academic_results_all]
                
                student_name = text_box_student_name.replace(" ", "")
                student_lastname = text_box_student_lastname.replace(" ", "")

                department_credits = [departcredit.replace(" ", "") for departcredit in text_box_department_credits]
                department_academic_results = [ac.replace(" ", "") for ac in text_box_department_academic_results]
                sum_department_credits = [sum.replace(" ", "") for sum in text_box_sum_department_credits]
                gpa = [gpa.replace(" ", "") for gpa in text_box_gpa]
            
                combined_list = list(zip(subject_codes, subject_names, credits, grades))

                enrolled_subjects = []
                for scode, sname, credit, grade in zip(subject_codes, subject_names, credits, grades):
                    enrolled_subjects.append({
                        "subject_code": scode,
                        "subject_name": sname,
                        "credit": credit,
                        "grade": grade,
                    })
            
                # สร้างโครงสร้าง JSON ในรูปแบบของ dictionary
                data = {
                    "student_info": {
                        "name": student_name,
                        "lastname": student_lastname,
                        "math_credits": department_credits[1],
                        "math_grade": department_academic_results[1],
                        "science_credits": department_credits[2],
                        "science_grade": department_academic_results[2],
                        "gpa": gpa[0],
                    },
                    "enrolled_subjects": enrolled_subjects
                }

                # แปลง Python dictionary เป็น JSON string
                json_path = media_path / "transcript.json"
                with json_path.open("w", encoding="utf-8") as json_file:
                    json.dump([data], json_file, indent=4, ensure_ascii=False)

                return render(request, 'high_school_tesseract.html', {
                    'saved_images': saved_images,
                    'student_name':student_name,
                    'student_lastname':student_lastname,
                    "math_credits": department_credits[1],
                    "math_grade": department_academic_results[1],
                    "science_credits": department_credits[2],
                    "science_grade": department_academic_results[2],
                    "gpa": gpa[0],                    
                    'combined_list':combined_list,
                    'json_url': settings.MEDIA_URL + "transcript.json"
                })
            else:
                # มีแค่ภาพด้านหน้า
                # รวมข้อมูล
                text_box_subject_code_all = text_box_subject_code + text_box_subject_code_2 + text_box_subject_code_3
                text_box_subject_name_all = text_box_subject_name + text_box_subject_name_2 + text_box_subject_name_3
                text_box_credit_all = text_box_credit + text_box_credit_2 + text_box_credit_3
                text_box_academic_results_all = text_box_academic_results + text_box_academic_results_2 + text_box_academic_results_3

                #ลบช่องว่าง
                subject_codes = [scode.replace(" ", "") for scode in text_box_subject_code_all]
                subject_names = [sname.strip() for sname in text_box_subject_name_all]
                credits = [credit.replace(" ", "") for credit in text_box_credit_all] 
                grades = [grade.replace(" ", "") for grade in text_box_academic_results_all]
                
                student_name = text_box_student_name.replace(" ", "")
                student_lastname = text_box_student_lastname.replace(" ", "")

                combined_list = list(zip(subject_codes, subject_names, credits, grades))

                enrolled_subjects = []
                for scode, sname, credit, grade in zip(subject_codes, subject_names, credits, grades):
                    enrolled_subjects.append({
                        "subject_code": scode,
                        "subject_name": sname,
                        "credit": credit,
                        "grade": grade,
                    })
            
                # สร้างโครงสร้าง JSON ในรูปแบบของ dictionary
                data = {
                    "student_info": {
                        "name": student_name,
                        "lastname": student_lastname,
                    },
                    "enrolled_subjects": enrolled_subjects
                }

                # แปลง Python dictionary เป็น JSON string
                json_path = media_path / "transcript.json"
                with json_path.open("w", encoding="utf-8") as json_file:
                    json.dump([data], json_file, indent=4, ensure_ascii=False)

                return render(request, 'high_school_tesseract.html', {
                    'saved_images': saved_images,
                    'student_name':student_name,
                    'student_lastname':student_lastname,               
                    'combined_list':combined_list,
                    'json_url': settings.MEDIA_URL + "transcript.json"
                }) 


    return render(request, 'high_school.html')

def technician_tesseract(request):
    if request.method == 'POST' and 'file_uploads' in request.FILES:
        uploaded_files = request.FILES.getlist('file_uploads')  # รับไฟล์ทั้งหมด

        media_path = Path(settings.MEDIA_ROOT)  # ใช้ pathlib.Path
        media_path.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ media/ ถ้ายังไม่มี

        saved_images = []
        pillow_images = []
        for idx, file in enumerate(uploaded_files):
            file_type = fc.check_file_type(file)

            if file_type == "Image":
                # บันทึกเฉพาะไฟล์รูปภาพลงโฟลเดอร์ media/
                image_path = media_path / file.name
                with image_path.open('wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                img = Image.open(file)
                pillow_images.append(img)
                        
                image_url = settings.MEDIA_URL + file.name
                saved_images.append(image_url)
                messages.success(request, f"อัปโหลด {file.name} สำเร็จ!")

            elif file_type == "PDF":
                messages.warning(request, f"อัปโหลด {file.name} สำเร็จ แต่ยังไม่ได้บันทึก (PDF)")

            else:
                messages.error(request, f"{file.name} ไม่ใช่ไฟล์ที่รองรับ!") 

        output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_TSR_DJ")
        output_folder.mkdir(exist_ok=True)

        print("pillow_images:", len(pillow_images))

        if len(saved_images) > 0:
            # f คือ front
            if len(saved_images) == 1:
                f_image = pillow_images[0]
            else:
                f_image = pillow_images[1]
            #f_image = Image.open(f"..{saved_images[0]}")
            f_new_size = (1660, 2347)
            f_resized_pil = f_image.resize(f_new_size, Image.LANCZOS)
            f_img_rgb = np.array(f_resized_pil)
            f_img_cv = cv2.cvtColor(f_img_rgb, cv2.COLOR_RGB2BGR)
            f_denoised = cv2.bilateralFilter(f_img_cv, d=9, sigmaColor=75, sigmaSpace=75) # จำกัด noise
            f_gray_img = cv2.cvtColor(f_denoised, cv2.COLOR_BGR2GRAY)
        
            f_binary_gaussian = cv2.adaptiveThreshold(
                f_gray_img, 
                maxValue=255, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV, 
                blockSize=51, #51
                C=9 #21 #15
            )

            # สร้าง kernel สำหรับ morphological operation
            kernel = np.ones((3, 3), np.uint8)
            f_dilated = cv2.dilate(f_binary_gaussian, kernel, iterations=1)
            f_closed_dummy = cv2.morphologyEx(f_binary_gaussian, cv2.MORPH_CLOSE, kernel, iterations=1)

            cv2.imwrite(f"{output_folder}/img_cv.png", f_img_cv)
            cv2.imwrite(f"{output_folder}/denoised.png", f_denoised)
            cv2.imwrite(f"{output_folder}/gray.png", f_gray_img)
            cv2.imwrite(f"{output_folder}/binary_g.png", f_binary_gaussian)
            cv2.imwrite(f"{output_folder}/dilated.png", f_dilated)
            cv2.imwrite(f"{output_folder}/closed_dummy.png", f_closed_dummy)

            # แยกตารางเกรดกับข้อมูลนักศึกษา
            table_img, table_dummy_img, table_original_img, student_info_img, student_info_fh_img, student_info_sh_img = tn_tsr.split_grade_table_and_students(f_binary_gaussian, f_denoised, f_dilated)
            table_persective_img, table_original_persective_img, table_dummy_persective_img = tn_tsr.persective_transformation(f_binary_gaussian, f_denoised, f_dilated)

            cv2.imwrite(f"{output_folder}/table_img.png", table_img)
            cv2.imwrite(f"{output_folder}/table_dummy_img.png", table_dummy_img)
            cv2.imwrite(f"{output_folder}/table_original_img.png", table_original_img)
            cv2.imwrite(f"{output_folder}/student_info_img.png", student_info_img)
            cv2.imwrite(f"{output_folder}/student_info_fh_img.png", student_info_fh_img)
            cv2.imwrite(f"{output_folder}/student_info_sh_img.png", student_info_sh_img)

            cv2.imwrite(f"{output_folder}/table_persective_img.png", table_persective_img)
            cv2.imwrite(f"{output_folder}/table_original_persective_img.png", table_original_persective_img)
            cv2.imwrite(f"{output_folder}/table_dummy_persective_img.png", table_dummy_persective_img)

            ## ตารางเกรด
            # หา column ของตาราง

            cell_images = tn_tsr.find_table_columns_rows(table_dummy_persective_img, table_persective_img)

            cell_subject_code_img = cell_images[0]
            cell_subject_name_img = cell_images[1]
            cell_credit_img = cell_images[2]
            cell_academic_results_img = cell_images[3]
            cell_subject_code_img_2 = cell_images[5]
            cell_subject_name_img_2 = cell_images[6]
            cell_credit_img_2 = cell_images[7]
            cell_academic_results_img_2 = cell_images[8]

            # จับกลุ่มข้อความของ cell ตาราง
            # ตารางครึ่งแรก
            text_subject_code_images, calculate_line_stats_1, subject_code_img = tn_tsr.detect_text_group_in_cell(cell_subject_code_img, 1)
            text_subject_name_images, subject_name_img = tn_tsr.detect_text_group_in_cell(cell_subject_name_img, 2, calculate_line_stats_1)
            text_credit_images, credit_img = tn_tsr.detect_text_group_in_cell(cell_credit_img, 2, calculate_line_stats_1)
            text_academic_results_images, academic_results_img = tn_tsr.detect_text_group_in_cell(cell_academic_results_img, 2, calculate_line_stats_1)

            # ตารางครึ่งหลัง
            text_subject_code_images_2, calculate_line_stats_2, subject_code_img_2 = tn_tsr.detect_text_group_in_cell(cell_subject_code_img_2, 1)
            text_subject_name_images_2, subject_name_img_2 = tn_tsr.detect_text_group_in_cell(cell_subject_name_img_2, 2, calculate_line_stats_2)
            text_credit_images_2, credit_img_2 = tn_tsr.detect_text_group_in_cell(cell_credit_img_2, 2, calculate_line_stats_2)
            text_academic_results_images_2, academic_results_img_2 = tn_tsr.detect_text_group_in_cell(cell_academic_results_img_2, 2, calculate_line_stats_2)

            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_img.jpg", subject_code_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_name_img.jpg", subject_name_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_img.jpg", credit_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_img.jpg", academic_results_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_img_2.jpg", subject_code_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_name_img_2.jpg", subject_name_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_img_2.jpg", credit_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_img_2.jpg", academic_results_img_2)

            # จับข้อความย่อยในกลุ่มข้อความของ cell ตาราง
            text_group_subject_code = tn_tsr.detect_sub_text_in_group(text_subject_code_images)
            text_group_subject_name = tn_tsr.detect_sub_text_in_group(text_subject_name_images)
            text_group_credit = tn_tsr.detect_sub_text_in_group(text_credit_images)
            text_group_academic_results = tn_tsr.detect_sub_text_in_group(text_academic_results_images)

            text_group_subject_code_2 = tn_tsr.detect_sub_text_in_group(text_subject_code_images_2)
            text_group_subject_name_2 = tn_tsr.detect_sub_text_in_group(text_subject_name_images_2)
            text_group_credit_2 = tn_tsr.detect_sub_text_in_group(text_credit_images_2)
            text_group_academic_results_2 = tn_tsr.detect_sub_text_in_group(text_academic_results_images_2)

            # การทำนาย
            text_box_subject_code = tn_tsr.predict_text(text_group_subject_code, 1)
            text_box_subject_name = tn_tsr.predict_text(text_group_subject_name)
            text_box_credit = tn_tsr.predict_text(text_group_credit, 3)
            text_box_academic_results = tn_tsr.predict_text(text_group_academic_results, 0)
            text_box_subject_code_2 = tn_tsr.predict_text(text_group_subject_code_2, 1)
            text_box_subject_name_2 = tn_tsr.predict_text(text_group_subject_name_2)
            text_box_credit_2 = tn_tsr.predict_text(text_group_credit_2, 3)
            text_box_academic_results_2 = tn_tsr.predict_text(text_group_academic_results_2, 0)
            
            ## ข้อมูลนักศึกษา
            text_stud_fh_images = tn_tsr.find_text_student_info_fh(student_info_fh_img)
            text_stud_sh_images = tn_tsr.find_text_student_info_sh(student_info_sh_img)

            indices_fh = [3, -2, -1]
            indices_sh = [-3, -1]
            student_name, field_of_study, field_of_work = [text_stud_fh_images[i] for i in indices_fh]
            gpa, graduation_date = [text_stud_sh_images[i] for i in indices_sh]

            # จับกลุ่มข้อความย่อย
            text_group_student_name = tn_tsr.detect_sub_text_in_group_stud(student_name)
            text_group_field_of_study = tn_tsr.detect_sub_text_in_group_stud(field_of_study)
            text_group_field_of_work = tn_tsr.detect_sub_text_in_group_stud(field_of_work)

            text_group_gpa = tn_tsr.detect_sub_text_in_group_stud(gpa)
            text_group_graduation_date = tn_tsr.detect_sub_text_in_group_stud(graduation_date)

            # ทำนาย
            text_box_student_name = tn_tsr.predict_text_stud(text_group_student_name[3:], 1)
            text_box_field_of_study = tn_tsr.predict_text_stud(text_group_field_of_study[1:], 1)
            text_box_field_of_work = tn_tsr.predict_text_stud(text_group_field_of_work[1:], 1)

            text_box_gpa = tn_tsr.predict_text_stud(text_group_gpa[1:], 1)
            text_box_graduation_date = tn_tsr.predict_text_stud(text_group_graduation_date[1:], 1)

            ## JSON
            # ทำการรวม list
            text_box_subject_code_all = text_box_subject_code + text_box_subject_code_2
            text_box_subject_name_all = text_box_subject_name + text_box_subject_name_2
            text_box_credit_all = text_box_credit + text_box_credit_2
            text_box_academic_results_all = text_box_academic_results + text_box_academic_results_2

            # ลบช่องว่างด้วย list comprehension
            subject_codes = [scode.replace(" ", "") for scode in text_box_subject_code_all]
            subject_names = [sname.strip() for sname in text_box_subject_name_all]
            credits = [credit.replace(" ", "") for credit in text_box_credit_all] 
            grades = [grade.replace(" ", "") for grade in text_box_academic_results_all]

            combined_list = list(zip(subject_codes, subject_names, credits, grades))

            student_name = text_box_student_name.strip()
            field_of_study = text_box_field_of_study.replace(" ", "")
            field_of_work = text_box_field_of_work.replace(" ", "")
            gpa = text_box_gpa.replace(" ", "")
            graduation_date = text_box_graduation_date.strip()

            enrolled_subjects = []
            for scode, sname, credit, grade in zip(subject_codes, subject_names, credits, grades):
                enrolled_subjects.append({
                "subject_code": scode,
                "subject_name": sname,
                "credit": credit,
                "grade": grade
            })
                
            # สร้างโครงสร้าง JSON ในรูปแบบของ dictionary
            data = {
                "student_info": {
                    "fullname": student_name,
                    "field_of_study": field_of_study,
                    "field_of_work": field_of_work,
                    "gpa": gpa,
                    "graduation_date":graduation_date
                },
                "enrolled_subjects": enrolled_subjects
            }

            # แปลง Python dictionary เป็น JSON string
            json_path = media_path / "transcript.json"
            with json_path.open("w", encoding="utf-8") as json_file:
                json.dump([data], json_file, indent=4, ensure_ascii=False)
                
            return render(request, 'technician_tesseract.html', {
                'saved_images': saved_images,
                'fullname':student_name,
                'field_of_study': field_of_study,
                'field_of_work': field_of_work,
                'gpa': gpa,
                'graduation_date':graduation_date,               
                'combined_list':combined_list,
                'json_url': settings.MEDIA_URL + "transcript.json"
            })               

    else:
        return render(request, 'technician_tesseract.html')

def technician(request):
    if request.method == 'POST' and 'file_uploads' in request.FILES:
        uploaded_files = request.FILES.getlist('file_uploads')  # รับไฟล์ทั้งหมด

        media_path = Path(settings.MEDIA_ROOT)  # ใช้ pathlib.Path
        media_path.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ media/ ถ้ายังไม่มี

        saved_images = []
        pillow_images = []
        for idx, file in enumerate(uploaded_files):
            file_type = fc.check_file_type(file)

            if file_type == "Image":
                # บันทึกเฉพาะไฟล์รูปภาพลงโฟลเดอร์ media/
                image_path = media_path / file.name
                with image_path.open('wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                img = Image.open(file)
                pillow_images.append(img)
                        
                image_url = settings.MEDIA_URL + file.name
                saved_images.append(image_url)
                messages.success(request, f"อัปโหลด {file.name} สำเร็จ!")

            elif file_type == "PDF":
                messages.warning(request, f"อัปโหลด {file.name} สำเร็จ แต่ยังไม่ได้บันทึก (PDF)")

            else:
                messages.error(request, f"{file.name} ไม่ใช่ไฟล์ที่รองรับ!") 

        output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_DJ")
        output_folder.mkdir(exist_ok=True)

        print("pillow_images:", len(pillow_images))

        if len(saved_images) > 0:
            # f คือ front
            if len(saved_images) == 1:
                f_image = pillow_images[0]
            else:
                f_image = pillow_images[1]
            #f_image = Image.open(f"..{saved_images[0]}")
            f_new_size = (1660, 2347)
            f_resized_pil = f_image.resize(f_new_size, Image.LANCZOS)
            f_img_rgb = np.array(f_resized_pil)
            f_img_cv = cv2.cvtColor(f_img_rgb, cv2.COLOR_RGB2BGR)
            f_denoised = cv2.bilateralFilter(f_img_cv, d=9, sigmaColor=75, sigmaSpace=75) # จำกัด noise
            f_gray_img = cv2.cvtColor(f_denoised, cv2.COLOR_BGR2GRAY)
        
            f_binary_gaussian = cv2.adaptiveThreshold(
                f_gray_img, 
                maxValue=255, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV, 
                blockSize=51, #51
                C=9 #21 #15
            )

            # สร้าง kernel สำหรับ morphological operation
            kernel = np.ones((3, 3), np.uint8)
            f_dilated = cv2.dilate(f_binary_gaussian, kernel, iterations=1)
            f_closed_dummy = cv2.morphologyEx(f_binary_gaussian, cv2.MORPH_CLOSE, kernel, iterations=1)

            cv2.imwrite(f"{output_folder}/img_cv.png", f_img_cv)
            cv2.imwrite(f"{output_folder}/denoised.png", f_denoised)
            cv2.imwrite(f"{output_folder}/gray.png", f_gray_img)
            cv2.imwrite(f"{output_folder}/binary_g.png", f_binary_gaussian)
            cv2.imwrite(f"{output_folder}/dilated.png", f_dilated)
            cv2.imwrite(f"{output_folder}/closed_dummy.png", f_closed_dummy)

            # แยกตารางเกรดกับข้อมูลนักศึกษา
            table_img, table_dummy_img, table_original_img, student_info_img, student_info_fh_img, student_info_sh_img = tn.split_grade_table_and_students(f_binary_gaussian, f_denoised, f_dilated)
            table_persective_img, table_original_persective_img, table_dummy_persective_img = tn.persective_transformation(f_binary_gaussian, f_denoised, f_dilated)

            cv2.imwrite(f"{output_folder}/table_img.png", table_img)
            cv2.imwrite(f"{output_folder}/table_dummy_img.png", table_dummy_img)
            cv2.imwrite(f"{output_folder}/table_original_img.png", table_original_img)
            cv2.imwrite(f"{output_folder}/student_info_img.png", student_info_img)
            cv2.imwrite(f"{output_folder}/student_info_fh_img.png", student_info_fh_img)
            cv2.imwrite(f"{output_folder}/student_info_sh_img.png", student_info_sh_img)

            cv2.imwrite(f"{output_folder}/table_persective_img.png", table_persective_img)
            cv2.imwrite(f"{output_folder}/table_original_persective_img.png", table_original_persective_img)
            cv2.imwrite(f"{output_folder}/table_dummy_persective_img.png", table_dummy_persective_img)

            ## ตารางเกรด
            # หา column ของตาราง

            cell_images = tn.find_table_columns_rows(table_dummy_persective_img, table_persective_img)

            cell_subject_code_img = cell_images[0]
            cell_subject_name_img = cell_images[1]
            cell_credit_img = cell_images[2]
            cell_academic_results_img = cell_images[3]
            cell_subject_code_img_2 = cell_images[5]
            cell_subject_name_img_2 = cell_images[6]
            cell_credit_img_2 = cell_images[7]
            cell_academic_results_img_2 = cell_images[8]

            # จับกลุ่มข้อความของ cell ตาราง
            # ตารางครึ่งแรก
            text_subject_code_images, calculate_line_stats_1, subject_code_img = tn.detect_text_group_in_cell(cell_subject_code_img, 1)
            text_subject_name_images, subject_name_img = tn.detect_text_group_in_cell(cell_subject_name_img, 2, calculate_line_stats_1)
            text_credit_images, credit_img = tn.detect_text_group_in_cell(cell_credit_img, 2, calculate_line_stats_1)
            text_academic_results_images, academic_results_img = tn.detect_text_group_in_cell(cell_academic_results_img, 2, calculate_line_stats_1)

            # ตารางครึ่งหลัง
            text_subject_code_images_2, calculate_line_stats_2, subject_code_img_2 = tn.detect_text_group_in_cell(cell_subject_code_img_2, 1)
            text_subject_name_images_2, subject_name_img_2 = tn.detect_text_group_in_cell(cell_subject_name_img_2, 2, calculate_line_stats_2)
            text_credit_images_2, credit_img_2 = tn.detect_text_group_in_cell(cell_credit_img_2, 2, calculate_line_stats_2)
            text_academic_results_images_2, academic_results_img_2 = tn.detect_text_group_in_cell(cell_academic_results_img_2, 2, calculate_line_stats_2)

            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_img.jpg", subject_code_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_name_img.jpg", subject_name_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_img.jpg", credit_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_img.jpg", academic_results_img)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_code_img_2.jpg", subject_code_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_subject_name_img_2.jpg", subject_name_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_credit_img_2.jpg", credit_img_2)
            cv2.imwrite(f"{output_folder}/cell_images/cca_academic_results_img_2.jpg", academic_results_img_2)

            # จับข้อความย่อยในกลุ่มข้อความของ cell ตาราง
            text_group_subject_code = tn.detect_sub_text_in_group(text_subject_code_images)
            text_group_subject_name = tn.detect_sub_text_in_group(text_subject_name_images)
            text_group_credit = tn.detect_sub_text_in_group(text_credit_images)
            text_group_academic_results = tn.detect_sub_text_in_group(text_academic_results_images)

            text_group_subject_code_2 = tn.detect_sub_text_in_group(text_subject_code_images_2)
            text_group_subject_name_2 = tn.detect_sub_text_in_group(text_subject_name_images_2)
            text_group_credit_2 = tn.detect_sub_text_in_group(text_credit_images_2)
            text_group_academic_results_2 = tn.detect_sub_text_in_group(text_academic_results_images_2)


            # จับตัวอักษร
            
            text_group_char_subject_code = tn.detect_one_level_of_char(text_group_subject_code[:])
            text_group_char_subject_name = tn.detect_char(text_group_subject_name[:])
            text_group_char_credit = tn.detect_one_level_of_char(text_group_credit[:])
            text_group_char_academic_results = tn.detect_one_level_of_char(text_group_academic_results[:])


            text_group_char_subject_code_2 = tn.detect_one_level_of_char(text_group_subject_code_2[:])
            text_group_char_subject_name_2 = tn.detect_char(text_group_subject_name_2[:])
            text_group_char_credit_2 = tn.detect_one_level_of_char(text_group_credit_2[:])
            text_group_char_academic_results_2 = tn.detect_one_level_of_char(text_group_academic_results_2[:])

           
            # โหลดโมเดล
            model_path = Path("C:/Users/Impan/Documents/ocr-engine-python/models")
            model_path_char_subject_code_tn = f"{model_path}/char_subject_code_tn_model.h5"
            model_path_char_academic_results_tn = f"{model_path}/char_academic_results_tn_model.h5"

            model_char_subject_code_tn = load_model(model_path_char_subject_code_tn)
            model_char_academic_results_tn= load_model(model_path_char_academic_results_tn)

            # ทำนายตัวอักษร 1 ระดับ
            text_box_subject_code = tn.predict_text_one_level(text_group_char_subject_code[:], 0, model_char_subject_code_tn, model_char_academic_results_tn)
            text_box_credit = tn.predict_text_one_level(text_group_char_credit, 0, model_char_subject_code_tn, model_char_academic_results_tn)
            text_box_academic_results = tn.predict_text_one_level(text_group_char_academic_results, 1, model_char_subject_code_tn, model_char_academic_results_tn)

            text_box_subject_code_2 = tn.predict_text_one_level(text_group_char_subject_code_2[:], 0, model_char_subject_code_tn, model_char_academic_results_tn)
            text_box_credit_2 = tn.predict_text_one_level(text_group_char_credit_2, 0, model_char_subject_code_tn, model_char_academic_results_tn)
            text_box_academic_results_2 = tn.predict_text_one_level(text_group_char_academic_results_2, 1, model_char_subject_code_tn, model_char_academic_results_tn)

            # ลบโมเดลทั้งหมดและคืนหน่วยความจำ
            del model_char_subject_code_tn, model_char_academic_results_tn
            tf.keras.backend.clear_session()  # เคลียร์ session ของ TensorFlow
            gc.collect()

            # โหลดโมเดล
            model_path_char_level_0 = f"{model_path}/char_level_0_model.h5"
            model_path_char_level_1 = f"{model_path}/char_level_1_model.h5"
            model_path_char_level_2 = f"{model_path}/char_level_2_model.h5"

            model_char_level_0 = load_model(model_path_char_level_0)
            model_char_level_1 = load_model(model_path_char_level_1)
            model_char_level_2 = load_model(model_path_char_level_2)

            # ทำนายตัวอักษรหลายระดับ
            text_box_subject_name = tn.predict_text_multi_level(text_group_char_subject_name[:], model_char_level_0, model_char_level_1, model_char_level_2)
            text_box_subject_name_2 = tn.predict_text_multi_level(text_group_char_subject_name_2[:], model_char_level_0, model_char_level_1, model_char_level_2)

            # ลบโมเดลทั้งหมดและคืนหน่วยความจำ
            del model_char_level_0, model_char_level_1, model_char_level_2
            tf.keras.backend.clear_session()  # เคลียร์ session ของ TensorFlow
            gc.collect()

            
            ## ข้อมูลนักศึกษา
            text_stud_fh_images = tn.find_text_student_info_fh(student_info_fh_img)
            text_stud_sh_images = tn.find_text_student_info_sh(student_info_sh_img)

            indices_fh = [3, -2, -1]
            indices_sh = [-3, -1]
            student_name, field_of_study, field_of_work = [text_stud_fh_images[i] for i in indices_fh]
            gpa, graduation_date = [text_stud_sh_images[i] for i in indices_sh]

            # จับกลุ่มข้อความย่อย
            text_group_student_name = tn.detect_sub_text_in_group_stud(student_name)
            text_group_field_of_study = tn.detect_sub_text_in_group_stud(field_of_study)
            text_group_field_of_work = tn.detect_sub_text_in_group_stud(field_of_work)

            text_group_gpa = tn.detect_sub_text_in_group_stud(gpa)
            text_group_graduation_date = tn.detect_sub_text_in_group_stud(graduation_date)

            # จับตัวอักษณหลายระดับ
            text_group_char_student_name = tn.detect_char_stud(text_group_student_name[3:])
            text_group_char_field_of_study = tn.detect_char_stud(text_group_field_of_study[1:])
            text_group_char_field_of_work = tn.detect_char_stud(text_group_field_of_work[1:])

            text_group_char_gpa = tn.detect_char_stud(text_group_gpa[1:])
            text_group_char_graduation_date = tn.detect_char_stud(text_group_graduation_date[1:])


            # โหลดโมเดล
            model_path_char_level_0 = f"{model_path}/char_level_0_model.h5"
            model_path_char_level_1 = f"{model_path}/char_level_1_model.h5"
            model_path_char_level_2 = f"{model_path}/char_level_2_model.h5"

            model_char_level_0 = load_model(model_path_char_level_0)
            model_char_level_1 = load_model(model_path_char_level_1)
            model_char_level_2 = load_model(model_path_char_level_2)

            # ทำนายตัวอักษรหลายระดับ
            text_box_student_name = tn.predict_text_multi_level_stud(text_group_char_student_name[:], model_char_level_0, model_char_level_1, model_char_level_2)
            text_box_field_of_study = tn.predict_text_multi_level_stud(text_group_char_field_of_study[:], model_char_level_0, model_char_level_1, model_char_level_2)
            text_box_field_of_work = tn.predict_text_multi_level_stud(text_group_char_field_of_work[:], model_char_level_0, model_char_level_1, model_char_level_2)

            text_box_gpa = tn.predict_text_multi_level_stud(text_group_char_gpa[:], model_char_level_0, model_char_level_1, model_char_level_2)
            text_box_graduation_date = tn.predict_text_multi_level_stud(text_group_char_graduation_date[:], model_char_level_0, model_char_level_1, model_char_level_2)

            # ลบโมเดลทั้งหมดและคืนหน่วยความจำ
            del model_char_level_0, model_char_level_1, model_char_level_2
            tf.keras.backend.clear_session()  # เคลียร์ session ของ TensorFlow
            gc.collect()


            
            ## JSON
            # ทำการรวม list
            text_box_subject_code_all = text_box_subject_code + text_box_subject_code_2
            text_box_subject_name_all = text_box_subject_name + text_box_subject_name_2
            text_box_credit_all = text_box_credit + text_box_credit_2
            text_box_academic_results_all = text_box_academic_results + text_box_academic_results_2

            # ลบช่องว่างด้วย list comprehension
            subject_codes = [scode.replace(" ", "") for scode in text_box_subject_code_all]
            subject_names = [sname.strip() for sname in text_box_subject_name_all]
            credits = [credit.replace(" ", "") for credit in text_box_credit_all] 
            grades = [grade.replace(" ", "") for grade in text_box_academic_results_all]

            combined_list = list(zip(subject_codes, subject_names, credits, grades))

            student_name = text_box_student_name.strip()
            field_of_study = text_box_field_of_study.replace(" ", "")
            field_of_work = text_box_field_of_work.replace(" ", "")
            gpa = text_box_gpa.replace(" ", "")
            graduation_date = text_box_graduation_date.strip()

            enrolled_subjects = []
            for scode, sname, credit, grade in zip(subject_codes, subject_names, credits, grades):
                enrolled_subjects.append({
                "subject_code": scode,
                "subject_name": sname,
                "credit": credit,
                "grade": grade
            })
                
            # สร้างโครงสร้าง JSON ในรูปแบบของ dictionary
            data = {
                "student_info": {
                    "fullname": student_name,
                    "field_of_study": field_of_study,
                    "field_of_work": field_of_work,
                    "gpa": gpa,
                    "graduation_date":graduation_date
                },
                "enrolled_subjects": enrolled_subjects
            }

            # แปลง Python dictionary เป็น JSON string
            json_path = media_path / "transcript.json"
            with json_path.open("w", encoding="utf-8") as json_file:
                json.dump([data], json_file, indent=4, ensure_ascii=False)
                
            return render(request, 'technician.html', {
                'saved_images': saved_images,
                'fullname':student_name,
                'field_of_study': field_of_study,
                'field_of_work': field_of_work,
                'gpa': gpa,
                'graduation_date':graduation_date,               
                'combined_list':combined_list,
                'json_url': settings.MEDIA_URL + "transcript.json"
            })
            
            
    else:
        return render(request, 'technician.html')

def download_json(request):
    """ ให้ผู้ใช้ดาวน์โหลดไฟล์ JSON """
    json_path = Path(settings.MEDIA_ROOT) / "transcript.json"
    if json_path.exists():
        return FileResponse(json_path.open('rb'), as_attachment=True, filename="transcript.json")
    else:
        return JsonResponse({"error": "JSON file not found"}, status=404)
    


    


