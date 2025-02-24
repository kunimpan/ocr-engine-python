import pytesseract
import cv2
from matplotlib import table
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_image_for_ocr(image):
    # จำกัด noise
    denoised = cv2.bilateralFilter(image, 50, 100, 100)
    gray_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # สำหรับภาพที่แสงไม่สม่ำเสมอ
    binary_gaussian_img = cv2.adaptiveThreshold(
        gray_img, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=41, 
        C=20
    )

    return binary_gaussian_img

def separate_table_and_studentInfo(transcript_img):
    # แยกตาราง
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(transcript_img, connectivity=8) # cca
    areas = [stat[4] for stat in stats]  # ดึงค่า area
    sorted_areas = sorted(areas, reverse=True)  # เรียงลำดับจากมากไปน้อย
    second_max_area = sorted_areas[1]  # ค่าอันดับ 2
    second_max_area_index = areas.index(second_max_area)  # หาตำแหน่งในลิสต์เดิม

    table_position = stats[second_max_area_index]
    x, y, w, h, area = table_position
    table_img = transcript_img[y:y+h, x:x+w]

    # แยกประวัติส่วนตัว
    # หาตำแหน่งรูปโปรไฟล์
    student_info_img = transcript_img[:y, :]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(student_info_img, connectivity=8) # cca
    areas = [stat[4] for stat in stats]  # ดึงค่า area
    sorted_areas = sorted(areas, reverse=True)  # เรียงลำดับจากมากไปน้อย
    second_max_area = sorted_areas[1]  # ค่าอันดับ 2
    second_max_area_index = areas.index(second_max_area)  # หาตำแหน่งในลิสต์เดิม

    profile_position = stats[second_max_area_index]
    x, y, w, h, area = profile_position
    profile_img = student_info_img[y:y+h, x:x+w]

    # ข้อมูลนักศึกที่เอารูปโปรไฟล์ออก
    studentInfo_no_profile_img = student_info_img[:, :x-10]

    return table_img, studentInfo_no_profile_img

def detect_text_group(binary_image):
    kernel = np.ones((6, 80), np.uint8)
    lines_text = cv2.dilate(binary_image, kernel, iterations=1)

    text_group_images = []

    # ใช้ Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lines_text, connectivity=8)
    char_stats = stats[1:]  # ข้าม Background (index 0)

    # จัดเรียงจากซายไปขวา (ตามค่า x)
    sorted_indices = np.lexsort((char_stats[:, 0], char_stats[:, 1]))  # (x, y)
    sorted_stats = char_stats[sorted_indices]

    expand_ratio = 0.0  # อัตราส่วนการขยาย (0.5 คือ 50% ของขนาดเดิม)
    reduce_size = 0.00

    for idx, stats in enumerate(sorted_stats):  # เริ่มจาก 1 เพราะ 0 คือ background
        x, y, w, h, area = stats

        x_exp = int(x - (expand_ratio-reduce_size) * w)
        y_exp = int(y - (expand_ratio) * h)
        w_exp = int(w + 2 * (expand_ratio-reduce_size) * w)
        h_exp = int(h + 2 * (expand_ratio) * h)

        # ตรวจสอบไม่ให้เกินขอบภาพ
        x_exp = max(0, x_exp)
        y_exp = max(0, y_exp)
        w_exp = min(binary_image.shape[1] - x_exp, w_exp)
        h_exp = min(binary_image.shape[0] - y_exp, h_exp)

        if w >= 90 and h >= 20:  # ปรับค่าขนาดขั้นต่ำและสูงสุดตามต้องการ
            cca_img = binary_image[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
            text_group_images.append(cca_img)

    return text_group_images

def table_cell_detection(table_img):
    # ใช้ Morphological Operations เพื่อตรวจจับเส้น
    #แนวนอน (Rows)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(table_img, cv2.MORPH_OPEN, horizontal_kernel)

    #แนวตั้ง (Columns)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(table_img, cv2.MORPH_OPEN, vertical_kernel)

    # รวมเส้น
    table_structure = cv2.add(horizontal_lines, vertical_lines)

    # Combine lines
    combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)

    # Dilate to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_lines = cv2.dilate(combined_lines, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# ฟังก์ชันสำหรับดึงค่า y (แนวตั้ง) สำหรับการจัดเรียงตามแถว
def get_y_position(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return y

# ฟังก์ชันสำหรับดึงค่า x (แนวนอน) สำหรับการจัดเรียงตามคอลัมน์
def get_x_position(cell):
    x, y, w, h = cell
    return x

def detect_text_group_in_cell(dilalated_image, binary_image, mode, calculate_line_stats=None):
    rgb_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

    text_group_images = []
    
    if(mode == 1):
        # ใช้ Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilalated_image, connectivity=8)

        # กรองข้อมูล Background และจัดเรียงจากซ้ายไปขวา (ตามค่า x)
        char_stats = stats[1:]  # ข้าม Background (index 0)
        sorted_indices = np.argsort(char_stats[:, 1])  # จัดเรียงตามค่า y (คอลัมน์ที่ 1)
        sorted_stats = char_stats[sorted_indices]

        # ใช้ Boolean Indexing เพื่อเอา noise ออก 
        sorted_stats = sorted_stats[sorted_stats[:, 4] >= 100]

        calculate_line_stats = []
        
        #หาช่องระยะห่างระหว่างบรรทัดเพื่อนนำไปใช้กับ col ชื่อวิชา
        for idx_stat, stat in enumerate(sorted_stats):
            if idx_stat == (len(sorted_stats)-1):
                print("เข้าเงื่อน",idx_stat)
                x, y, w, h, area = stat
                new_y = round(y-(h/2))
                new_h = round(h+h)
                calculate_line_stats.append([x, new_y, w, new_h, area])
            else:
                #print("index: ",idx_stat)
                current_stat = stat
                next_stat = sorted_stats[idx_stat+1]

                distance = next_stat[1] - current_stat[1]
                line_spacing = distance/current_stat[3]

                if line_spacing > 3 and line_spacing < 7: # เป็นชื่อวิชาที่มีความยาวมากกว่า 1 บรรทัด
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+(h*2.5))
                    calculate_line_stats.append([x, new_y, w, new_h, area])
                elif line_spacing > 7:
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+h)
                    calculate_line_stats.append([x, new_y, w, new_h, area])
                else:
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+h)
                    calculate_line_stats.append([x, new_y, w, new_h, area])

        calculate_line_stats = np.array(calculate_line_stats)
        #print(idx_stat, current_stat)
        #print(distance, line_spacing)

    expand_ratio = 0.0  # อัตราส่วนการขยาย (0.5 คือ 50% ของขนาดเดิม)
    reduce_size = 0.00
    text_stats = sorted_stats if mode == 1 else calculate_line_stats
    
    for idx, stats in enumerate(text_stats):  # เริ่มจาก 1 เพราะ 0 คือ background
        x, y, w, h, area = stats
        if area >= 100:  # ปรับค่าขนาดขั้นต่ำและสูงสุดตามต้องการ
            if mode == 1:
                cca_img = binary_image[y:y+h, x:x+w]
            else:
                cca_img = binary_image[y:y+h, :]
            text_group_images.append(cca_img)

            image_height, image_width, _ = rgb_image.shape  # ได้ค่า (สูง, กว้าง, ช่องสี)
            cv2.rectangle(rgb_image, (x, y), (image_width, y + h), (0, 255, 0), 1)
    
    if mode == 1:
        return text_group_images, calculate_line_stats
    else:
        return text_group_images

def detect_text_in_group(binary_image):
    kernel = np.ones((6, 6), np.uint8)
    dummy_image = cv2.dilate(binary_image, kernel, iterations=3)
    sub_text_images = []

    # เช็คว่าภาพเป็นสีดำทั้งหมดหรือไม่
    if not np.any(binary_image):  # ถ้าค่าพิกเซลทั้งหมดเป็น 0 (ดำสนิท)
        print("ภาพเป็นสีดำทั้งหมด")
        sub_text_images.append(binary_image)
        return sub_text_images 
    else:
        # ใช้ Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dummy_image, connectivity=8)

        # กรองข้อมูล Background และจัดเรียงจากซ้ายไปขวา (ตามค่า x)
        char_stats = stats[1:]  # ข้าม Background (index 0)
        sorted_indices = np.argsort(char_stats[:, 0])  # จัดเรียงตามค่า x (คอลัมน์ที่ 0)
        sorted_stats = char_stats[sorted_indices]

        expand_ratio = 0.0  # อัตราส่วนการขยาย (0.5 คือ 50% ของขนาดเดิม)
        reduce_size = 0.00
        #for idx, i in enumerate(range(1, num_labels)):  # เริ่มจาก 1 เพราะ 0 คือ background
        for idx, stats in enumerate(sorted_stats):
            #x, y, w, h, area = stats[i]
            x, y, w, h, area = stats
            x_exp = int(x - (expand_ratio-reduce_size) * w)
            y_exp = int(y - (expand_ratio) * h)
            w_exp = int(w + 2 * (expand_ratio-reduce_size) * w)
            h_exp = int(h + 2 * (expand_ratio) * h)

            # ตรวจสอบไม่ให้เกินขอบภาพ
            x_exp = max(0, x_exp)
            y_exp = max(0, y_exp)
            w_exp = min(binary_image.shape[1] - x_exp, w_exp)
            h_exp = min(binary_image.shape[0] - y_exp, h_exp)

            cca_img = binary_image[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
            sub_text_images.append(cca_img)
        return sub_text_images 

def predict_text_in_cell(text_group_sub_images):
    custom_config = r'--oem 3 --psm 7'
    text_box = []
    for idx_group, text_group in enumerate(text_group_sub_images):
        for idx_sub, sub_text in enumerate(text_group):
            if not np.any(sub_text):
                text = "-"
            else:
                text = pytesseract.image_to_string(sub_text, config=custom_config, lang='tha')
            text_cleaned = text.replace("\n", "")  # ลบ \n ออก
            text_box.append(text_cleaned)
    return text_box



