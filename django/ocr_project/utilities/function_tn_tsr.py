import pytesseract
import cv2
from matplotlib import table
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_TSR_DJ")
output_folder.mkdir(exist_ok=True)


def split_grade_table_and_students(binary_img, denoised, dummy):
    
    # แยกตาราง
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dummy, connectivity=8)
    areas = [stat[4] for stat in stats]  # ดึงค่า area
    sorted_areas = sorted(areas, reverse=True)  # เรียงลำดับจากมากไปน้อย
    second_max_area = sorted_areas[1]  # ค่าอันดับ 2
    second_max_area_index = areas.index(second_max_area)  # หาตำแหน่งในลิสต์เดิม
    table_position = stats[second_max_area_index]
    x, y, w, h, area = table_position

    table_img = binary_img[y:y+h, x:x+w]
    table_dummy_img = dummy[y:y+h, x:x+w]
    table_original_img = denoised[y:y+h, x:x+w]

    # ข้อมูลนักเรียน
    x_end = int((x+w) * 0.85) # ความกว้าง 85% ของตาราง
    x_split_half = int((x+w) * 0.38) # ความกว้าง 38% ของตาราง

    student_info_img = binary_img[:y, :x_end]
    student_info_fh_img = binary_img[:y, :x_split_half] # ครึ่งแรก
    student_info_sh_img = binary_img[:y, x_split_half:x_end] # ครึ่งหลัง

    return table_img, table_dummy_img, table_original_img, student_info_img, student_info_fh_img, student_info_sh_img

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        #print(area)
        if area > 1000:
            #print("มา")
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest

def persective_transformation(table_binary_img, table_original_img, table_dummy_img):

    # ค้นหาคอนทัวร์
    contours, hierarchy = cv2.findContours(table_dummy_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(table_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # ค้นหาสี่เหลี่ยมที่ใหญ่ที่สุด
    biggest = biggest_contour(contours)

    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_points

    # Euclidean Distance Formula
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    rigth_height = np.sqrt(((top_left[0] - bottom_right[0]) ** 2) + ((top_left[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    # Output image size
    #max_width = max(int(bottom_width), int(top_width))
    expand_width = round(max(int(bottom_width), int(top_width)) * 0.4)
    max_width = max(int(bottom_width), int(top_width)) + expand_width
    max_height = max(int(rigth_height), int(left_height))

    # Desird points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformaxtion
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_out = cv2.warpPerspective(table_binary_img.copy(), matrix, (max_width, max_height))
    img_original_out = cv2.warpPerspective(table_original_img.copy(), matrix, (max_width, max_height))
    img_dummy_out = cv2.warpPerspective(table_dummy_img.copy(), matrix, (max_width, max_height))

    #return img_out, img_original_out
    return img_out, img_original_out, img_dummy_out

# ฟังก์ชันสำหรับรวมกลุ่ม indices ที่ติดกันเข้าด้วยกัน
def group_indices(indices, gap=1):
    groups = []
    if not indices:
        return groups
    current = [indices[0]]
    for idx in indices[1:]:
        if idx - current[-1] <= gap:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return groups

def find_table_columns_rows(table_dummy_persective_img, table_persective_img):

    # คำนวณ horizontal projection (ผลรวมของ pixel ในแต่ละแถว)
    horizontal_proj = np.sum(table_dummy_persective_img, axis=1)
    vertical_proj = np.sum(table_dummy_persective_img, axis=0)

    # ตั้ง threshold สำหรับเลือกเฉพาะลอลัมที่มี "เส้น"
    horizontal_line_threshold = np.max(horizontal_proj) * 0.8
    vertical_line_threshold = np.max(vertical_proj) * 0.7

    # หา indices ของแถวที่มีค่า projection มากกว่าค่า threshold
    row_line_indices = [i for i, value in enumerate(horizontal_proj) if value > horizontal_line_threshold]
    col_line_indices = [i for i, value in enumerate(vertical_proj) if value > vertical_line_threshold]

    # รวมกลุ่ม indices ที่ติดกัน
    groups_row = group_indices(row_line_indices, gap=1)
    groups_col = group_indices(col_line_indices, gap=1)

    # สำหรับแต่ละกลุ่ม หา index กลางเป็นตำแหน่งของเส้น
    # แต่แทนที่จะใช้ความหนาแน่น เราจะใช้ความยาวของแต่ละกลุ่ม (จำนวน index ในกลุ่ม)
    groups_row_with_length = [(group, len(group)) for group in groups_row]
    groups_col_with_length = [(group, len(group)) for group in groups_col]

    # เรียงลำดับกลุ่มจากความยาวมากไปหาน้อย
    groups_row_with_length.sort(key=lambda x: x[1], reverse=True)
    groups_col_with_length.sort(key=lambda x: x[1], reverse=True)

    # เลือกเฉพาะกลุ่มที่มีความยาวมากที่สุด
    top_groups_row = groups_row_with_length[:3]
    top_groups_col = groups_col_with_length[:11]

    # คำนวณตำแหน่งเส้นโดยการหาค่าเฉลี่ยของแต่ละกลุ่ม แล้วเรียงลำดับ
    row_lines = [int(np.mean(group)) for group, _ in top_groups_row]
    col_lines = [int(np.mean(group)) for group, _ in top_groups_col]
    row_lines.sort()
    col_lines.sort()

    print("ตำแหน่งของเส้นแถวที่ตรวจจับได้:", row_lines)
    print("ตำแหน่งของเส้นคอลัมน์ที่ตรวจจับได้:", col_lines)

    mask_row = np.zeros_like(table_dummy_persective_img)
    mask_col = np.zeros_like(table_dummy_persective_img)

    # วาดเส้นแถวลงใน mask (เส้นแนวนอน)
    for y in row_lines:
        cv2.line(mask_row, (0, y), (mask_row.shape[1]-1, y), 255, thickness=10)
    cv2.imwrite(f"{output_folder}/table/row_lines_mask.png", mask_row)

    # วาดเส้นคอลัมน์ลงใน mask (เส้นแนวตั้ง)
    for x in col_lines:
        cv2.line(mask_col, (x, 0), (x, mask_col.shape[0]-1), 255, thickness=10)
    cv2.imwrite(f"{output_folder}/table/col_lines_mask.png", mask_col)

    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นแถวออกจากภาพ ---
    mask_row_inv = cv2.bitwise_not(mask_row)
    img_no_lines_row = cv2.bitwise_and(table_persective_img, table_persective_img, mask=mask_row_inv)
    cv2.imwrite(f"{output_folder}/table/table_no_lines_bitwise_row.png", img_no_lines_row)

    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นแนวตั้งออกจากภาพ ---
    mask_col_inv = cv2.bitwise_not(mask_col)
    img_no_lines_col = cv2.bitwise_and(table_persective_img, table_persective_img, mask=mask_col_inv)
    cv2.imwrite(f"{output_folder}/table/table_no_lines_bitwise_col.png", img_no_lines_col)

    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นแนวตั้งและแนวนอนออกจากภาพ ---
    combined_mask = cv2.bitwise_or(mask_row, mask_col)
    combined_mask_inv = cv2.bitwise_not(combined_mask)
    img_no_lines_row_col = cv2.bitwise_and(table_persective_img, table_persective_img, mask=combined_mask_inv)

    cv2.imwrite(f"{output_folder}/table/img_no_lines_row_col.png", img_no_lines_row_col)

    cropped_col_segments = []

    for i in range(len(col_lines) - 1):
        x_start = col_lines[i]
        x_end = col_lines[i+1]
        y_start = row_lines[1]
        y_end = row_lines[-1]
        
        cropped = img_no_lines_row_col[y_start:y_end, x_start:x_end]  # crop ทุกคอลัมน์ในช่วงแถวที่กำหนด
        cropped_col_segments.append(cropped)
        cv2.imwrite(f"{output_folder}/table/cropped_segment_{i+1}.png", cropped)

    return cropped_col_segments

def detect_text_group_in_cell(cell_img, mode=0, calculate_line_stats=None):
    text_group_images = []

    kernel_open = np.ones((4, 4), np.uint8)
    #kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    remove_noise = cv2.morphologyEx(cell_img, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel = np.ones((3, 13), np.uint8)
    group_text_img = cv2.dilate(remove_noise, kernel, iterations=2)
    rgb_image = cv2.cvtColor(cell_img.copy(), cv2.COLOR_GRAY2RGB)


    if(mode == 1):
        # ใช้ Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(group_text_img, connectivity=8)

        text_stats = stats[1:]
        sorted_indices = np.argsort(text_stats[:, 1])  # จัดเรียงตามค่า y (คอลัมน์ที่ 1)
        sorted_stats = text_stats[sorted_indices]

        # ใช้ Boolean Indexing เพื่อเอา noise ออก 
        sorted_stats = sorted_stats[sorted_stats[:, 4] >= 2000]
        calculate_line_stats = []

        for idx_stat, stat in enumerate(sorted_stats):
            x, y, w, h, area = stat
            #print(f"CCA #{idx_stat}: bounding box = (x={x}, y={y}, w={w}, h={h}, area={area})")
            if idx_stat == (len(sorted_stats)-1):
                #print("เข้าเงื่อนลำดัยสุดท้าย",idx_stat)
                x, y, w, h, area = stat
                new_y = round(y-(h/2))
                new_h = round(h+(h*0.8))
                calculate_line_stats.append([x, new_y, w, new_h, area])
            else:
                current_stat = stat
                next_stat = sorted_stats[idx_stat+1]

                distance = next_stat[1] - current_stat[1]
                line_spacing = distance/current_stat[3]

                if line_spacing > 3 and line_spacing <= 5: # เป็นชื่อวิชาที่มีความยาวมากกว่า 1 บรรทัด
                    #print("เข้าเงื่อนไข มากกว่า 1 บรรทัด")
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+(h*2.5))
                    calculate_line_stats.append([x, new_y, w, new_h, area])

                elif line_spacing > 6: # เป็นช่องว่างที่ไม่มีวิชา
                    #print("เข้าเงื่อนไข เป็นช่องว่างที่ไม่มีวิชา")
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+h)
                    calculate_line_stats.append([x, new_y, w, new_h, area])
                
                else: # เป็นชื่อวิชาที่มีความยาวแค่ว่า 1 บรรทัด
                    #print("เข้าเงื่อนไข 1 บรรทัด")
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+(h*0.8))
                    calculate_line_stats.append([x, new_y, w, new_h, area])

        calculate_line_stats = np.array(calculate_line_stats) 

    text_stats = sorted_stats if mode == 1 else calculate_line_stats

    
    for idx, stats in enumerate(text_stats): # เก็บภาพกลุม
        x, y, w, h, area = stats
   
        if mode == 1:
            cca_img = cell_img[y:y+h, x:x+w]
        if mode == 2:
            if idx == 0: # ดักบัค crop รูปเกินขอบเขต
                cca_img = cell_img[y+5:y+h, :]
            elif idx == (len(text_stats)-1):
                cca_img = cell_img[y:y+h-5, :]
            else:
                cca_img = cell_img[y:y+h, :]
        text_group_images.append(cca_img)

        # หาขนาดของภาพ (ความกว้างและความสูง)
        image_height, image_width, _ = rgb_image.shape  # ได้ค่า (สูง, กว้าง, ช่องสี)
        cv2.rectangle(rgb_image, (x, y), (image_width, y + h), (0, 255, 0), 1)

    if mode == 1:
        return text_group_images, calculate_line_stats, rgb_image
    else:
        return text_group_images, rgb_image

def detect_sub_text_in_group(binary_image):

    debug = False
    text_group = []
    for idx, img in enumerate(binary_image):

        sub_text_images = []

        kernel_open = np.ones((3, 3), np.uint8)
        remove_noise = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # เช็คว่าภาพเป็นสีดำทั้งหมดหรือไม่
        if not np.any(remove_noise):  # ถ้าค่าพิกเซลทั้งหมดเป็น 0 (ดำสนิท)
            print("ภาพเป็นสีดำทั้งหมด")
            sub_text_images.append(remove_noise)
            #return sub_text_images 
        
        else:
            kernel = np.ones((6, 6), np.uint8)
            dummy_image = cv2.dilate(remove_noise, kernel, iterations=2)

            # ใช้ Connected Component Analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dummy_image, connectivity=8)
            char_stats = stats[1:] # ข้าม Background (index 0)
            sorted_indices = np.argsort(char_stats[:, 0]) # จัดเรียงตามค่า x (คอลัมน์ที่ 0)
            sorted_stats = char_stats[sorted_indices]

            # ใช้ Boolean Indexing เพื่อเอา noise ออก 
            sorted_stats = sorted_stats[sorted_stats[:, 4] >= 200]

            for idx, stats in enumerate(sorted_stats):
                #x, y, w, h, area = stats[i]
                x, y, w, h, area = stats

                crop_img = img[y:y+h, x:x+w]
                sub_text_images.append(crop_img)

        text_group.append(sub_text_images)

    return text_group

def predict_text(text_group, mode=0):
    
    if mode == 1:
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-'
    elif mode == 3:
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    else:
        custom_config = r'--oem 3 --psm 7'

    text_box = []
    for idx_g, text_g in enumerate(text_group):
        sub_text_group = ""
        for idx_sub, sub_text in enumerate(text_g):
            if not np.any(sub_text):
                print("เข้าเงื่อนไข")
                text = "-"
            else:
                text = pytesseract.image_to_string(sub_text, config=custom_config, lang='tha+eng')
 
            sub_text_group += text + " "
            text_cleaned = sub_text_group.replace("\n", "")  # ลบ \n ออก
        text_box.append(text_cleaned)

    return text_box

def crop_border(image, left_percent=0, right_percent=0, top_percent=0, bottom_percent=0):
    
    # หาความกว้างและความสูงของภาพ
    height, width = image.shape
    print(height)
    print(width)

    # คำนวณพิกัดที่จะตัด (แปลงเป็นพิกเซล)
    x_start = int(width * left_percent)
    x_end = int(width * (1 - right_percent))
    y_start = int(height * top_percent)
    y_end = int(height * (1 - bottom_percent))

    # ตัดภาพ (Crop)
    cropped_img = image[y_start:y_end, x_start:x_end]

    #cv2.imwrite(f"{output_folder}/cropped_fh.jpg", cropped_img)
    
    return cropped_img

def find_text_student_info_fh(student_info_fh_img):
    output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_TSR_DJ")
    output_folder.mkdir(exist_ok=True)

    student_info_fh_img = crop_border(student_info_fh_img.copy(), 0.06, 0.06, 0.06, 0.01)

    rgb_image = cv2.cvtColor(student_info_fh_img.copy(), cv2.COLOR_GRAY2RGB)
    
    # กำหนด kernel (ขนาดของ kernel สามารถปรับเปลี่ยนได้ตามความเหมาะสม)
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((6, 50), np.uint8)
    
    opening = cv2.morphologyEx(student_info_fh_img.copy(), cv2.MORPH_OPEN, kernel=kernel_open, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=kernel_close, iterations=2)

    rgb_closing_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(f"{output_folder}/opening.jpg", opening)
    cv2.imwrite(f"{output_folder}/closing.jpg", closing)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)

    # 1. เอา stats ตัวแรก (background) ออก
    stats_no_bg = stats[1:]

    # 2. เรียง stats ใหม่โดยใช้ค่า area (คอลัมน์ที่ 4) จากมากไปน้อย
    sorted_indices = np.argsort(-stats_no_bg[:, 4])
    sorted_stats = stats_no_bg[sorted_indices]

    # 3. เลือกแค่ 13 element ที่มีค่า area สูงสุด
    top_13_stats = sorted_stats[:13]
    top_13_stats_sorted_by_y = top_13_stats[np.argsort(top_13_stats[:, 1])]

    img_height, img_width = student_info_fh_img.shape[:2]

    # กำหนด margin เป็นเปอร์เซ็นต์ของขนาด bounding box
    # เช่น กำหนด 10% ของความกว้าง/ความสูงของ bounding box สำหรับแต่ละด้าน
    left_margin_percent = 0.1    # ขยายซ้าย 10%
    right_margin_percent = 0.1   # ขยายขวา 10%
    top_margin_percent = 0.2     # ขยายบน 20%
    bottom_margin_percent = 0.1  # ขยายล่าง 10%

    text_group_stud_fh = []
    for idx, stats in enumerate(top_13_stats_sorted_by_y): # เก็บภาพกลุม
        x, y, w, h, area = stats

        # คำนวณ margin ตามเปอร์เซ็นต์ของ bounding box
        left_margin = int(w * left_margin_percent)
        right_margin = int(w * right_margin_percent)
        top_margin = int(h * top_margin_percent)
        bottom_margin = int(h * bottom_margin_percent)

        # คำนวณพิกัดใหม่โดยใช้ margin ที่คำนวณได้
        x_new = max(x - left_margin, 0)
        y_new = max(y - top_margin, 0)
        x_end = min(x + w + right_margin, img_width)
        y_end = min(y + h + bottom_margin, img_height)

        cluster_img = student_info_fh_img[y_new:y_end, x_new:x_end]
        text_group_stud_fh.append(cluster_img)

        # วาดกรอบที่ขยายแล้วลงบนภาพ
        cv2.rectangle(rgb_image, (x_new, y_new), (x_end, y_end), (0, 255, 0), 1)
        cv2.rectangle(rgb_closing_image, (x_new, y_new), (x_end, y_end), (0, 255, 0), 1)
        
    cv2.imwrite(f"{output_folder}/cca_top_13_stats.jpg", rgb_image)
    cv2.imwrite(f"{output_folder}/cca_rgb_closing_image.jpg", rgb_closing_image)

    return text_group_stud_fh[1:]

def find_text_student_info_sh(student_info_sh_img):
    student_info_sh_img = crop_border(student_info_sh_img.copy(), 0.05, 0.00, 0.05, 0.01)

    rgb_image = cv2.cvtColor(student_info_sh_img.copy(), cv2.COLOR_GRAY2RGB)
    
    # กำหนด kernel (ขนาดของ kernel สามารถปรับเปลี่ยนได้ตามความเหมาะสม)
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((8, 50), np.uint8)
    
    opening = cv2.morphologyEx(student_info_sh_img.copy(), cv2.MORPH_OPEN, kernel=kernel_open, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=kernel_close, iterations=2)

    rgb_closing_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(f"{output_folder}/opening_sh.jpg", opening)
    cv2.imwrite(f"{output_folder}/closing_sh.jpg", closing)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)

    # 1. เอา stats ตัวแรก (background) ออก
    stats_no_bg = stats[1:]

    # 2. เรียง stats ใหม่โดยใช้ค่า area (คอลัมน์ที่ 4) จากมากไปน้อย
    sorted_indices = np.argsort(-stats_no_bg[:, 4])
    sorted_stats = stats_no_bg[sorted_indices]

    # 3. เลือกแค่ 14 element ที่มีค่า area สูงสุด
    top_14_stats = sorted_stats[:14]
    top_14_stats_sorted_by_y = top_14_stats[np.argsort(top_14_stats[:, 1])]

    img_height, img_width = student_info_sh_img.shape[:2]

    # กำหนด margin เป็นเปอร์เซ็นต์ของขนาด bounding box
    # เช่น กำหนด 10% ของความกว้าง/ความสูงของ bounding box สำหรับแต่ละด้าน
    left_margin_percent = 0.1    # ขยายซ้าย 10%
    right_margin_percent = 0.1   # ขยายขวา 10%
    top_margin_percent = 0.2     # ขยายบน 20%
    bottom_margin_percent = 0.1  # ขยายล่าง 10%

    text_group_stud_sh = []
    for idx, stats in enumerate(top_14_stats_sorted_by_y): # เก็บภาพกลุม
        x, y, w, h, area = stats

        # คำนวณ margin ตามเปอร์เซ็นต์ของ bounding box
        left_margin = int(w * left_margin_percent)
        right_margin = int(w * right_margin_percent)
        top_margin = int(h * top_margin_percent)
        bottom_margin = int(h * bottom_margin_percent)

        # คำนวณพิกัดใหม่โดยใช้ margin ที่คำนวณได้
        x_new = max(x - left_margin, 0)
        y_new = max(y - top_margin, 0)
        x_end = min(x + w + right_margin, img_width)
        y_end = min(y + h + bottom_margin, img_height)

        cluster_img = student_info_sh_img[y_new:y_end, x_new:x_end]
        text_group_stud_sh.append(cluster_img)

        # วาดกรอบที่ขยายแล้วลงบนภาพ
        cv2.rectangle(rgb_image, (x_new, y_new), (x_end, y_end), (0, 255, 0), 1)
        cv2.rectangle(rgb_closing_image, (x_new, y_new), (x_end, y_end), (0, 255, 0), 1)
        
    cv2.imwrite(f"{output_folder}/cca_top_14_stats.jpg", rgb_image)
    cv2.imwrite(f"{output_folder}/cca_rgb_closing_image.jpg", rgb_closing_image)

    return text_group_stud_sh[3:]

def detect_sub_text_in_group_stud(binary_image):
    
    text_group = []

    kernel = np.ones((3, 3), np.uint8)
    dummy_image = cv2.dilate(binary_image, kernel, iterations=2)

    # ใช้ Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dummy_image, connectivity=8)
    word_stats = stats[1:] # ข้าม Background (index 0)
    sorted_indices = np.argsort(word_stats[:, 0]) # จัดเรียงตามค่า x (คอลัมน์ที่ 0)
    sorted_stats = word_stats[sorted_indices]

    for idx, stats in enumerate(sorted_stats):
        x, y, w, h, area = stats
        cluster_img = binary_image[y:y+h, x:x+w]
        text_group.append(cluster_img)

    return text_group

def predict_text_stud(text_group, mode=0):
    
    custom_config = r'--oem 3 --psm 7'

    sub_text_group = ""
    for idx_sub, sub_text in enumerate(text_group):
        if not np.any(sub_text):
            print("เข้าเงื่อนไข")
            text = "-"
        else:
            text = pytesseract.image_to_string(sub_text, config=custom_config, lang='tha')
   
        sub_text_group += text + " "
        text_cleaned = sub_text_group.replace("\n", "")  # ลบ \n ออก

    return text_cleaned