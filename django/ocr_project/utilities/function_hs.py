import pytesseract
import cv2
from matplotlib import table
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_HS_DJ/front")
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
    x_end = int((x+w) * 0.76) # ความกว้าง 76% ของตาราง
    x_split_half = int((x+w) * 0.53) # ความกว้าง 53% ของตาราง

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

    return img_out, img_original_out, img_dummy_out

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

def find_table_columns(table_dummy_persective_img, table_persective_img):

    # คำนวณ horizontal projection (ผลรวมของ pixel ในแต่ละแถว)
    vertical_proj = np.sum(table_dummy_persective_img, axis=0)

    # ตั้ง threshold สำหรับเลือกเฉพาะลอลัมที่มี "เส้น"
    vertical_line_threshold = np.max(vertical_proj) * 0.7

    # หา indices ของแถวที่มีค่า projection มากกว่าค่า threshold
    col_line_indices = [i for i, value in enumerate(vertical_proj) if value > vertical_line_threshold]

    # รวมกลุ่ม indices ที่ติดกัน
    groups_col = group_indices(col_line_indices, gap=1)

    # สำหรับแต่ละกลุ่ม หา index กลางเป็นตำแหน่งของเส้น
    # แต่แทนที่จะใช้ความหนาแน่น เราจะใช้ความยาวของแต่ละกลุ่ม (จำนวน index ในกลุ่ม)
    groups_col_with_length = [(group, len(group)) for group in groups_col]

    # เรียงลำดับกลุ่มจากความยาวมากไปหาน้อย
    groups_col_with_length.sort(key=lambda x: x[1], reverse=True)

    # เลือกเฉพาะกลุ่มที่มีความยาวมากที่สุด
    top_groups_col = groups_col_with_length[:10]

    # คำนวณตำแหน่งเส้นโดยการหาค่าเฉลี่ยของแต่ละกลุ่ม แล้วเรียงลำดับ
    col_lines = [int(np.mean(group)) for group, _ in top_groups_col]
    col_lines.sort()

    print("ตำแหน่งของเส้นคอลัมน์ที่ตรวจจับได้:", col_lines)
    mask_col = np.zeros_like(table_dummy_persective_img)

    # วาดเส้นคอลัมน์ลงใน mask (เส้นแนวตั้ง)
    for x in col_lines:
        cv2.line(mask_col, (x, 0), (x, mask_col.shape[0]-1), 255, thickness=10)
    cv2.imwrite(f"{output_folder}/cols/col_lines_mask.png", mask_col)

    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นแถวออกจากภาพ ---
    mask_col_inv = cv2.bitwise_not(mask_col)
    img_no_lines_col = cv2.bitwise_and(table_persective_img, table_persective_img, mask=mask_col_inv)
    cv2.imwrite(f"{output_folder}/cols/table_no_lines_bitwise_col.png", img_no_lines_col)

    cropped_col_segments = []

    for i in range(len(col_lines) - 1):
        x_start = col_lines[i]
        x_end = col_lines[i+1]
        cropped = img_no_lines_col[:, x_start:x_end]  # crop ทุกคอลัมน์ในช่วงแถวที่กำหนด
        cropped_col_segments.append(cropped)
        cv2.imwrite(f"{output_folder}/cols/cropped_segment_{i+1}.png", cropped)

    return cropped_col_segments

def crop_top_bottom(image, top_percent=6.5, bottom_percent=20):
  
    # ดึงขนาดของภาพ (สูง, กว้าง)
    h, w = image.shape[:2]
    
    # คำนวณตำแหน่งที่ต้องตัด
    top_crop = int(h * (top_percent / 100))
    bottom_crop = int(h * (bottom_percent / 100))
    
    # ตำแหน่งเริ่มต้นและสิ้นสุดหลังจากตัด
    new_top = top_crop
    new_bottom = h - bottom_crop
    
    # ตรวจสอบว่า new_bottom > new_top
    if new_bottom <= new_top:
        raise ValueError("เปอร์เซ็นต์การตัดสูงเกินไปสำหรับความสูงของภาพ")
    
    # Crop ภาพ: ตัดเฉพาะแนวแกน y แล้วใช้ทุกคอลัมน์
    cropped_image = image[new_top:new_bottom, :]
    
    return cropped_image

def detect_text_in_cell(cell_img, mode=0, calculate_line_stats=None):
    text_group_images = []

    kernel_open = np.ones((4, 4), np.uint8)
    #kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    remove_noise = cv2.morphologyEx(cell_img, cv2.MORPH_OPEN, kernel_open, iterations=1)

    cv2.imwrite(f"{output_folder}/cell_images/remove_noise.jpg", remove_noise)

    kernel = np.ones((3, 13), np.uint8)
    group_text_img = cv2.dilate(remove_noise, kernel, iterations=2)
    rgb_image = cv2.cvtColor(cell_img.copy(), cv2.COLOR_GRAY2RGB)

    #plt.figure(figsize=(15, 15))
    #plt.imshow(group_text_img, cmap="gray")

    if(mode == 1):
        # ใช้ Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(group_text_img, connectivity=8)
        text_stats = stats[1:]
        sorted_indices = np.argsort(text_stats[:, 1])  # จัดเรียงตามค่า y (คอลัมน์ที่ 1)
        sorted_stats = text_stats[sorted_indices]
        # ใช้ Boolean Indexing เพื่อเอา noise ออก 
        sorted_stats = sorted_stats[sorted_stats[:, 4] >= 400]
        calculate_line_stats = []

        for idx_stat, stat in enumerate(sorted_stats):
            x, y, w, h, area = stat
            #cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
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

                if line_spacing > 4: # เป็นชื่อวิชาที่มีความยาวมากกว่า 1 บรรทัด
                    #print("เข้าเงื่อนไข มากกว่า 1 บรรทัด")
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+(h*2.5))
                    calculate_line_stats.append([x, new_y, w, new_h, area])

                elif line_spacing >= 3: # เป็นช่องว่างที่ไม่มีวิชา
                    #print("เข้าเงื่อนไข เป็นช่องว่างที่ไม่มีวิชา")
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2))
                    new_h = round(h+(h*0.8)) # round(h+h)
                    calculate_line_stats.append([x, new_y, w, new_h, area])
                
                else: # เป็นชื่อวิชาที่มีความยาวแค่ว่า 1 บรรทัด
                    #print("เข้าเงื่อนไข 1 บรรทัด")
                    x, y, w, h, area = current_stat
                    new_y = round(y-(h/2.2)) # round(y-(h/2))
                    new_h = round(h+(h*0.8)) # round(h+(h*0.8))
                    calculate_line_stats.append([x, new_y, w, new_h, area])

        calculate_line_stats = np.array(calculate_line_stats) 

            #print(f"CCA #{idx_stat}: bounding box = (x={x}, y={y}, w={w}, h={h}, area={area})")
            #cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #cv2.imwrite(f"{output_folder}/cell_images/cca.jpg", rgb_image)

    text_stats = sorted_stats if mode == 1 else calculate_line_stats
    
    for idx, stats in enumerate(text_stats): # เก็บภาพกลุม
        x, y, w, h, area = stats
   
        if mode == 1:
            cca_img = cell_img[y:y+h, x:x+w]
        if mode == 2:
            #print(f"stats #{idx}: bounding box = (x={x}, y={y}, w={w}, h={h}, area={area})")
            if idx == 0: # ดักบัค crop รูปเกินขอบเขต
                cca_img = cell_img[y+5:y+h, :]
            elif idx == (len(text_stats)-1):
                cca_img = cell_img[y:y+h, :]
            else:
                cca_img = cell_img[y:y+h, :]
        text_group_images.append(cca_img)

        # หาขนาดของภาพ (ความกว้างและความสูง)
        image_height, image_width, _ = rgb_image.shape  # ได้ค่า (สูง, กว้าง, ช่องสี)
        cv2.rectangle(rgb_image, (0, y), (image_width, y + h), (0, 255, 0), 1)
        #cv2.rectangle(rgb_image, (x, y), (image_width, y + h), (0, 255, 0), 1)

    if mode == 1:
        return text_group_images, calculate_line_stats, rgb_image
    else:
        return text_group_images, rgb_image
    
def detect_sub_text_in_group(binary_images):

    text_group = []
    for idx, img in enumerate(binary_images):
        #print(idx+1)

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
                x, y, w, h, area = stats
                cca_img = img[y:y+h, x:x+w]
                sub_text_images.append(cca_img)

        text_group.append(sub_text_images)

    return text_group

def separate_subject_code_and_name(text_group_subject_code_name):
    subject_code = []
    subject_name = []
    for idx_g, text_group in enumerate(text_group_subject_code_name):
        subject_code.append(text_group[:1])
        subject_name.append(text_group[1:])

    return subject_code, subject_name

# เพิ่มมา
def detect_one_level_of_char(text_group):
    debug = False
    text_group_char = []
    for idx_g, text_g in enumerate(text_group):

        sub_text_char = []
        for idx_s, sub_text in enumerate(text_g):

            if not np.any(sub_text): # เช็คว่าเป็นภาพว่างรึเปล่า
                print("ภาพว่างเปล่า")
                sub_text_char.append([sub_text])
                continue

            #skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            skeleton_guohall = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_GUOHALL)

            #kernel_open = np.ones((2, 2), np.uint8)
            kernel_dummy = np.ones((2, 2), np.uint8)
            #opening = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel=kernel_open, iterations=2)
            #closing = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel=kernel_open, iterations=2)
            dummy_image = cv2.dilate(skeleton_guohall, kernel_dummy, iterations=1)

            rgb_image = cv2.cvtColor(sub_text.copy(), cv2.COLOR_GRAY2RGB)

            contours, hierarchy = cv2.findContours(dummy_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

            char_images = []
            for idx_c, cnt in enumerate(sorted_contours):

                x, y, w, h = cv2.boundingRect(cnt)
                contour_area = cv2.contourArea(cnt)

                mask = np.zeros(sub_text.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)

                kernel_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ปรับขนาด kernel ตามต้องการ
                dilated_mask = cv2.dilate(mask, kernel_mask, iterations=1)

                # ใช้ mask กับภาพต้นฉบับ เพื่อดึงเฉพาะส่วนภายใน contour
                char_result = cv2.bitwise_and(sub_text, sub_text, mask=dilated_mask)

                contours_char, hierarchy_char = cv2.findContours(char_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours_char, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(largest_contour)
                area = int(cv2.contourArea(largest_contour))
                crop_img = char_result[y:y+h, x:x+w]
                char_images.append(crop_img)
                

            sub_text_char.append(char_images)
        text_group_char.append(sub_text_char)
    return text_group_char

# เพิ่มมา
def resize_with_min_padding(image, desired_size, min_padding):
    
    """
    ปรับขนาดภาพให้ใกล้เคียง desired_size โดยลด Padding และเพิ่มการขยายภาพต้นฉบับ
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid numpy array.")

    if not isinstance(desired_size, int) or desired_size <= 0:
        raise ValueError("desired_size must be a positive integer.")

    old_size = image.shape[:2]  # (height, width)
    max_size = max(old_size)

    # คำนวณอัตราส่วนการปรับขนาดให้ใกล้เคียง desired_size
    ratio = float(desired_size - 2 * min_padding) / max_size
    new_size = tuple([int(x * ratio) for x in old_size])  # ขนาดใหม่ (height, width)

    # Resize ภาพให้คงสัดส่วนเดิม แต่ใหญ่ขึ้น
    resized_image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    # คำนวณ Padding ใหม่
    delta_w = max(desired_size - new_size[1], 0)  # Padding ด้านความกว้าง
    delta_h = max(desired_size - new_size[0], 0)  # Padding ด้านความสูง
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # ตรวจสอบสีสำหรับ Grayscale หรือ RGB
    color = [0] if len(image.shape) == 2 else [0, 0, 0]

    # เพิ่ม Padding รอบภาพ
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_image

# เพิ่มมา
def predict_text_one_level(text_group_char, char_model, model_char_subject_code_hs, model_char_credit, model_char_academic_results_hs):
    models_one_level = {
        0: model_char_subject_code_hs,
        1: model_char_credit,
        2: model_char_academic_results_hs,
    }

    char_subject_code_hs = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'I', 'ค', 'ง', 'จ', 'ญ', 'ท', 'พ', 'ว', 'ศ', 'ส', 'อ'
    ]

    char_credit = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'
    ]

    char_academic_results_hs = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'ท', 'ผ', 'ม', 'ร', '.'
    ]

    char_labels = {
        0: char_subject_code_hs,
        1: char_credit,
        2: char_academic_results_hs,
    }


    debug = False
    # กำหนดขนาด Input ของโมเดล
    input_size = 32  # ขนาด 32x32
    text_block = []

    for idx_g, text_group in enumerate(text_group_char):
        text_result = ""

        for idx_s, sub_text in enumerate(text_group):
            sub_text_result = ""

            for idx_c, char in enumerate(sub_text):
                
                if char is None:
                    print(f"Character image {idx_c} is None.")
                    continue  # ข้ามภาพนี้
                else:
                    '''
                    if not np.any(sub_text):
                        print("ไม่มี")
                        sub_text_result = "-"
                        continue
                    '''

                    # เพิ่ม Padding และปรับขนาดภาพ
                    padded_img = resize_with_min_padding(char, input_size, min_padding=1)

                    # Normalization (เปลี่ยนค่าพิกเซลให้อยู่ในช่วง [0, 1])
                    normalized_img = padded_img / 255.0

                    if len(normalized_img.shape) == 2:  # หากภาพเป็น Grayscale (2D)
                        normalized_img = np.expand_dims(normalized_img, axis=-1)
                        processed_image = np.expand_dims(normalized_img, axis=0)  # เพิ่ม Batch Dimension

                    if char_model in models_one_level:
                        prediction = models_one_level[char_model].predict(processed_image)
                        predicted_class = np.argmax(prediction)
                        confidence_score = np.max(prediction)

                        class_char = char_labels[char_model]
                        predicted_letter = class_char[predicted_class]
                        sub_text_result += predicted_letter

            text_result += sub_text_result
            text_result += " "
        text_block.append(text_result)

    print("ประมวลผลเสร็จสิ้น")
    return text_block 

# เพิ่มมา
def get_centroid(contour):
    # คำนวณ moments ของ contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0, 0
    return (cx, cy)

# เพิ่มมา
def char_level(char_images):

    char_box = []
    
    skeleton = cv2.ximgproc.thinning(char_images, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    #skeleton = cv2.ximgproc.thinning(char_images, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    kernel_dummy = np.ones((3, 3), np.uint8)
    closing_skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel=kernel_dummy, iterations=1)
    dummy_image = cv2.dilate(skeleton, kernel_dummy, iterations=1)
    closing = cv2.morphologyEx(dummy_image, cv2.MORPH_CLOSE, kernel=kernel_dummy, iterations=1)

    contours, hierarchy = cv2.findContours(dummy_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # เรียง contours ตามค่า y ของ centroid จากน้อยไปหามาก
    sorted_contours = sorted(contours, key=lambda cnt: get_centroid(cnt)[1], reverse=True)

    # คำนวณพื้นที่ของ contour แต่ละตัว และหาพื้นที่ที่ใหญ่ที่สุด
    max_area = max(cv2.contourArea(cnt) for cnt in sorted_contours)

    # กำหนด threshold เป็น 5% ของพื้นที่ที่ใหญ่ที่สุด
    min_area_threshold = max_area * 0.05

    # กรอง contours ที่มีพื้นที่ไม่น้อยกว่า threshold
    filtered_contours = [cnt for cnt in sorted_contours if cv2.contourArea(cnt) >= min_area_threshold]

    # หาก filtered_contours มีแค่ 2 ตัว
    if len(filtered_contours) == 2:

        area0 = cv2.contourArea(filtered_contours[0])
        area1 = cv2.contourArea(filtered_contours[1])
        # คำนวณความแตกต่างเป็นอัตราส่วนของ contour ที่มีพื้นที่ใหญ่กว่า
        diff_ratio = abs(area0 - area1) / max(area0, area1)
        if diff_ratio <= 0.10: # อันเก่า 0.05
            print("เหมือนกัน")

            kernel_same = np.ones((4, 4), np.uint8)
            dilated_same = cv2.dilate(char_images, kernel_same, iterations=2)
            contours, hierarchy = cv2.findContours(dilated_same, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            level = 1
            x, y, w, h = cv2.boundingRect(largest_contour)
            contour_area = int(cv2.contourArea(largest_contour))
            crop_img = char_images[y:y+h, x:x+w]
            char_box.append([crop_img, level])

            return char_box
    elif len(filtered_contours) >= 3: # เอาไว้แก้ปัญหาพวกคำว่า "ชั้น, บัน" หรือทุกคำที่มี "  ั "
        print("len(filtered_contours) >= 3")
        # คำนวณ centroid และพื้นที่ของแต่ละ contour ใน filtered_contours
        centroids_filtered = [get_centroid(cnt) for cnt in filtered_contours]
        areas_filtered = [cv2.contourArea(cnt) for cnt in filtered_contours]
    
        # หาตำแหน่งของ contour ที่มีพื้นที่มากที่สุด
        max_area_idx = areas_filtered.index(max(areas_filtered))
        max_centroid_y = centroids_filtered[max_area_idx][1]
        max_area = areas_filtered[max_area_idx]
    
        # สร้าง list ใหม่เพื่อจัดกลุ่ม contour ที่จะอยู่ก่อนและหลัง
        remain_contours = []
        move_to_end = []

        find_status = False
        # วนลูปผ่านแต่ละ contour พร้อมกับ centroid และพื้นที่
        for idx, (cnt, cent, area) in enumerate(zip(filtered_contours, centroids_filtered, areas_filtered)):
            # ถ้าเป็น contour ที่มีพื้นที่มากที่สุด ให้เก็บไว้ใน remain_contours
            if idx == max_area_idx:
                remain_contours.append(cnt)
            else:
                # ตรวจสอบว่า centroid แกน y ของ contour นี้ต่างจาก max_centroid_y ไม่เกิน 20%
                # และมีพื้นที่ไม่น้อยกว่า 30% ของ max_area
                if abs(cent[1] - max_centroid_y) <= 0.2 * max_centroid_y and area >= 0.3 * max_area:
                    print("มีอยู่หลังสุด")
                    move_to_end.append(cnt)
                    find_status = True
                else:
                    remain_contours.append(cnt)
        
        # รวม list ที่เหลือเข้าด้วยกัน โดย contour ที่ตรงเงื่อนไขจะอยู่ท้ายสุด
        filtered_contours = remain_contours + move_to_end

        if find_status == True:
            
            # คำนวณพื้นที่และ centroid แกน x ของ contour ตัวแรกและตัวสุดท้าย
            first_area = cv2.contourArea(filtered_contours[0])
            last_area = cv2.contourArea(filtered_contours[-1])
            first_centroid = get_centroid(filtered_contours[0])
            last_centroid = get_centroid(filtered_contours[-1])

            # ตรวจสอบเงื่อนไข ถ้า contour ตัวแรกมี area มากกว่าและ centroid_x มากกว่าตัวสุดท้าย
            if first_area > last_area and first_centroid[0] > last_centroid[0]:
                print("มีการสลับตัวแรกกับตัวสุดท้าย")
                # สลับตำแหน่ง contour ตัวแรกกับตัวสุดท้าย
                filtered_contours[0], filtered_contours[-1] = filtered_contours[-1], filtered_contours[0]

            # เก็บ x, y, w, h, area ของ filtered_contours เข้าไปใน char_box
            for idx_c, cnt in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                contour_area = int(cv2.contourArea(cnt))
                char_box.append([x, y, w, h, contour_area])

            # คำนวณ index ของ list ที่มี area มากที่สุด
            areas = [item[4] for item in char_box]
            max_index = areas.index(max(areas))

            # ปรับปรุงแต่ละ list ตามเงื่อนไขที่ระบุ
            for idx, box in enumerate(char_box):
                if idx == 0:
                    box.append(1)
                elif idx == (len(char_box)-1):
                    box.append(1)
                else:
                    box.append(2)

            char_level_images = []
            for idx, box in enumerate(char_box):
                x, y, w, h, area, level = box
                crop_img = char_images[y:y+h, x:x+w]
                char_level_images.append([crop_img, level])

            return char_level_images

        else:
            # เก็บ x, y, w, h, area ของ filtered_contours เข้าไปใน char_box
            for idx_c, cnt in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                contour_area = int(cv2.contourArea(cnt))
                char_box.append([x, y, w, h, contour_area])

            # คำนวณ index ของ list ที่มี area มากที่สุด
            areas = [item[4] for item in char_box]
            max_index = areas.index(max(areas))

            # ปรับปรุงแต่ละ list ตามเงื่อนไขที่ระบุ
            for idx, box in enumerate(char_box):
                if idx == max_index:
                    # ถ้าเป็น list ที่มี area มากที่สุด append 1
                    box.append(1)
                elif idx < max_index:
                    # ถ้าอยู่ก่อน list ที่มี area มากที่สุด append 0
                    box.append(0)
                #elif idx == (len(char_box)-1):
                #    # ถ้าเป็น list append 1
                #    box.append(1)
                else:
                    # ถ้าอยู่หลัง list ที่มี area มากที่สุด ให้ append 2
                    box.append(2)

            # คำนวณ index ของ list ที่มี area มากที่สุด
            areas = [item[4] for item in char_box]
            max_index = areas.index(max(areas))

            # ถ้า contour ที่มีพื้นที่มากที่สุดไม่ได้อยู่ลำดับแรก ให้สลับตำแหน่งกับ contour ตัวแรก
            if max_index != 0:
                char_box[0], char_box[max_index] = char_box[max_index], char_box[0]

            char_level_images = []
            for idx, box in enumerate(char_box):
                x, y, w, h, area, level = box
                crop_img = char_images[y:y+h, x:x+w]
                char_level_images.append([crop_img, level])

            return char_level_images


    # เป็นพวกตัวอักษร 1 ระดับ
    # เก็บ x, y, w, h, area ของ filtered_contours เข้าไปใน char_box
    for idx_c, cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        contour_area = int(cv2.contourArea(cnt))
        char_box.append([x, y, w, h, contour_area])

    # คำนวณ index ของ list ที่มี area มากที่สุด
    areas = [item[4] for item in char_box]
    max_index = areas.index(max(areas))

    # สำหรับ list ที่อยู่หลัง list ที่มี area มากที่สุด เราจะเริ่มนับจาก 2
    after_counter = 2

    # ปรับปรุงแต่ละ list ตามเงื่อนไขที่ระบุ
    for idx, box in enumerate(char_box):
        if idx == max_index:
            # ถ้าเป็น list ที่มี area มากที่สุด append 1
            box.append(1)
        elif idx < max_index:
            # ถ้าอยู่ก่อน list ที่มี area มากที่สุด append 0
            box.append(0)
        else:
            # ถ้าอยู่หลัง list ที่มี area มากที่สุด ให้ append sequential number เริ่มจาก 2
            box.append(after_counter)
            #after_counter += 1

    # คำนวณ index ของ list ที่มี area มากที่สุด
    areas = [item[4] for item in char_box]

    max_index = areas.index(max(areas))

    # ถ้า contour ที่มีพื้นที่มากที่สุดไม่ได้อยู่ลำดับแรก ให้สลับตำแหน่งกับ contour ตัวแรก
    if max_index != 0:
        char_box[0], char_box[max_index] = char_box[max_index], char_box[0]

    char_level_images = []
    for idx, box in enumerate(char_box):
        x, y, w, h, area, level = box
        crop_img = char_images[y:y+h, x:x+w]
        char_level_images.append([crop_img, level])

    return char_level_images

# เพิ่มมา
def detect_char(text_group):
    text_group_char = []
    for idx_g, text_g in enumerate(text_group):

        sub_text_char = []
        for idx_s, sub_text in enumerate(text_g):
            rgb_image = cv2.cvtColor(sub_text, cv2.COLOR_GRAY2RGB)
            
            #skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_GUOHALL)

            kernel_dummy = np.ones((3, 3), np.uint8)
            dummy_image = cv2.dilate(skeleton, kernel_dummy, iterations=1)
            kernel_closing = np.ones((7, 1), np.uint8)
            closing = cv2.morphologyEx(dummy_image, cv2.MORPH_CLOSE, kernel=kernel_closing, iterations=1)

            cv2.dilate(skeleton, kernel_dummy, iterations=1)

            contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
            
            char_images_with_levels = []

            for idx_c, cnt in enumerate(sorted_contours):

                x, y, w, h = cv2.boundingRect(cnt)
                contour_area = cv2.contourArea(cnt)
                mask = np.zeros(sub_text.shape[:2], dtype=np.uint8)

                rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

                # วาด contour ลงใน mask โดยเติมเต็ม (thickness = -1) ให้ภายใน contour เป็นสีขาว (255)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                kernel_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ปรับขนาด kernel ตามต้องการ
                dilated_mask = cv2.dilate(mask, kernel_mask, iterations=1)
                
                # ใช้ mask กับภาพต้นฉบับ เพื่อดึงเฉพาะส่วนภายใน contour
                char_result = cv2.bitwise_and(sub_text, sub_text, mask=dilated_mask)
                char_images_with_levels.extend(char_level(char_result))

            sub_text_char.append(char_images_with_levels)
            
        text_group_char.append(sub_text_char)
    return text_group_char

# เพิ่มมา
def Rule_Based_Post_Processing(word):

    if not word:
        return word
    
    word = word.replace("เเ", "แ")
    return word

# เพิ่มมา
def predict_text_multi_level(text_group_char, model_char_level_0, model_char_level_1, model_char_level_2):
    models = {
        0: model_char_level_0,
        1: model_char_level_1,
        2: model_char_level_2,
    }

    char_level_0_label = [
        'ุ', 'ู'
    ]

    char_level_1_label = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',

        'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 
        'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 
        'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 
        'ห', 'ฬ', 'อ', 'ฮ',

        'ะ','า', 'เ', 'แ', 'โ', 'ฤ', 'ใ', 'ไ',

        '+', '-', '*', '(', ')', '.', '=', '/', '|'
    ]

    char_level_2_label = [
         'ิ', 'ี', 'ึ', 'ื', '็', 'ั', 'ํ', '่', '้', '๊', '๋', '์'
    ]

    char_level_labels = {
        0: char_level_0_label,
        1: char_level_1_label,
        2: char_level_2_label,
    }

    # กำหนดขนาด Input ของโมเดล
    input_size = 32  # ขนาด 32x32
    text_block = []

    for idx_g, text_group in enumerate(text_group_char):
        text_result = ""

        for idx_s, sub_text in enumerate(text_group):
            sub_text_result = ""

            for idx_c, char in enumerate(sub_text):
                char_image, char_level = char

                if char is None:
                    print(f"Character image {idx_c} is None.")
                    continue  # ข้ามภาพนี้
                else:

                    # เพิ่ม Padding และปรับขนาดภาพ
                    padded_img = resize_with_min_padding(char_image, input_size, min_padding=1)
                    
                    # Normalization (เปลี่ยนค่าพิกเซลให้อยู่ในช่วง [0, 1])
                    normalized_img = padded_img / 255.0

                    if len(normalized_img.shape) == 2:  # หากภาพเป็น Grayscale (2D)
                        normalized_img = np.expand_dims(normalized_img, axis=-1)
                        processed_image = np.expand_dims(normalized_img, axis=0)  # เพิ่ม Batch Dimension

                        if char_level in models:
                            prediction = models[char_level].predict(processed_image)
                            predicted_class = np.argmax(prediction)
                            confidence_score = np.max(prediction)

                            class_level = char_level_labels[char_level]
                            predicted_letter = class_level[predicted_class]
                        else:
                            print("level ไม่ตรง")
                            prediction = models[2].predict(processed_image)
                            predicted_class = np.argmax(prediction)
                            confidence_score = np.max(prediction)

                            class_level = char_level_labels[2]
                            predicted_letter = class_level[predicted_class]

                        sub_text_result += predicted_letter
            sub_text_result += " "
            text_result += sub_text_result
            text_result_post = Rule_Based_Post_Processing(text_result)
        text_block.append(text_result_post)
        
    print("ประมวลผลเสร็จสิ้น")
    return text_block


def crop_border(image, left_percent=0, right_percent=0, top_percent=0, bottom_percent=0):
    
    # หาความกว้างและความสูงของภาพ
    height, width = image.shape

    # คำนวณพิกัดที่จะตัด (แปลงเป็นพิกเซล)
    x_start = int(width * left_percent)
    x_end = int(width * (1 - right_percent))
    y_start = int(height * top_percent)
    y_end = int(height * (1 - bottom_percent))

    # ตัดภาพ (Crop)
    cropped_img = image[y_start:y_end, x_start:x_end]

    return cropped_img

def find_text_student_info_fh(student_info_fh_img):
    student_info_fh_img = crop_border(student_info_fh_img.copy(), 0.02, 0.80, 0.23, 0.01)

    rgb_image = cv2.cvtColor(student_info_fh_img.copy(), cv2.COLOR_GRAY2RGB)
    
    # กำหนด kernel (ขนาดของ kernel สามารถปรับเปลี่ยนได้ตามความเหมาะสม)
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((6, 100), np.uint8)
    
    opening = cv2.morphologyEx(student_info_fh_img.copy(), cv2.MORPH_OPEN, kernel=kernel_open, iterations=1)
    closing = cv2.morphologyEx(student_info_fh_img, cv2.MORPH_CLOSE, kernel=kernel_close, iterations=1)

    rgb_closing_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)

    # ใช้ Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
    char_stats = stats[1:] # ข้าม Background (index 0)
    sorted_indices = np.argsort(char_stats[:, 1]) # จัดเรียงตามค่า y (คอลัมน์ที่ 1)
    sorted_stats = char_stats[sorted_indices]

    # ใช้ Boolean Indexing เพื่อเอา noise ออก 
    sorted_stats = sorted_stats[sorted_stats[:, 4] >= 1000]
    
    name_coordinate = sorted_stats[2]
    lastname_coordinate = sorted_stats[3]

    cv2.imwrite(f"{output_folder}/opening.jpg", opening)
    cv2.imwrite(f"{output_folder}/closing.jpg", closing)
 
    return name_coordinate, lastname_coordinate

def find_text_student_info_sh(student_info_sh_img, name_coordinate, lastname_coordinate):
    student_info_sh_img = crop_border(student_info_sh_img.copy(), 0.00, 0.15, 0.23, 0.01)

    #student_info_sh_edges = cv2.Canny(student_info_sh_img, 50, 150, apertureSize=5)
    lines_student_info_sh = cv2.HoughLinesP(student_info_sh_img, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=3)
    mask_student_info_sh = np.ones(student_info_sh_img.shape[:2], dtype=np.uint8) * 255

    if lines_student_info_sh is not None:
        for line in lines_student_info_sh:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask_student_info_sh, (x1, y1), (x2, y2), 0, 3)  # ความหนา 3 พิกเซล (ปรับได้ตามต้องการ)
    
    student_info_sh_not_line = cv2.bitwise_and(student_info_sh_img, student_info_sh_img, mask=mask_student_info_sh)

    x, y, w, h, area = name_coordinate
    name_crop = student_info_sh_not_line[y-10:y+h+5, :]
    x, y, w, h, area = lastname_coordinate
    lastname_crop = student_info_sh_not_line[y-10:y+h+7, :]

    cv2.imwrite(f"{output_folder}/name_crop.jpg", name_crop)
    cv2.imwrite(f"{output_folder}/lastname_crop.jpg", lastname_crop)

    return name_crop, lastname_crop

# เพิ่มมา
def detect_sub_text_in_group_stud(binary_images):

    text_group = []

    kernel_open = np.ones((2, 2), np.uint8)
    remove_noise = cv2.morphologyEx(binary_images, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # เช็คว่าภาพเป็นสีดำทั้งหมดหรือไม่
    if not np.any(remove_noise):  # ถ้าค่าพิกเซลทั้งหมดเป็น 0 (ดำสนิท)
        print("ภาพเป็นสีดำทั้งหมด")
        text_group.append(remove_noise)
        #return sub_text_images 
        
    else:
        kernel = np.ones((6, 6), np.uint8)
        dummy_image = cv2.dilate(remove_noise, kernel, iterations=1)

        # ใช้ Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dummy_image, connectivity=8)
        char_stats = stats[1:] # ข้าม Background (index 0)
        sorted_indices = np.argsort(char_stats[:, 0]) # จัดเรียงตามค่า x (คอลัมน์ที่ 0)
        sorted_stats = char_stats[sorted_indices]

        # ใช้ Boolean Indexing เพื่อเอา noise ออก 
        sorted_stats = sorted_stats[sorted_stats[:, 4] >= 200]
        for idx, stats in enumerate(sorted_stats):
            x, y, w, h, area = stats
            cca_img = binary_images[y:y+h, x:x+w]
            text_group.append(cca_img)

    return text_group

# เพิ่มมา
def char_level_stud(char_images):
    debug = False

    char_box = []
    
    skeleton = cv2.ximgproc.thinning(char_images, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    #skeleton = cv2.ximgproc.thinning(char_images, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    kernel_dummy = np.ones((3, 3), np.uint8)
    closing_skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel=kernel_dummy, iterations=1)
    dummy_image = cv2.dilate(skeleton, kernel_dummy, iterations=1)
    closing = cv2.morphologyEx(dummy_image, cv2.MORPH_CLOSE, kernel=kernel_dummy, iterations=1)

    contours, hierarchy = cv2.findContours(dummy_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # เรียง contours ตามค่า y ของ centroid จากน้อยไปหามาก
    sorted_contours = sorted(contours, key=lambda cnt: get_centroid(cnt)[1], reverse=True)

    # คำนวณพื้นที่ของ contour แต่ละตัว และหาพื้นที่ที่ใหญ่ที่สุด
    max_area = max(cv2.contourArea(cnt) for cnt in sorted_contours)

    # กำหนด threshold เป็น 5% ของพื้นที่ที่ใหญ่ที่สุด
    min_area_threshold = max_area * 0.05

    # กรอง contours ที่มีพื้นที่ไม่น้อยกว่า threshold
    filtered_contours = [cnt for cnt in sorted_contours if cv2.contourArea(cnt) >= min_area_threshold]


    # หาก filtered_contours มีแค่ 2 ตัว
    if len(filtered_contours) == 2:

        area0 = cv2.contourArea(filtered_contours[0])
        area1 = cv2.contourArea(filtered_contours[1])
        # คำนวณความแตกต่างเป็นอัตราส่วนของ contour ที่มีพื้นที่ใหญ่กว่า
        diff_ratio = abs(area0 - area1) / max(area0, area1)
        if diff_ratio <= 0.20:
            print("เหมือนกัน")

            kernel_same = np.ones((4, 4), np.uint8)
            dilated_same = cv2.dilate(char_images, kernel_same, iterations=2)
            contours, hierarchy = cv2.findContours(dilated_same, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            level = 1
            x, y, w, h = cv2.boundingRect(largest_contour)
            contour_area = int(cv2.contourArea(largest_contour))
            crop_img = char_images[y:y+h, x:x+w]
            char_box.append([crop_img, level])

            return char_box
    elif len(filtered_contours) >= 3: # เอาไว้แก้ปัญหาพวกคำว่า "ชั้น, บัน" หรือทุกคำที่มี "  ั "
        print("len(filtered_contours) >= 3")
        # คำนวณ centroid และพื้นที่ของแต่ละ contour ใน filtered_contours
        centroids_filtered = [get_centroid(cnt) for cnt in filtered_contours]
        areas_filtered = [cv2.contourArea(cnt) for cnt in filtered_contours]
    
        # หาตำแหน่งของ contour ที่มีพื้นที่มากที่สุด
        max_area_idx = areas_filtered.index(max(areas_filtered))
        max_centroid_y = centroids_filtered[max_area_idx][1]
        max_area = areas_filtered[max_area_idx]
    
        # สร้าง list ใหม่เพื่อจัดกลุ่ม contour ที่จะอยู่ก่อนและหลัง
        remain_contours = []
        move_to_end = []

        find_status = False
        # วนลูปผ่านแต่ละ contour พร้อมกับ centroid และพื้นที่
        for idx, (cnt, cent, area) in enumerate(zip(filtered_contours, centroids_filtered, areas_filtered)):
            # ถ้าเป็น contour ที่มีพื้นที่มากที่สุด ให้เก็บไว้ใน remain_contours
            if idx == max_area_idx:
                remain_contours.append(cnt)
            else:
                # ตรวจสอบว่า centroid แกน y ของ contour นี้ต่างจาก max_centroid_y ไม่เกิน 20%
                # และมีพื้นที่ไม่น้อยกว่า 30% ของ max_area
                if abs(cent[1] - max_centroid_y) <= 0.2 * max_centroid_y and area >= 0.3 * max_area:
                    print("มีอยู่หลังสุด")
                    move_to_end.append(cnt)
                    find_status = True
                else:
                    remain_contours.append(cnt)
        
        # รวม list ที่เหลือเข้าด้วยกัน โดย contour ที่ตรงเงื่อนไขจะอยู่ท้ายสุด
        filtered_contours = remain_contours + move_to_end

        if find_status == True:
            
            # คำนวณพื้นที่และ centroid แกน x ของ contour ตัวแรกและตัวสุดท้าย
            first_area = cv2.contourArea(filtered_contours[0])
            last_area = cv2.contourArea(filtered_contours[-1])
            first_centroid = get_centroid(filtered_contours[0])
            last_centroid = get_centroid(filtered_contours[-1])

            # ตรวจสอบเงื่อนไข ถ้า contour ตัวแรกมี area มากกว่าและ centroid_x มากกว่าตัวสุดท้าย
            if first_area > last_area and first_centroid[0] > last_centroid[0]:
                print("มีการสลับตัวแรกกับตัวสุดท้าย")
                # สลับตำแหน่ง contour ตัวแรกกับตัวสุดท้าย
                filtered_contours[0], filtered_contours[-1] = filtered_contours[-1], filtered_contours[0]

            # เก็บ x, y, w, h, area ของ filtered_contours เข้าไปใน char_box
            for idx_c, cnt in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                contour_area = int(cv2.contourArea(cnt))
                char_box.append([x, y, w, h, contour_area])

            # คำนวณ index ของ list ที่มี area มากที่สุด
            areas = [item[4] for item in char_box]
            max_index = areas.index(max(areas))

            # ปรับปรุงแต่ละ list ตามเงื่อนไขที่ระบุ
            for idx, box in enumerate(char_box):
                if idx == 0:
                    box.append(1)
                elif idx == (len(char_box)-1):
                    box.append(1)
                else:
                    box.append(2)

            char_level_images = []
            for idx, box in enumerate(char_box):
                x, y, w, h, area, level = box
                crop_img = char_images[y:y+h, x:x+w]
                char_level_images.append([crop_img, level])

            return char_level_images

        else:
            # เก็บ x, y, w, h, area ของ filtered_contours เข้าไปใน char_box
            for idx_c, cnt in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                contour_area = int(cv2.contourArea(cnt))
                char_box.append([x, y, w, h, contour_area])

            # คำนวณ index ของ list ที่มี area มากที่สุด
            areas = [item[4] for item in char_box]
            max_index = areas.index(max(areas))

            # ปรับปรุงแต่ละ list ตามเงื่อนไขที่ระบุ
            for idx, box in enumerate(char_box):
                if idx == max_index:
                    # ถ้าเป็น list ที่มี area มากที่สุด append 1
                    box.append(1)
                elif idx < max_index:
                    # ถ้าอยู่ก่อน list ที่มี area มากที่สุด append 0
                    box.append(0)
                #elif idx == (len(char_box)-1):
                #    # ถ้าเป็น list append 1
                #    box.append(1)
                else:
                    # ถ้าอยู่หลัง list ที่มี area มากที่สุด ให้ append 2
                    box.append(2)

            # คำนวณ index ของ list ที่มี area มากที่สุด
            areas = [item[4] for item in char_box]
            max_index = areas.index(max(areas))

            # ถ้า contour ที่มีพื้นที่มากที่สุดไม่ได้อยู่ลำดับแรก ให้สลับตำแหน่งกับ contour ตัวแรก
            if max_index != 0:
                char_box[0], char_box[max_index] = char_box[max_index], char_box[0]

            char_level_images = []
            for idx, box in enumerate(char_box):
                x, y, w, h, area, level = box
                crop_img = char_images[y:y+h, x:x+w]
                char_level_images.append([crop_img, level])
        
            return char_level_images

 #########################
    # เก็บ x, y, w, h, area ของ filtered_contours เข้าไปใน char_box
    for idx_c, cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        contour_area = int(cv2.contourArea(cnt))
        char_box.append([x, y, w, h, contour_area])

    # คำนวณ index ของ list ที่มี area มากที่สุด
    areas = [item[4] for item in char_box]
    max_index = areas.index(max(areas))

    # สำหรับ list ที่อยู่หลัง list ที่มี area มากที่สุด เราจะเริ่มนับจาก 2
    after_counter = 2

    # ปรับปรุงแต่ละ list ตามเงื่อนไขที่ระบุ
    for idx, box in enumerate(char_box):
        if idx == max_index:
            # ถ้าเป็น list ที่มี area มากที่สุด append 1
            box.append(1)
        elif idx < max_index:
            # ถ้าอยู่ก่อน list ที่มี area มากที่สุด append 0
            box.append(0)
        else:
            # ถ้าอยู่หลัง list ที่มี area มากที่สุด ให้ append sequential number เริ่มจาก 2
            box.append(after_counter)
            #after_counter += 1

    # คำนวณ index ของ list ที่มี area มากที่สุด
    areas = [item[4] for item in char_box]

    max_index = areas.index(max(areas))

    # ถ้า contour ที่มีพื้นที่มากที่สุดไม่ได้อยู่ลำดับแรก ให้สลับตำแหน่งกับ contour ตัวแรก
    if max_index != 0:
        char_box[0], char_box[max_index] = char_box[max_index], char_box[0]
        
    char_level_images = []
    for idx, box in enumerate(char_box):
        x, y, w, h, area, level = box
        crop_img = char_images[y:y+h, x:x+w]
        char_level_images.append([crop_img, level])
        
    return char_level_images

# เพิ่มมา
def detect_char_stud(text_group):

    sub_text_char = []
    for idx_s, sub_text in enumerate(text_group):
        rgb_image = cv2.cvtColor(sub_text, cv2.COLOR_GRAY2RGB)
            
        #skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_GUOHALL)

        kernel_dummy = np.ones((2, 2), np.uint8)
        dummy_image = cv2.dilate(skeleton, kernel_dummy, iterations=1)
        kernel_closing = np.ones((7, 1), np.uint8)
        closing = cv2.morphologyEx(dummy_image, cv2.MORPH_CLOSE, kernel=kernel_closing, iterations=1)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
            
        char_images_with_levels = []
        
        for idx_c, cnt in enumerate(sorted_contours):

            x, y, w, h = cv2.boundingRect(cnt)
            contour_area = cv2.contourArea(cnt)

            mask = np.zeros(sub_text.shape[:2], dtype=np.uint8)

            rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

            # วาด contour ลงใน mask โดยเติมเต็ม (thickness = -1) ให้ภายใน contour เป็นสีขาว (255)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            kernel_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ปรับขนาด kernel ตามต้องการ
            dilated_mask = cv2.dilate(mask, kernel_mask, iterations=1)

            # ใช้ mask กับภาพต้นฉบับ เพื่อดึงเฉพาะส่วนภายใน contour
            char_result = cv2.bitwise_and(sub_text, sub_text, mask=dilated_mask)
            char_images_with_levels.extend(char_level_stud(char_result))

        sub_text_char.append(char_images_with_levels)
         
    return sub_text_char

# เพิ่มมา
def predict_text_multi_level_stud(text_group_char, model_char_level_0, model_char_level_1, model_char_level_2):
    models = {
        0: model_char_level_0,
        1: model_char_level_1,
        2: model_char_level_2,
    }

    char_level_0_label = [
        'ุ', 'ู'
    ]

    char_level_1_label = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',

        'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 
        'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 
        'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 
        'ห', 'ฬ', 'อ', 'ฮ',

        'ะ','า', 'เ', 'แ', 'โ', 'ฤ', 'ใ', 'ไ',

        '+', '-', '*', '(', ')', '.', '=', '/', '|'
    ]

    char_level_2_label = [
         'ิ', 'ี', 'ึ', 'ื', '็', 'ั', 'ํ', '่', '้', '๊', '๋', '์'
    ]

    char_level_labels = {
        0: char_level_0_label,
        1: char_level_1_label,
        2: char_level_2_label,
    }

    # กำหนดขนาด Input ของโมเดล
    input_size = 32  # ขนาด 32x32

    text_result = ""
    for idx_s, sub_text in enumerate(text_group_char):
        sub_text_result = ""

        for idx_c, char in enumerate(sub_text):
            char_image, char_level = char

            if char is None:
                print(f"Character image {idx_c} is None.")
                continue  # ข้ามภาพนี้
            else:

                # เพิ่ม Padding และปรับขนาดภาพ
                padded_img = resize_with_min_padding(char_image, input_size, min_padding=1)

                '''
                plt.figure(figsize=(2, 2))
                plt.imshow(padded_img, cmap="gray")
                plt.title(f"Char, text group:{idx_g+1}, sub text:{idx_s+1}, char:{idx_c+1}, level: {char_level}")
                plt.show()
                '''
                
                # Normalization (เปลี่ยนค่าพิกเซลให้อยู่ในช่วง [0, 1])
                normalized_img = padded_img / 255.0

                if len(normalized_img.shape) == 2:  # หากภาพเป็น Grayscale (2D)
                    normalized_img = np.expand_dims(normalized_img, axis=-1)
                    processed_image = np.expand_dims(normalized_img, axis=0)  # เพิ่ม Batch Dimension

                    if char_level in models:
                        prediction = models[char_level].predict(processed_image)
                        predicted_class = np.argmax(prediction)
                        confidence_score = np.max(prediction)

                        class_level = char_level_labels[char_level]
                        predicted_letter = class_level[predicted_class]
                    else:
                        print("level ไม่ตรง")
                        prediction = models[2].predict(processed_image)
                        predicted_class = np.argmax(prediction)
                        confidence_score = np.max(prediction)

                        class_level = char_level_labels[2]
                        predicted_letter = class_level[predicted_class]

                    sub_text_result += predicted_letter

        sub_text_result += " "
        text_result += sub_text_result
        text_result_post = Rule_Based_Post_Processing(text_result)
        #text_block.append(text_result_post)
        
    print("ประมวลผลเสร็จสิ้น")
    return text_result_post

## หน้าหลัง
def fine_table(binary_img, original_denoised, dummy):

    # แยกตาราง
    #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dummy, connectivity=8)
    areas = [stat[4] for stat in stats]  # ดึงค่า area
    sorted_areas = sorted(areas, reverse=True)  # เรียงลำดับจากมากไปน้อย
    second_max_area = sorted_areas[1]  # ค่าอันดับ 2
    second_max_area_index = areas.index(second_max_area)  # หาตำแหน่งในลิสต์เดิม
    table_position = stats[second_max_area_index]
    x, y, w, h, area = table_position

    table_img = binary_img[y:y+h, x:x+w]
    table_dummy_img = dummy[y:y+h, x:x+w]
    table_original_img = original_denoised[y:y+h, x:x+w]

    return table_img, table_dummy_img, table_original_img

def find_table_columns_b(table_dummy_persective_img, table_persective_img):

    output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_HS_DJ/back")
    output_folder.mkdir(exist_ok=True)

    # คำนวณ vertical projection (ผลรวมของ pixel ในแต่ละคอลัมน์)
    vertical_proj = np.sum(table_dummy_persective_img, axis=0)

    # ตั้ง threshold สำหรับเลือกเฉพาะคอลัมน์ที่มี "เส้น"
    line_threshold = np.max(vertical_proj) * 0.40

    # หา indices ของคอลัมน์ที่มีค่า projection มากกว่าค่า threshold
    col_line_indices = [i for i, value in enumerate(vertical_proj) if value > line_threshold]

    # รวมกลุ่ม indices ที่ติดกัน
    groups = group_indices(col_line_indices, gap=1)

    # สำหรับแต่ละกลุ่มหา index กลางเป็นตำแหน่งของเส้น
    col_lines = [int(np.mean(group)) for group in groups]
    col_lines.sort()  # เรียงลำดับจากซ้ายไปขวา

    # คำนวณความหนาแน่นในแต่ละกลุ่มโดยรวมค่า vertical projection ในช่วงนั้น
    group_densities = [np.sum(vertical_proj[group[0]:group[-1]+1]) for group in groups]
    groups_with_density = list(zip(groups, group_densities))
    groups_with_density.sort(key=lambda x: x[1], reverse=True)
    top_groups = groups_with_density[:3]
    col_lines = [int(np.mean(group)) for group, _ in top_groups]
    col_lines.sort()

    print("ตำแหน่งของเส้นคอลัมน์ที่ตรวจจับได้:", col_lines)

    # --- สร้าง Mask จากเส้นคอลัมน์ ---
    mask = np.zeros_like(table_dummy_persective_img)

    # วาดเส้นคอลัมน์ลงใน mask image (ใช้สีขาว = 255)
    for x in col_lines:
        # กำหนดความหนาของเส้นได้ตามต้องการ (ที่นี้ใช้ thickness=2)
       cv2.line(mask, (x, 0), (x, mask.shape[0]-1), 255, thickness=13)
    cv2.imwrite(f"{output_folder}/cols/Column Lines Mask.png", mask)

    # --- ใช้ Mask เพื่อลบเส้นออกจากภาพด้วย inpainting ---
    # โดยจะใช้เทคนิค inpaint (Telea method) ในการเติมเต็มบริเวณที่มีเส้น
    table_no_lines = cv2.inpaint(table_persective_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(f"{output_folder}/cols/Table_No_Lines.png", table_no_lines)

    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นออกจากภาพ ---
    # โดยเราจะใช้ inverted mask (mask_inv) ที่มีค่า 0 ในบริเวณเส้น
    mask_inv = cv2.bitwise_not(mask)

    # bitwise_and จะเก็บเฉพาะส่วนที่ mask_inv มีค่า != 0 (ส่วนที่ไม่เป็นเส้น)
    img_no_lines = cv2.bitwise_and(table_persective_img, table_persective_img, mask=mask_inv)
    cv2.imwrite(f"{output_folder}/cols/Table_No_Lines_Bitwise.png", img_no_lines)

    # Crop ภาพโดยใช้พิกัดของเส้นที่ตรวจจับได้
    cropped_segments = []
    cropped_dummy_segments = []
    # loop จะทำงานจากเส้นที่ 1 ถึงเส้นที่ 10 (หมายถึง crop ภาพระหว่างเส้นที่ i และ i+1)
    for i in range(len(col_lines) - 1):
        x_start = col_lines[i]
        x_end = col_lines[i+1]
        cropped = img_no_lines[:, x_start:x_end]  # crop ทุกแถวในช่วง x ที่กำหนด
        dummy_cropped = table_dummy_persective_img[:, x_start:x_end]  # crop ทุกแถวในช่วง x ที่กำหนด
        cropped_segments.append(cropped)
        cropped_dummy_segments.append(dummy_cropped)
        cv2.imwrite(f"{output_folder}/cols/cropped_segment_{i+1}.png", cropped)
        cv2.imwrite(f"{output_folder}/cols/cropped_dummy_segment_{i+1}.png", dummy_cropped)

    return cropped_segments, cropped_dummy_segments

def find_table_subject_group(cell_img):
    output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_HS_DJ/back")
    output_folder.mkdir(exist_ok=True)

    kernel = np.ones((3, 3), np.uint8)
    cell_img_dilated = cv2.dilate(cell_img.copy(), kernel, iterations=1)
    cv2.imwrite(f"{output_folder}/cols/rows/cell_img_dilated.png", cell_img_dilated)
    
    # คำนวณ horizontal projection (ผลรวมของ pixel ในแต่ละแถว)
    horizontal_proj = np.sum(cell_img_dilated, axis=1)
    vertical_proj = np.sum(cell_img_dilated, axis=0)
    
    # ตั้ง threshold สำหรับเลือกเฉพาะแถวที่มี "เส้น"
    horizontal_line_threshold = np.max(horizontal_proj) * 0.9
    vertical_line_threshold = np.max(vertical_proj) * 0.7
    
    # หา indices ของแถวที่มีค่า projection มากกว่าค่า threshold
    row_line_indices = [i for i, value in enumerate(horizontal_proj) if value > horizontal_line_threshold]
    col_line_indices = [i for i, value in enumerate(vertical_proj) if value > vertical_line_threshold]
    
    # รวมกลุ่ม indices ที่ติดกัน (ต้องมีฟังก์ชัน group_indices ที่คุณนิยามไว้แล้ว)
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
    top_groups_row = groups_row_with_length[:4]
    top_groups_col = groups_col_with_length[:2]

    # คำนวณตำแหน่งเส้นโดยการหาค่าเฉลี่ยของแต่ละกลุ่ม แล้วเรียงลำดับจากบนลงล่าง
    row_lines = [int(np.mean(group)) for group, _ in top_groups_row]
    col_lines = [int(np.mean(group)) for group, _ in top_groups_col]
    row_lines.sort()
    col_lines.sort()
    
    print("ตำแหน่งของเส้นแถวที่ตรวจจับได้:", row_lines)
    print("ตำแหน่งของเส้นคอลัมน์ที่ตรวจจับได้:", col_lines)
    
    # --- สร้าง Mask จากเส้นแถว ---
    mask_row = np.zeros_like(cell_img_dilated)
    mask_col = np.zeros_like(cell_img_dilated)
    
    # วาดเส้นแถวลงใน mask (เส้นแนวนอน)
    for y in row_lines:
        cv2.line(mask_row, (0, y), (mask_row.shape[1]-1, y), 255, thickness=10)
    cv2.imwrite(f"{output_folder}/cols/rows/row_lines_mask.png", mask_row)

    # วาดเส้นคอลัมน์ลงใน mask (เส้นแนวตั้ง)
    for x in col_lines:
        cv2.line(mask_col, (x, 0), (x, mask_col.shape[0]-1), 255, thickness=10)
    cv2.imwrite(f"{output_folder}/cols/rows/col_lines_mask.png", mask_col)
    
    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นแถวออกจากภาพ ---
    mask_row_inv = cv2.bitwise_not(mask_row)
    img_no_lines_row = cv2.bitwise_and(cell_img, cell_img, mask=mask_row_inv)
    cv2.imwrite(f"{output_folder}/cols/rows/table_no_lines_bitwise_row.png", img_no_lines_row)

    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นแนวตั้งออกจากภาพ ---
    mask_col_inv = cv2.bitwise_not(mask_col)
    img_no_lines_col = cv2.bitwise_and(cell_img, cell_img, mask=mask_col_inv)
    cv2.imwrite(f"{output_folder}/cols/rows/table_no_lines_bitwise_col.png", img_no_lines_col)

    # --- ใช้ cv2.bitwise_and เพื่อลบเส้นแนวตั้งและแนวนอนออกจากภาพ ---
    combined_mask = cv2.bitwise_or(mask_row, mask_col)
    combined_mask_inv = cv2.bitwise_not(combined_mask)
    img_no_lines_row_col = cv2.bitwise_and(cell_img, cell_img, mask=combined_mask_inv)
    
    cv2.imwrite(f"{output_folder}/cols/rows/img_no_lines_row_col.png", img_no_lines_row_col)


    # --- Crop ภาพโดยใช้พิกัดของเส้นที่ตรวจจับได้ ---
    cropped_row_segments = []
    # loop จะทำงานจากเส้นที่ 1 ถึงเส้นที่ n-1 (crop ภาพระหว่างเส้นที่ i และ i+1)\

    for i in range(len(row_lines) - 1):
        y_start = row_lines[i]
        y_end = row_lines[i+1]
        cropped = img_no_lines_row_col[y_start:y_end, :]  # crop ทุกคอลัมน์ในช่วงแถวที่กำหนด
        cropped_row_segments.append(cropped)
        cv2.imwrite(f"{output_folder}/cols/rows/Cropped Segment {i+1}.png", cropped)

    for idx, row_img in enumerate(cropped_row_segments[1:]):
        x_start = col_lines[0]
        x_end = col_lines[1]
        if(idx == 0):
            department_credits_img = row_img[:, x_start:x_end]
            department_academic_results_img = row_img[:, x_end:]
            cv2.imwrite(f"{output_folder}/cols/rows/department_credits.png", department_credits_img)
            cv2.imwrite(f"{output_folder}/cols/rows/department_academic_results.png", department_academic_results_img)
        else:
            sum_department_credits_img = row_img[:, x_start:x_end]
            gpa_img = row_img[:, x_end:]
            cv2.imwrite(f"{output_folder}/cols/rows/sum_department_credits.png", sum_department_credits_img)
            cv2.imwrite(f"{output_folder}/cols/rows/gpa.png", gpa_img)

    return department_credits_img, department_academic_results_img, sum_department_credits_img, gpa_img

def detect_text_subject_group(cell_img):
    texts = []

    kernel_open = np.ones((3, 3), np.uint8)
    remove_noise = cv2.morphologyEx(cell_img, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel = np.ones((3, 8), np.uint8)
    dilate_img = cv2.dilate(remove_noise, kernel, iterations=2)
    rgb_image = cv2.cvtColor(cell_img.copy(), cv2.COLOR_GRAY2RGB)

    # ใช้ Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_img, connectivity=8)
    text_stats = stats[1:]
    sorted_indices = np.argsort(text_stats[:, 1])  # จัดเรียงตามค่า y (คอลัมน์ที่ 1)
    sorted_stats = text_stats[sorted_indices]
    
    # ใช้ Boolean Indexing เพื่อเอา noise ออก 
    sorted_stats = sorted_stats[sorted_stats[:, 4] >= 800]

    for idx, stats in enumerate(sorted_stats): # เก็บภาพกลุม
        x, y, w, h, area = stats
        #print(f"CCA #{idx}: bounding box = (x={x}, y={y}, w={w}, h={h}, area={area})")
        crop_img = cell_img[y:y+h, x:x+w]
        texts.append(crop_img)

    return texts

def detect_one_level_of_char_back(text_group):
    text_group_char = []
    for idx_g, text in enumerate(text_group):

        #skeleton = cv2.ximgproc.thinning(text, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        skeleton_guohall = cv2.ximgproc.thinning(text, thinningType=cv2.ximgproc.THINNING_GUOHALL)

        #kernel_open = np.ones((2, 2), np.uint8)
        kernel_dummy = np.ones((2, 2), np.uint8)
        #opening = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel=kernel_open, iterations=2)
        #closing = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel=kernel_open, iterations=2)
        dummy_image = cv2.dilate(skeleton_guohall, kernel_dummy, iterations=1)

        contours, hierarchy = cv2.findContours(dummy_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        char_images = []
        for idx_c, cnt in enumerate(sorted_contours):

            x, y, w, h = cv2.boundingRect(cnt)
            contour_area = cv2.contourArea(cnt)

            mask = np.zeros(text.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            kernel_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ปรับขนาด kernel ตามต้องการ
            dilated_mask = cv2.dilate(mask, kernel_mask, iterations=1)

            # ใช้ mask กับภาพต้นฉบับ เพื่อดึงเฉพาะส่วนภายใน contour
            char_result = cv2.bitwise_and(text, text, mask=dilated_mask)

            contours_char, hierarchy_char = cv2.findContours(char_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours_char, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)
            area = int(cv2.contourArea(largest_contour))
            crop_img = char_result[y:y+h, x:x+w]
            char_images.append(crop_img)

        text_group_char.append(char_images)
    return text_group_char


def predict_text_one_level_b(text_group_char, char_model, model_char_credit):
    models_one_level = {
        1: model_char_credit,
    }
    char_credit = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'
    ]

    char_labels = {
        1: char_credit,
    }
    # กำหนดขนาด Input ของโมเดล
    input_size = 32  # ขนาด 32x32
    text_block = []

    for idx_g, text_group in enumerate(text_group_char):
        text_result = ""

        for idx_c, char in enumerate(text_group):

            if char is None:
                print(f"Character image {idx_c} is None.")
                continue  # ข้ามภาพนี้
            else:
                '''
                if not np.any(sub_text):
                    print("ไม่มี")
                    sub_text_result = "-"
                    continue
                '''

                # เพิ่ม Padding และปรับขนาดภาพ
                padded_img = resize_with_min_padding(char, input_size, min_padding=1)

                # Normalization (เปลี่ยนค่าพิกเซลให้อยู่ในช่วง [0, 1])
                normalized_img = padded_img / 255.0

                if len(normalized_img.shape) == 2:  # หากภาพเป็น Grayscale (2D)
                    normalized_img = np.expand_dims(normalized_img, axis=-1)
                    processed_image = np.expand_dims(normalized_img, axis=0)  # เพิ่ม Batch Dimension

                if char_model in models_one_level:
                    prediction = models_one_level[char_model].predict(processed_image)
                    predicted_class = np.argmax(prediction)
                    confidence_score = np.max(prediction)

                    class_char = char_labels[char_model]
                    predicted_letter = class_char[predicted_class]
                    text_result += predicted_letter
        text_block.append(text_result)

    print("ประมวลผลเสร็จสิ้น")
    return text_block