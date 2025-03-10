import pytesseract
import cv2
from matplotlib import table
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_DJ")
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
    output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_DJ")
    output_folder.mkdir(exist_ok=True)

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

                cca_img = img[y:y+h, x:x+w]
                sub_text_images.append(cca_img)

        text_group.append(sub_text_images)

    return text_group

# เพิ่ม
def get_centroid(contour):
    # คำนวณ moments ของ contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0, 0
    return (cx, cy)

# เพื่ม
def detect_one_level_of_char(text_group):

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

# เพิ่ม
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
        if diff_ratio <= 0.15:
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

    # เป็นพวกตัวอักษร 1 ตัว
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

# เพิ่ม
def detect_char(text_group):
    output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_DJ")
    output_folder.mkdir(exist_ok=True)

    text_group_char = []
    for idx_g, text_g in enumerate(text_group):


        sub_text_char = []
        for idx_s, sub_text in enumerate(text_g):
            rgb_image = cv2.cvtColor(sub_text, cv2.COLOR_GRAY2RGB)
            
            skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            #skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_GUOHALL)

            kernel_dummy = np.ones((3, 3), np.uint8)
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
                char_images_with_levels.extend(char_level(char_result))

            sub_text_char.append(char_images_with_levels)
            
        text_group_char.append(sub_text_char)
    return text_group_char

# เพิ่ม
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

# เพิ่ม
def predict_text_one_level(text_group_char, char_model, model_char_subject_code_tn, model_char_academic_results_tn):
    models_one_level = {
        0: model_char_subject_code_tn,
        1: model_char_academic_results_tn,
    }

    char_subject_code_tn = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-',
    ]

    char_academic_results_tn = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'ก', 'ข', 'ถ', 'ท', 'น', 'ป', 'ผ', 'ม', 'ร', 'ล', 'ส', '.'
    ]

    char_labels = {
        0: char_subject_code_tn,
        1: char_academic_results_tn,
    }

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

# เพื่ม
def Rule_Based_Post_Processing(word):

    if not word:
        return word
    word = word.replace("เเ", "แ")

    return word

# เพิ่ม
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

    char_level_3_label = [
        '่', '้', '๊', '๋',
    ]

    char_level_labels = {
        0: char_level_0_label,
        1: char_level_1_label,
        2: char_level_2_label,
        3: char_level_3_label,
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
        text_block.append(text_result_post)
        
    print("ประมวลผลเสร็จสิ้น")
    return text_block 



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
    output_folder = Path("C:/Users/Impan/Documents/ocr-engine-python/data/output_images/output_V6_TN_DJ")
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

# เพิ่ม
def char_level_stud(char_images):

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
        if diff_ratio <= 0.05:
            print("เหมือนกัน")

            kernel_same = np.ones((4, 4), np.uint8)
            contours, hierarchy = cv2.findContours(char_images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # 1 ตัวอักษร
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
        
# เพิ่ม
def detect_char_stud(text_group):

    sub_text_char = []
    for idx_s, sub_text in enumerate(text_group):
        rgb_image = cv2.cvtColor(sub_text, cv2.COLOR_GRAY2RGB)
            
        skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        #skeleton = cv2.ximgproc.thinning(sub_text, thinningType=cv2.ximgproc.THINNING_GUOHALL)

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

# เพิ่ม
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
        
    print("ประมวลผลเสร็จสิ้น")
    return text_result_post