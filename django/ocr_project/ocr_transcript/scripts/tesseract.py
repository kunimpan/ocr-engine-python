import pytesseract
import cv2
from matplotlib import table
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# โหลดรูป
output_folder = Path("data/output_images/output_V5")
output_folder.mkdir(exist_ok=True)

image = cv2.imread("data/test_images/transcript/img-3.png")
if image is None:
    raise FileNotFoundError("ไม่พบไฟล์ภาพ กรุณาตรวจสอบเส้นทางของไฟล์")

# จำกัด noise
denoised = cv2.bilateralFilter(image, 50, 100, 100)
cv2.imwrite(f"{output_folder}/transcript_denoised.png", denoised)

gray_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

# สำหรับภาพที่แสงไม่สม่ำเสมอ
binary_gaussian = cv2.adaptiveThreshold(
    gray_img, 
    maxValue=255, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=41, 
    C=20
)

cv2.imwrite(f"{output_folder}/transcript.png", image)
cv2.imwrite(f"{output_folder}/transcript_gray.png", gray_img)
cv2.imwrite(f"{output_folder}/transcript_binary_g.png", binary_gaussian)

# แยกตารางกับข้อมูลส่วนบน
# แยกตาราง
original_img = image
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_gaussian, connectivity=8) # cca
areas = [stat[4] for stat in stats]  # ดึงค่า area
sorted_areas = sorted(areas, reverse=True)  # เรียงลำดับจากมากไปน้อย
second_max_area = sorted_areas[1]  # ค่าอันดับ 2
second_max_area_index = areas.index(second_max_area)  # หาตำแหน่งในลิสต์เดิม

table_position = stats[second_max_area_index]
x, y, w, h, area = table_position 
table_img = binary_gaussian[y:y+h, x:x+w]
cv2.imwrite(f"{output_folder}/table.png", table_img)

# แยกประวัติส่วนตัวของจากตาราง
upper_part = binary_gaussian[:y, :]
upper_part_gray = gray_img[:y, :]
upper_part_color = original_img[:y, :]
cv2.imwrite(f"{output_folder}/upper_part_color.png", upper_part_color)

# หารูปโปรไฟล์
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(upper_part, connectivity=8)
areas = [stat[4] for stat in stats]  # ดึงค่า area
sorted_areas = sorted(areas, reverse=True)  # เรียงลำดับจากมากไปน้อย
second_max_area = sorted_areas[1]  # ค่าอันดับ 2
second_max_area_index = areas.index(second_max_area)  # หาตำแหน่งในลิสต์เดิม

profile_position = stats[second_max_area_index]
x, y, w, h, area = profile_position
profile_img = upper_part[y:y+h, x:x+w]

# ประวัติส่วนตัวที่เอารูปโปรไฟล์ออก
personal_record = upper_part[:, :x-10]
personal_record_gray = upper_part_gray[:, :x-10]
cv2.imwrite(f"{output_folder}/personal_record.png", personal_record)
cv2.imwrite(f"{output_folder}/personal_record_gray.png", personal_record_gray)

def detect_text_group(dilalated_image, binary_image):

    text_group_images = []
    # ใช้ Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilalated_image, connectivity=8)

    # กรองข้อมูล Background และจัดเรียงจากบนไปลงล่าง (ตามค่า y)
    char_stats = stats[1:]  # ข้าม Background (index 0)
    #sorted_indices = np.argsort(char_stats[:, 1])  # จัดเรียงตามค่า y (คอลัมน์ที่ 1)
    #sorted_stats = char_stats[sorted_indices]

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
            cv2.imwrite(f"{output_folder}/personal_record/cca_{idx}.jpg", cca_img)
            cv2.rectangle(image, (x_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), (0, 255, 0), 1)

    return text_group_images

kernel = np.ones((6, 80), np.uint8)
lines_personal = cv2.dilate(personal_record, kernel, iterations=1)
cv2.imwrite(f"{output_folder}/personal_record/find_lines.png", lines_personal)

text_group_personal_images = detect_text_group(lines_personal, personal_record)

custom_config = r'--oem 3 --psm 7'  # ใช้ OCR Engine Mode และ Page Segmentation Mode ที่เหมาะสม

record_text_box = []
for idx, text_img in enumerate(text_group_personal_images):
    text = pytesseract.image_to_string(text_img, config=custom_config, lang='tha')
    #plt.figure(figsize=(5, 5))
    #plt.imshow(text_img)
    #plt.imshow(text_img, cmap="gray")
    #plt.title(f"dilated_lines")
    plt.show()
    text_cleaned = text.replace("\n", "")  # ลบ \n ออก
    record_text_box.append(text_cleaned)
    print(text_cleaned)






