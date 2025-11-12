import cv2
import numpy as np
import math
from easyocr import Reader

def find_risks(image_path, threshold):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 1. Улучшение контраста и уменьшение шума
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 2. Обнаружение границ
    edges = cv2.Canny(denoised, 50, 150, apertureSize=3)

    # 3. Усиление вертикальных/горизонтальных линий (рисок)
    kernel_vertical = np.ones((10, 2), np.uint8)  # Для вертикальных рисок
    kernel_horizontal = np.ones((2, 10), np.uint8)  # Для горизонтальных рисок

    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_horizontal)

    # 4. Поиск контуров рисок
    all_risks = []
    contours_vertical, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_horizontal, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация маленьких контуров (рисок)
    min_risk_area = 5  # Минимальная площадь риски
    max_risk_area = 500  # Максимальная площадь риски

    for contour in contours_vertical + contours_horizontal:
        area = cv2.contourArea(contour)
        if min_risk_area < area < max_risk_area:
            # Получаем bounding box риски
            x, y, w, h = cv2.boundingRect(contour)
            all_risks.append((x, y, w, h))


    avg_value = 0
    n = len(all_risks)
    for _, _, w, h in all_risks:
        avg_value += (w*h)
    avg_value /= n


    new_filtered = set()
    for i in range(len(all_risks)):
        x, y, w, h = all_risks[i]
        cnt = 0
        for j in range(len(all_risks)):
            x2, y2, _, _ = all_risks[j]
            if i == j:
                continue
            
            dist = math.sqrt((x2 - x)**2 + (y - y2)**2)
            if dist <= threshold:
                cnt += 1
            if cnt >= 5:
                new_filtered.add(all_risks[i])
                break
    
    
    return list(new_filtered)


def find_digits(img):
    reader = Reader(['en'])
    res = reader.readtext(img)

    digits = []

    for i in res:
        coords, number, _ = i
        leftup, rightup, rightdown, leftdown = coords

        center_x = int((leftup[0] + rightup[0] + rightdown[0] + leftdown[0]) / 4)
        center_y = int((leftup[1] + rightup[1] + rightdown[1] + leftdown[1]) / 4)

        if number.isdigit():
            digits.append((int(number), center_x, center_y))
    
    digits.sort()

    return digits


def assign_values_to_risks(risks, digits, image_path):
    digit_mapping = []
    for digit, x, y in digits:
        minDist = float('inf')
        sqr = 0
        xmin, ymin = 0, 0
        for x1, y1, w, h in risks:
            dist = math.sqrt((x1 - x)**2 + (y - y1)**2)
            cur_sqr = w*h
            if dist < 150:
                if cur_sqr > sqr:
                    minDist = dist
                    xmin, ymin = x1, y1
                    sqr = cur_sqr

            elif dist < minDist:
                minDist = dist
                xmin, ymin = x1, y1
                sqr = cur_sqr
        digit_mapping.append((digit, xmin, ymin))


    risks.sort() # сортируем по первому числу
    print(risks)
    


    if len(digit_mapping) < 2:
        raise RuntimeError("not enough digits were found")
    d1, firstx, firsty = digit_mapping[0]
    d2, secx, secy = digit_mapping[1]
    first, second = 0, 0

    for i in range(len(risks)):
        if risks[i][0] == firstx and risks[i][1] == firsty:
            first = i
        elif risks[i][0] == secx and risks[i][1] == secy:
            second = i
        if second != 0 and first != 0:
            break

    diff = second - first
    value_division = (d2 - d1) / diff
    print("value_division", value_division, "=====================================")

    assinged_risks = []
    i, j = 0, 0
    cnt = 0
    while i < len(digit_mapping) and j < len(risks):
        x1, y1, w, h = risks[j]
        digit, x, y = digit_mapping[i]
        if (x1, y1) == (x, y):
            cnt = digit
            assinged_risks.append((cnt, x1, y1))
            i += 1
            continue
        assinged_risks.append((cnt, x1, y1))
        cnt += value_division
        j += 1


    image = cv2.imread(image_path)
    result = image.copy()
    for i, (x, y, w, h) in enumerate(risks):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for d, x, y in assinged_risks:
        cv2.putText(result, str(d), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    cv2.imwrite("result_image.jpg", result)

    return assinged_risks




def main():
    image = "photo/ank.jpg"
    risks = find_risks(image,60)
    print(risks)
    digits = find_digits(image)
    
    assinged_risks = assign_values_to_risks(risks, digits, image)
    print(assinged_risks)


if __name__ == "__main__":
    main()









