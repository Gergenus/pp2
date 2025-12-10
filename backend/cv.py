import cv2
import numpy as np
import math
from easyocr import Reader
import os
import tempfile
from fastapi import UploadFile


def find_risks(image_path, threshold):
    """
    Находит риски (деления) на шкале.
    Возвращает список рисок, Canny-edges и изображение в оттенках серого.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    edges = cv2.Canny(denoised, 50, 150, apertureSize=3)

    kernel_vertical = np.ones((10, 2), np.uint8)
    kernel_horizontal = np.ones((2, 10), np.uint8)

    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_horizontal)

    all_risks = []
    contours_vertical, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_horizontal, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_risk_area = 5
    max_risk_area = 500

    for contour in contours_vertical + contours_horizontal:
        area = cv2.contourArea(contour)
        if min_risk_area < area < max_risk_area:
            x, y, w, h = cv2.boundingRect(contour)
            all_risks.append((x, y, w, h))

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
    
    return list(new_filtered), edges, gray

def find_digits(img):
    """
    Находит цифры и их центральные координаты на изображении.
    """
    reader = Reader(['en'])
    res = reader.readtext(img)

    digits = []

    for i in res:
        coords, number, _ = i
        cleaned_number = number.replace('.', '').replace('-', '')
        
        if cleaned_number.isdigit():
            try:
                numeric_value = float(number) 
                leftup, rightup, rightdown, leftdown = coords

                center_x = int((leftup[0] + rightup[0] + rightdown[0] + leftdown[0]) / 4)
                center_y = int((leftup[1] + rightup[1] + rightdown[1] + leftdown[1]) / 4)

                digits.append((numeric_value, center_x, center_y))
            except ValueError:
                continue
    
    digits.sort()
    return digits

def find_pivot_point(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")
        
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    roi_height = int(height * 0.30) 
    roi_y_start = height - roi_height
    
    roi_gray = gray[roi_y_start:height, 0:width]
    
    blurred = cv2.medianBlur(roi_gray, 5)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 
                               minDist=50,
                               param1=50, 
                               param2=25,
                               minRadius=5, 
                               maxRadius=int(width * 0.1))

    pivot_center = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        best_circle = None
        min_dist_from_mid_x = float('inf')
        
        mid_x = width // 2
        
        for (cx, cy, r) in circles:
            dist = abs(cx - mid_x)
            if dist < min_dist_from_mid_x:
                min_dist_from_mid_x = dist
                best_circle = (cx, cy, r)
        
        local_x, local_y, r = best_circle
        global_x = local_x
        global_y = local_y + roi_y_start
        
        pivot_center = (global_x, global_y)
        print(f"Центр (ось) найден через Hough: {pivot_center}")

    else:
        print("Винтик оси не найден. Использeтся геометрический центр.")

        fallback_x = width // 2
        fallback_y = int(height * 0.90) 
        
        pivot_center = (fallback_x, fallback_y)

    return pivot_center

def find_pivot(risks):
    """
    Находит центр (ось) вращения шкалы, аппроксимируя окружность по центрам рисок.
    """
    if not risks:
        raise ValueError("Риски не найдены, невозможно определить центр.")
        
    risk_centers = np.array(
        [[int(x + w/2), int(y + h/2)] for x, y, w, h in risks], 
        dtype=np.float32
    )
    
    (cx, cy), radius = cv2.minEnclosingCircle(risk_centers)
    return (int(cx), int(cy))


def assign_values_to_risks(risks, digits, center):
    """
    Присваивает значения рискам, сортируя их по углу относительно центра.
    """
    if len(digits) < 2:
        raise RuntimeError("Найдено менее двух цифр, невозможно интерполировать шкалу.")

    cx, cy = center
    
    angular_risks = []
    for (x, y, w, h) in risks:
        rx = x + w/2
        ry = y + h/2
        angle = math.atan2(ry - cy, rx - cx)
        angular_risks.append(((x, y, w, h), angle))

    angular_risks.sort(key=lambda r: r[1])

    digit_mapping = []
    for digit_val, dx, dy in digits:
        min_dist = float('inf')
        best_risk_index = -1
        
        for i, (risk_bb, risk_angle) in enumerate(angular_risks):
            (x, y, w, h) = risk_bb
            rx = x + w/2
            ry = y + h/2
            dist = math.sqrt((dx - rx)**2 + (dy - ry)**2)
            
            if dist < min_dist and dist < 150:
                min_dist = dist
                best_risk_index = i
        
        if best_risk_index != -1:
            if best_risk_index not in [idx for val, idx in digit_mapping]:
                digit_mapping.append((digit_val, best_risk_index))

    digit_mapping.sort()

    if len(digit_mapping) < 2:
        raise RuntimeError(f"Не удалось сопоставить риски как минимум двум цифрам. Найдено сопоставлений: {len(digit_mapping)}")

    d1, first_index = digit_mapping[0]
    d2, second_index = digit_mapping[1]

    if first_index == second_index:
        raise RuntimeError("Две разные цифры сопоставлены одной и той же риске.")

    num_ticks_between = second_index - first_index
    value_span = d2 - d1
    value_per_tick = value_span / num_ticks_between

    print(f"Опорные точки: {d1} (индекс {first_index}) и {d2} (индекс {second_index})")
    print(f"Делений между ними: {num_ticks_between}, разница значений: {value_span}")
    print(f"Цена деления: {value_per_tick}")

    final_assigned_risks = [None] * len(angular_risks)
    
    current_val = d1
    for i in range(first_index, len(angular_risks)):
        (x, y, w, h), angle = angular_risks[i]
        final_assigned_risks[i] = (current_val, x, y, w, h, angle)
        current_val += value_per_tick

    current_val = d1
    for i in range(first_index - 1, -1, -1):
        current_val -= value_per_tick
        (x, y, w, h), angle = angular_risks[i]
        final_assigned_risks[i] = (current_val, x, y, w, h, angle)

    return final_assigned_risks

def find_needle_and_value(image_path, edges_input, center, center_mask, assigned_risks):
    cx1, cy1 = center_mask
    cx, cy = center
    
    edges = edges_input.copy()
    center_mask_radius = 75
    cv2.circle(edges, (cx1, cy1), center_mask_radius, 0, -1)

    hough_threshold = 50
    min_line_len = 40  
    max_line_gap = 10
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                            threshold=hough_threshold, 
                            minLineLength=min_line_len, 
                            maxLineGap=max_line_gap)

    if lines is None:
        raise RuntimeError("Стрелка не найдена после маскирования центра.")
    
    grouped_lines = group_similar_lines(lines, center, angle_threshold=10, distance_threshold=20)
    best_group = None
    max_group_score = -1.0

    for group in grouped_lines:
        group_score = evaluate_line_group(group, center)
        if group_score > max_group_score:
            max_group_score = group_score
            best_group = group

    if best_group is None:
        raise RuntimeError("Стрелка не найдена (ни одна группа линий не прошла фильтрацию).")

    best_line = create_average_line(best_group)
    tip = find_needle_tip(best_line, center)
    value = find_nearest_risk(assigned_risks, tip)

    return value, tip, best_line

def group_similar_lines(lines, center, angle_threshold=10, distance_threshold=20):
    """
    Группирует похожие линии по углу и расстоянию от центра.
    """
    groups = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist_to_center = np.abs((y2 - y1) * center[0] - (x2 - x1) * center[1] + x2 * y1 - y2 * x1) / length
        
        found_group = False
        for group in groups:
            group_angles = [l['angle'] for l in group]
            avg_group_angle = np.mean(group_angles)
            
            angle_diff = min(abs(angle - avg_group_angle), 180 - abs(angle - avg_group_angle))
            dist_diff = abs(dist_to_center - np.mean([l['dist'] for l in group]))
            
            if angle_diff < angle_threshold and dist_diff < distance_threshold:
                group.append({'line': line[0], 'angle': angle, 'dist': dist_to_center})
                found_group = True
                break
        
        if not found_group:
            groups.append([{'line': line[0], 'angle': angle, 'dist': dist_to_center}])
    
    return groups

def evaluate_line_group(group, center):
    """
    Оценивает группу линий как кандидата на стрелку.
    """
    cx, cy = center
    
    avg_line = create_average_line(group)
    x1, y1, x2, y2 = avg_line
    
    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    dist_to_center = np.abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / length
    
    dist1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
    dist2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
    
    line_count = len(group)
    
    score = (length * line_count) / (dist_to_center + 1.0)
    
    direction_ratio = max(dist1, dist2) / (min(dist1, dist2) + 1.0)
    if direction_ratio > 1.5:
        score *= direction_ratio
    
    return score

def create_average_line(group):
    """
    Создает усредненную линию из группы похожих линий.
    Возвращает кортеж (x1, y1, x2, y2)
    """
    all_points = []
    for line_data in group:
        x1, y1, x2, y2 = line_data['line']
        all_points.append((x1, y1))
        all_points.append((x2, y2))
    
    points_array = np.array(all_points)
    mean = np.mean(points_array, axis=0)
    
    centered_points = points_array - mean
    
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    main_direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    projections = np.dot(centered_points, main_direction)
    
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    
    start_point = mean + min_proj * main_direction
    end_point = mean + max_proj * main_direction
    
    return (int(start_point[0]), int(start_point[1]), 
            int(end_point[0]), int(end_point[1]))

def find_needle_tip(line, center):
    """
    Определяет кончик стрелки (точка, наиболее удаленная от центра).
    """
    x1, y1, x2, y2 = line
    cx, cy = center
    
    dist1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
    dist2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
    
    return (x1, y1) if dist1 > dist2 else (x2, y2)

def find_nearest_risk(assigned_risks, tip):
    """
    Определяет между какими рисками расположен tip и возвращает nearest current_val
    
    assigned_risks: список кортежей (current_val, x, y, w, h, angle)
    tip: кортеж (x, y)
    """
    
    def distance_to_risk(risk, point):
        """Расстояние от точки до центра риски"""
        _, risk_x, risk_y, _, _, _ = risk
        return math.sqrt((point[0] - risk_x)**2 + (point[1] - risk_y)**2)
    
    sorted_risks = sorted(assigned_risks, key=lambda risk: risk[1])
    
    min_dist = float('inf')
    nearest_risk = None
    second_nearest = None
    
    for risk in sorted_risks:
        dist = distance_to_risk(risk, tip)
        if dist < min_dist:
            second_nearest = nearest_risk
            nearest_risk = risk
            min_dist = dist
        elif second_nearest is None or dist < distance_to_risk(second_nearest, tip):
            second_nearest = risk
    
    if nearest_risk and second_nearest:
        if nearest_risk[1] < second_nearest[1]:
            left_risk, right_risk = nearest_risk, second_nearest
        else:
            left_risk, right_risk = second_nearest, nearest_risk
        
        tip_x = tip[0]
        if left_risk[1] <= tip_x <= right_risk[1]:
            dist_to_left = distance_to_risk(left_risk, tip)
            dist_to_right = distance_to_risk(right_risk, tip)
            
            if dist_to_left < dist_to_right:
                return left_risk[0]
            else:
                return right_risk[0]
    
    return nearest_risk[0] if nearest_risk else None

def process_image(uploaded_file: UploadFile, risk_threshold: int = 60):
    """
    Адаптированная функция обработки изображения для работы с FastAPI
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            contents = uploaded_file.file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        try:
            risks, edges, gray_image = find_risks(temp_file_path, risk_threshold)
            
            if not risks:
                raise ValueError("Риски не найдены")

            digits = find_digits(gray_image)
            if not digits:
                raise ValueError("Цифры не найдены")

            center_mask_radius = find_pivot(risks)
            center = find_pivot_point(temp_file_path)

            assigned_risks = assign_values_to_risks(risks, digits, center)
            value, tip, best_line = find_needle_and_value(
                temp_file_path, edges, center, center_mask_radius, assigned_risks
            )
            
            return {
                "value": float(value),
                "tip": tip,
                "best_line": best_line,
                "risks_found": len(risks),
                "digits_found": len(digits)
            }

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        raise
