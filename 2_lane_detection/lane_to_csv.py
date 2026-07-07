import cv2
import numpy as np
import csv
import os

# 마우스 콜백 함수
def draw_polygon(event, x, y, flags, param):
    global points, img, polygons, current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        if len(current_polygon) > 1:
            cv2.line(img, current_polygon[-2], (x, y), (0, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Video', img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 폴리곤 완성 (시작점과 연결)
        if len(current_polygon) > 2:
            cv2.line(img, current_polygon[-1], current_polygon[0], (0, 255, 0), 2)
            polygons.append(current_polygon.copy())
            cv2.fillPoly(img, [np.array(current_polygon)], (0, 255, 0, 64))  # 반투명한 채우기
            current_polygon.clear()
            cv2.imshow('Video', img)

# 키 이벤트 처리
def handle_key(key):
    global current_polygon, img
    if key == 26:  # Ctrl+Z
        if len(current_polygon) > 0:
            current_polygon.pop()
            img = frame.copy()
            if len(current_polygon) > 1:
                for i in range(len(current_polygon) - 1):
                    cv2.line(img, current_polygon[i], current_polygon[i+1], (0, 255, 0), 2)
                for point in current_polygon:
                    cv2.circle(img, point, 3, (0, 0, 255), -1)
            cv2.imshow('Video', img)

# 좌표 저장
def save_coordinates_to_csv(polygons, video_name):
    csv_file = video_name + '.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'X', 'Y'])
        for idx, polygon in enumerate(polygons, 1):
            for point in polygon:
                writer.writerow([f'polygon{idx}', f"{point[0]}", f"{point[1]}"])

# 메인 루프
for i in range(21, 22):
    video_path = f'{i}.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    polygons = []
    current_polygon = []

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', draw_polygon)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()
        
        # 완성된 폴리곤 그리기
        for polygon in polygons:
            cv2.polylines(img, [np.array(polygon)], True, (0, 255, 0), 2)
            cv2.fillPoly(img, [np.array(polygon)], (0, 255, 0, 64))  # 반투명한 채우기

        # 현재 그리는 폴리곤 그리기
        if len(current_polygon) > 1:
            cv2.polylines(img, [np.array(current_polygon)], False, (0, 255, 0), 2)
            for point in current_polygon:
                cv2.circle(img, point, 3, (0, 0, 255), -1)

        cv2.imshow('Video', img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        handle_key(key)

    # 마무리 작업
    cap.release()
    cv2.destroyAllWindows()

    # 좌표를 CSV 파일로 저장
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    save_coordinates_to_csv(polygons, video_basename)

print("All videos processed.")