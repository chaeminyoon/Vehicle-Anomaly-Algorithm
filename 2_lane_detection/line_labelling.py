import cv2
import numpy as np
import csv
import os
from shapely.geometry import Polygon, Point

# 마우스 콜백 함수
def draw_polygon(event, x, y, flags, param):
    global points, img, polygons, current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        if len(current_polygon) > 1:
            cv2.line(img, current_polygon[-2], current_polygon[-1], (0, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Video', img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) > 2:
            cv2.line(img, current_polygon[-1], current_polygon[0], (0, 255, 0), 2)
            polygons.append(current_polygon.copy())
            current_polygon.clear()
        cv2.imshow('Video', img)

# 키 이벤트 처리
def handle_key(key):
    global current_polygon, img
    if key == 26:  # Ctrl+Z
        if len(current_polygon) > 0:
            current_polygon.pop()
            img = frame.copy()
            for polygon in polygons:
                cv2.polylines(img, [np.array(polygon)], True, (0, 255, 0), 2)
            if len(current_polygon) > 1:
                cv2.polylines(img, [np.array(current_polygon)], False, (0, 255, 0), 2)
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
                writer.writerow([f'lane{idx}', f"{point[0]}", f"{point[1]}"])

# 차선 변경 감지 함수
def detect_lane_changes(trajectory, lane_polygons):
    current_lane = None
    lane_changes = 0
    for point in trajectory:
        point = Point(point)
        for lane_index, lane in enumerate(lane_polygons):
            if Polygon(lane).contains(point):
                if current_lane is not None and lane_index != current_lane:
                    lane_changes += 1
                current_lane = lane_index
                break
    return lane_changes

# 메인 프로세스
for i in range(30, 36):
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
       
        # 모든 polygon 그리기
        for polygon in polygons:
            cv2.polylines(img, [np.array(polygon)], True, (0, 255, 0), 2)
        
        # 현재 그리는 polygon 그리기
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

    # 예시: 차선 변경 감지 (실제 궤적 데이터로 대체 필요)
    sample_trajectory = [(100, 100), (150, 150), (200, 200), (250, 250)]  # 예시 궤적
    lane_changes = detect_lane_changes(sample_trajectory, polygons)
    print(f"Video {i}: Detected {lane_changes} lane changes")

print("Processing complete. Polygon data saved to CSV files.")