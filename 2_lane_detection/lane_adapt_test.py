import cv2
import csv
import numpy as np

# CSV 파일에서 좌표를 로드하는 함수
def load_coordinates(csv_path):
    lanes = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lane_id = row['ID']
            x, y = int(row['X']), int(row['Y'])
            if lane_id not in lanes:
                lanes[lane_id] = []
            lanes[lane_id].append((x, y))
    return lanes

# 좌표를 사용하여 투명한 폴리곤 그리기
def draw_transparent_polygons(img, lanes, alpha=0.3):
    overlay = img.copy()
    for lane_id, coordinates in lanes.items():
        if len(coordinates) > 2:  # 폴리곤을 그리려면 최소 3개의 점이 필요합니다
            pts = np.array(coordinates, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))  # 녹색으로 채우기
            cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)  # 빨간색 테두리
    
    # 원본 이미지와 오버레이 이미지를 블렌딩
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
for i in range(11, 51):
    
    # 동영상 파일 로드
    cap = cv2.VideoCapture(f"D:/trajectory_project/CCTV_video/{i}.mp4")

    # CSV 파일에서 좌표 로드
    lanes = load_coordinates(f"D:/trajectory_project/roadline_csv/{i}.csv")

    # 영상 처리
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 영상의 각 프레임에 투명한 폴리곤 그리기
        draw_transparent_polygons(frame, lanes)

        cv2.imshow('Video with Transparent Polygons', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()