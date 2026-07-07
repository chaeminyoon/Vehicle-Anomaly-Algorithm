import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import pdist, squareform

# YOLO 모델 로드, GPU 사용 설정
model = YOLO("yolov8l.pt").to("cuda")

def calculate_congestion(distances, threshold=100):
    """
    거리 행렬을 기반으로 혼잡도를 계산합니다.
    threshold 이하의 거리를 가진 차량 쌍의 비율을 반환합니다.
    """
    if len(distances) == 0:
        return 0
    close_pairs = np.sum(distances < threshold)
    total_pairs = len(distances) * (len(distances) - 1) / 2
    return close_pairs / total_pairs if total_pairs > 0 else 0

for i in range(11, 51):
    if i == 23:
        continue

    cap = cv2.VideoCapture(f"H:/새 폴더/trajectory_project/video1/{i}.mp4")
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define line points
    line_points = [(0, h // 2), (w, h // 2)]

    # Video writer
    video_writer = cv2.VideoWriter(f"congestion_analysis_output_{i}.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    vehicle_count = 0
    detected_ids = set()

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"Video {i} frame is empty or video processing has been successfully completed.")
            break

        # Perform tracking on GPU
        results = model.track(im0, persist=True, show=False)

        tracks = results[0].boxes  # Retrieve tracked objects

        centers = []
        for box in tracks:
            bbox = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)  # Calculate the center of the bounding box
            centers.append(center)

            # Check if the object has already been counted to avoid double counting
            track_id = box.id[0].item()  # Unique identifier for the tracked object
            if track_id not in detected_ids and line_points[0][1] - 5 < center[1] < line_points[0][1] + 5:
                vehicle_count += 1
                detected_ids.add(track_id)

            # Draw the bounding box
            cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # Calculate distances between all pairs of vehicles
        if len(centers) > 1:
            distances = pdist(centers)
            congestion_level = calculate_congestion(distances)
        else:
            congestion_level = 0

        # Draw the counting line
        cv2.line(im0, line_points[0], line_points[1], (0, 255, 0), 2)

        # Display vehicle count and congestion level on the frame
        cv2.putText(im0, f"Vehicle Count: {vehicle_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Write the frame to the video writer
        video_writer.write(im0)

        # Show the frame
        cv2.imshow(f"Congestion Analysis - Video {i}", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Final Vehicle Count for Video {i}: {vehicle_count}")
