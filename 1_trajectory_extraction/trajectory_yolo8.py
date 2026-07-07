import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import datetime
import os
import time
from multiprocessing import Pool
import torch

# 차량, 버스, 트럭만 식별 (클래스 ID: 2, 3, 5, 7)
CLASS_ID = [2, 5, 7]

def process_video(i):
    try:
        # GPU 사용 가능 여부 확인 (각 프로세스마다 개별적으로 확인)
        device = torch.device(f"cuda:{i % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
        print(f"Processing video {i} using device: {device}")

        # Load the YOLOv8 model (각 프로세스마다 개별적으로 로드)
        model = YOLO('yolov8l.pt')
        model.to(device)

        video_path = f'/home/cmyoon/personal/trajectory_project/video1/{i}.mp4'
        output_path = f'/home/cmyoon/personal/trajectory_project/video1/trajectory1/track_history_{i}.csv'
        temp_output_path = f'/home/cmyoon/personal/trajectory_project/video1/trajectory1/temp_track_history_{i}.csv'

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / fps
        frame_skip = 1  # 3프레임마다 1프레임만 처리

        track_history = {}
        track_data = []
        frame_count = 0
        last_save_time = time.time()
        save_interval = 200 

        while cap.isOpened():
            for _ in range(frame_skip):
                success = cap.grab()
                if not success:
                    break
            success, frame = cap.retrieve()
            if not success:
                break

            current_time = frame_count * frame_time * frame_skip
            timestamp = datetime.timedelta(seconds=current_time)

            results = model.track(frame, persist=True, classes=CLASS_ID)

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywh.cpu()
                class_ids = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

                for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                    if int(class_id) in CLASS_ID:
                        x, y, w, h = box
                        track = track_history.get(track_id, [])
                        track.append((float(x), float(y)))
                        track_history[track_id] = track[-10:]  # 최대 10개의 포인트만 유지

                        track_data.append({'TrackID': track_id, 'Time': str(timestamp), 'X': float(x), 'Y': float(y)})

            frame_count += 1

            # 시간 기반 중간 저장
            if time.time() - last_save_time > save_interval:
                temp_df = pd.DataFrame(track_data)
                temp_df.to_csv(temp_output_path, index=False)
                print(f"Temporary data saved for video {i} at frame {frame_count}")
                last_save_time = time.time()

        cap.release()

        # 최종 데이터 저장
        df = pd.DataFrame(track_data)
        df.to_csv(output_path, index=False)
        print(f"Final data saved for video {i}")

        # 임시 파일 삭제
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

    except Exception as e:
        print(f"Error processing video {i}: {str(e)}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    video_numbers = range(11, 51)
    num_processes = min(5, torch.cuda.device_count())  # GPU 수에 따라 프로세스 수 조정
    
    with Pool(num_processes) as p:
        p.map(process_video, video_numbers)

    print("All videos processed.")