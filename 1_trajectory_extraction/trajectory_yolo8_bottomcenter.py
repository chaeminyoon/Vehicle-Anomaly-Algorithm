# 개선 실험 ⑦용 재추출: 바운딩박스 중심(x, y)과 하단 중앙점(x, y+h/2)을 동시 저장
# - 호모그래피(원근 보정)는 "도로 평면 위의 점"에만 유효한 변환인데, 박스 중심은
#   차량 높이만큼 평면에서 떠 있어 보정 시 계통 오차가 생김 -> 바퀴 접지점에 가까운
#   하단 중앙점이 물리적으로 올바른 입력
# - 같은 추적 실행에서 두 좌표를 함께 저장하므로 좌표 선택만 다른 짝비교(paired)가 가능
# - 로컬 재추출용: 경로/모델/디바이스/프레임 수를 인자로 지정 (원 실험은 서버 CUDA 환경)
import argparse
import datetime
import os

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

CLASS_ID = [2, 5, 7]  # car, bus, truck


def pick_device(arg):
    if arg != 'auto':
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def process_video(video_path, out_csv, model_path, device, max_frames=None):
    # 영상마다 모델을 새로 로드해 추적기 상태(TrackID)를 초기화
    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    frame_count = 0
    while cap.isOpened():
        if max_frames is not None and frame_count >= max_frames:
            break
        success, frame = cap.read()
        if not success:
            break
        timestamp = datetime.timedelta(seconds=frame_count / fps)
        results = model.track(frame, persist=True, classes=CLASS_ID, verbose=False)
        b = results[0].boxes
        if b is not None and len(b) > 0 and b.id is not None:
            for box, tid in zip(b.xywh.cpu(), b.id.int().cpu().tolist()):
                x, y, w, h = (float(v) for v in box)
                rows.append(dict(TrackID=tid, Time=str(timestamp),
                                 X_center=x, Y_center=y,
                                 X_bottom=x, Y_bottom=y + h / 2))
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"  {os.path.basename(video_path)}: {frame_count}/{n_total} 프레임", flush=True)
    cap.release()

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"저장: {out_csv} ({len(rows)} 레코드, {frame_count} 프레임)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', type=int, nargs='+', default=[11, 14, 17])
    ap.add_argument('--video-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model', default='yolov8n.pt')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-frames', type=int, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}, model: {args.model}")
    os.makedirs(args.out_dir, exist_ok=True)
    for i in args.videos:
        process_video(os.path.join(args.video_dir, f"{i}.mp4"),
                      os.path.join(args.out_dir, f"track_history_bc_{i}.csv"),
                      args.model, device, args.max_frames)
    print("EXTRACT_OK")
