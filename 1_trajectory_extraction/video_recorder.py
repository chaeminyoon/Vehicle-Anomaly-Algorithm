import cv2
import datetime
import os

for CN in range(11, 51):
    # RTSP 스트림 URL
    rtsp_url = f"rtsp://210.99.70.120:1935/live/cctv0{CN}.stream"

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(rtsp_url)

    # 시작 시간 기록
    start_time = datetime.datetime.now()

    # 파일 이름 및 경로 설정
    currentTime = datetime.datetime.now()
    fileName = currentTime.strftime('%Y-%m-%d %H-%M-%S')
    path = os.path.join(f'../video/{fileName}_{CN}.mp4')  # 경로 수정

    # 첫 번째 프레임 읽기 (해상도 확인을 위해)
    ret, frame = cap.read()
    if ret:
        height, width, layers = frame.shape
    else:
        print("Error: Cannot read frame from stream")
        cap.release()
        continue  # 에러 시 다음 카메라로 넘어갑니다.

    # 비디오 저장을 위한 VideoWriter 객체 생성 (코덱 및 해상도 수정)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (width, height))

    while ret:
        # 현재 시간과 시작 시간의 차이 계산
        current_time = datetime.datetime.now()
        elapsed_time = current_time - start_time

        # 5분(300초)이 경과했는지 확인
        if elapsed_time.total_seconds() > 300:
            break

        out.write(frame)  # 읽은 프레임을 비디오 파일에 쓰기

        ret, frame = cap.read()

    # 자원 해제
    cap.release()
    out.release()

cv2.destroyAllWindows()
