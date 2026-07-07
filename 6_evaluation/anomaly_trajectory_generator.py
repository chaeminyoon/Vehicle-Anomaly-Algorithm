import cv2
import pygame
import numpy as np
import csv
import sys
import os

def process_video(video_path, csv_path):
    # 초기화
    pygame.init()

    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pygame 화면 설정
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"영상 위 궤적 그리기 - {os.path.basename(video_path)}")

    # 색상 정의
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)

    # 폰트 설정
    font = pygame.font.Font(None, 36)

    # 궤적 데이터 저장
    trajectories = {}
    current_track_id = 1

    # CSV 파일 생성
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['TrackID', 'X_center', 'Y_center'])

    def draw_text(text, x, y):
        surface = font.render(text, True, WHITE)
        screen.blit(surface, (x, y))

    def save_trajectories():
        for track_id, points in trajectories.items():
            for x, y in points:
                csv_writer.writerow([track_id, x, y])
        print(f"궤적이 {csv_path}에 저장되었습니다.")

    clock = pygame.time.Clock()
    running = True
    drawing = False
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 왼쪽 마우스 버튼
                    drawing = True
                    trajectories[current_track_id] = []
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    current_track_id += 1
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    x, y = event.pos
                    trajectories[current_track_id].append((x, y))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_trajectories()
                elif event.key == pygame.K_c:
                    trajectories.clear()
                    current_track_id = 1
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_q:
                    running = False

        if not paused:
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오 루프
                continue

            # 프레임 좌우 반전
            frame = cv2.flip(frame, 1)
            
            # OpenCV BGR to Pygame RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0, 0))

        # 궤적 그리기
        for track_id, points in trajectories.items():
            if len(points) > 1:
                pygame.draw.lines(screen, RED, False, points, 2)

        # 사용법 표시
        draw_text("press left button to make trajectory, S: save, C: delete, Spacebar: pause, Q: exit", 10, 10)

        pygame.display.flip()
        clock.tick(fps)

    # 정리
    csv_file.close()
    video.release()
    pygame.quit()

# 메인 실행 부분
if __name__ == "__main__":
    base_path = "D:/새 폴더/trajectory_project/video1/"
    for i in range(12, 52):  # 11부터 51까지
        video_path = f"{base_path}{i}.mp4"
        csv_path = f"D:/새 폴더/trajectory_project/기능구현/이상궤적/trajectories_{i}.csv"
        
        if os.path.exists(video_path):
            print(f"Processing video: {video_path}")
            process_video(video_path, csv_path)
        else:
            print(f"Video file not found: {video_path}")

    print("모든 비디오 처리 완료")
    sys.exit()

