import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def preprocess_yolo_trajectories(df, method='relative', seq_length=50, min_length=10):
    trajectories = []
    track_ids = []
    
    for track_id in df['TrackID'].unique():
        traj = df[df['TrackID'] == track_id].sort_values('Time')
        
        # 너무 짧은 궤적은 제외
        if len(traj) < min_length:
            continue
        
        # 시간을 숫자로 변환 (첫 시간을 0으로)
        start_time = datetime.strptime(traj['Time'].iloc[0], '%Y-%m-%d %H:%M:%S.%f')
        times = [(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') - start_time).total_seconds() for t in traj['Time']]
        
        coords = traj[['X_center', 'Y_center']].values
        
        if method == 'relative':
            # 상대적 좌표 사용
            start_x, start_y = coords[0]
            coords[:, 0] -= start_x
            coords[:, 1] -= start_y
        
        elif method == 'velocity':
            # 방향과 속도 정보 사용
            velocity = np.diff(coords, axis=0)
            direction = np.arctan2(velocity[:, 1], velocity[:, 0])
            speed = np.linalg.norm(velocity, axis=1)
            coords = np.column_stack((direction, speed))
            times = times[1:]  # velocity 계산으로 인해 첫 시간 제거
        
        # 시간 정보 추가
        traj_data = np.column_stack((times, coords))
        
        # 시퀀스 길이 맞추기
        if len(traj_data) >= seq_length:
            trajectories.append(traj_data[:seq_length])
        else:
            pad_width = ((0, seq_length - len(traj_data)), (0, 0))
            padded_traj = np.pad(traj_data, pad_width, mode='constant', constant_values=0)
            trajectories.append(padded_traj)
        
        track_ids.append(track_id)
    
    return np.array(trajectories), track_ids

def visualize_sample_trajectories(trajectories, track_ids, num_samples=50):
    plt.figure(figsize=(12, 8))
    for i in range(min(num_samples, len(trajectories))):
        traj = trajectories[i]
        if traj.shape[1] == 3:  # relative coordinates
            plt.plot(traj[:, 1], traj[:, 2], label=f'Vehicle {track_ids[i]}')
        elif traj.shape[1] == 3:  # velocity method
            cumulative_x = np.cumsum(np.cos(traj[:, 1]) * traj[:, 2])
            cumulative_y = np.cumsum(np.sin(traj[:, 1]) * traj[:, 2])
            plt.plot(cumulative_x, cumulative_y, label=f'Vehicle {track_ids[i]}')
    
    plt.xlabel('X coordinate / Cumulative X movement')
    plt.ylabel('Y coordinate / Cumulative Y movement')
    plt.title('Sample Vehicle Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()

# 메인 실행 코드
file_path = 'D:/trajectory_project/trajectory_csv/track_history_11.csv'  
df = pd.read_csv(file_path)

# 궤적 추출 (상대적 좌표 방법 사용)
trajectories, track_ids = preprocess_yolo_trajectories(df, method='relative')

print(f"추출된 궤적 수: {len(trajectories)}")
print(f"궤적 데이터 형태: {trajectories.shape}")
print(f"샘플 궤적 (처음 10개 프레임):\n{trajectories[0][:5]}")

# 샘플 궤적 시각화
visualize_sample_trajectories(trajectories, track_ids)