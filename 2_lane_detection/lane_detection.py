import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def parse_trajectory_data(file_path):
    trajectories = defaultdict(list)
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            try:
                track_id = int(row[0])
                timestamp = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S.%f')
                x, y = float(row[2]), float(row[3])
                trajectories[track_id].append((x, y))
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid row: {row}. Error: {e}")
    
    valid_trajectories = [np.array(traj) for traj in trajectories.values() if len(traj) > 1]
    print(f"총 {len(valid_trajectories)}개의 유효한 궤적이 발견되었습니다.")
    return valid_trajectories

def calculate_lateral_density(trajectories, num_bins=200):
    all_points = np.concatenate(trajectories)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    # 이미지 좌표계에 맞춰 y축 반전
    hist, x_edges, y_edges = np.histogram2d(
        all_points[:, 0], -all_points[:, 1] + y_max + y_min, 
        bins=(num_bins, num_bins), range=[[x_min, x_max], [y_min, y_max]]
    )
    
    lateral_density = np.sum(hist, axis=1)
    return lateral_density, x_edges

def find_lane_positions(lateral_density, x_edges, n_lanes=7, smoothing_factor=5):
    smoothed_density = gaussian_filter1d(lateral_density, smoothing_factor)
    peaks, _ = find_peaks(smoothed_density, distance=len(smoothed_density)/(n_lanes*2))
    
    # n_lanes 개수만큼 가장 높은 피크 선택
    sorted_peaks = sorted(peaks, key=lambda i: smoothed_density[i], reverse=True)[:n_lanes]
    sorted_peaks.sort()  # 위치 순으로 다시 정렬
    
    lane_positions = x_edges[sorted_peaks]
    return lane_positions, smoothed_density

def calculate_lane_directions(trajectories, lane_positions, distance_threshold=50):
    lane_points = [[] for _ in lane_positions]
    
    for trajectory in trajectories:
        for point in trajectory:
            distances = np.abs(lane_positions - point[0])
            nearest_lane = np.argmin(distances)
            if distances[nearest_lane] < distance_threshold:
                lane_points[nearest_lane].append(point)
    
    lane_directions = []
    for points in lane_points:
        if len(points) > 1:
            points = np.array(points)
            dx = points[-1, 0] - points[0, 0]
            dy = points[-1, 1] - points[0, 1]
            direction = np.arctan2(dy, dx)
            lane_directions.append(direction)
        else:
            lane_directions.append(0)  # 기본 방향 (충분한 포인트가 없는 경우)
    
    return lane_directions

def visualize_lanes(trajectories, lane_positions, lane_directions, lateral_density, output_path):
    plt.figure(figsize=(15, 10))
    
    # 궤적 그리기
    for traj in trajectories:
        plt.plot(traj[:, 0], -traj[:, 1], 'b-', alpha=0.1)
    
    # 차선 위치 및 방향 표시
    for i, (pos, direction) in enumerate(zip(lane_positions, lane_directions)):
        plt.arrow(pos, 0, 100*np.cos(direction), -100*np.sin(direction), 
                  color='r', width=2, head_width=20, head_length=30, alpha=0.7)
        plt.text(pos, 50, f'Lane {i+1}', fontsize=12, color='red', ha='center')
    
    # 횡방향 밀도 그래프
    density_scale = -np.max(lateral_density) / 5  # 스케일 조정
    plt.plot(np.linspace(np.min(trajectories[0][:, 0]), np.max(trajectories[0][:, 0]), len(lateral_density)),
             lateral_density * density_scale, 'g-', linewidth=2, alpha=0.7)
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Detected Lanes from Vehicle Trajectories')
    plt.gca().invert_yaxis()
    plt.savefig(output_path)
    plt.close()

def detect_and_visualize_lanes(input_file, output_file='detected_lanes.png', n_lanes=7, smoothing_factor=5):
    try:
        input_file = input_file.strip('"')
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")
        
        trajectories = parse_trajectory_data(input_file)
        if not trajectories:
            raise ValueError("유효한 궤적 데이터가 없습니다.")
        
        lateral_density, x_edges = calculate_lateral_density(trajectories)
        lane_positions, smoothed_density = find_lane_positions(lateral_density, x_edges, n_lanes, smoothing_factor)
        lane_directions = calculate_lane_directions(trajectories, lane_positions)
        
        print(f"감지된 차선 수: {len(lane_positions)}")
        print("차선 위치:", lane_positions)
        print("차선 방향 (라디안):", lane_directions)
        
        visualize_lanes(trajectories, lane_positions, lane_directions, smoothed_density, output_file)
        print(f"시각화 결과가 {output_file}에 저장되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    input_file_path = 'D:/trajectory_project/trajectory_results/track_history_17.csv'
    output_file_path = 'D:/trajectory_project/기능구현/17_detected_lanes.png'

    detect_and_visualize_lanes(
        input_file_path,
        output_file=output_file_path,
        n_lanes=7,
        smoothing_factor=5
    )