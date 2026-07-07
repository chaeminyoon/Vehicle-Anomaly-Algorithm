import pandas as pd
import numpy as np
from datetime import datetime

# 픽셀을 km로 변환하는 함수
def pixel_to_km(pixels):
    # 예: 100 픽셀 = 10 미터라고 가정
    # 이 비율은 실제 CCTV 설정에 따라 조정해야 합니다
    return pixels * (10 / 100) / 1000  # km로 변환

def calculate_average_speed(data):
    df = pd.DataFrame(data, columns=['TrackID', 'Time', 'X', 'Y'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    grouped = df.groupby('TrackID')
    
    vehicle_speeds = []
    
    for track_id, group in grouped:
        group = group.sort_values('Time')
        
        # 픽셀 거리를 km로 변환
        distances = np.sqrt(np.diff(group['X'])**2 + np.diff(group['Y'])**2)
        distances_km = pixel_to_km(distances)
        
        time_diffs = group['Time'].diff().dt.total_seconds()
        
        # km/s로 속도 계산
        speeds = distances_km / time_diffs.iloc[1:]
        
        avg_speed = np.mean(speeds)
        vehicle_speeds.append(avg_speed)
    
    overall_avg_speed = np.mean(vehicle_speeds)
    
    return overall_avg_speed, dict(zip(grouped.groups.keys(), vehicle_speeds))

# 테스트를 위한 샘플 데이터
sample_data = [
    ('1', '0:00:00', 384.0718079, 407.9208069),
    ('1', '0:00:01', 385.8363953, 405.239624),
    ('1', '0:00:02', 387.4332275, 402.003418),
    ('2', '0:00:00', 319.208252, 227.2042542),
    ('2', '0:00:01', 317.7050781, 228.7080231),
    ('2', '0:00:02', 316.752594, 229.6623535)
]

overall_speed, vehicle_speeds = calculate_average_speed(sample_data)

print(f"전체 평균 속도: {overall_speed:.6f} km/s")
print("차량별 평균 속도:")
for vehicle, speed in vehicle_speeds.items():
    print(f"차량 {vehicle}: {speed:.6f} km/s")

# km/h로 변환하여 출력
print(f"\n전체 평균 속도: {overall_speed * 3600:.2f} km/h")
print("차량별 평균 속도:")
for vehicle, speed in vehicle_speeds.items():
    print(f"차량 {vehicle}: {speed * 3600:.2f} km/h")