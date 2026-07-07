import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일에서 데이터 읽기
df = pd.read_csv('track_history_26.csv')

# 각 추적 ID별로 궤적을 그립니다.
fig, ax = plt.subplots()
for track_id, track_data in df.groupby('TrackID'):
    x_centers = track_data['X_center']
    y_centers = track_data['Y_center']
    ax.plot(x_centers, y_centers, label=f'Track ID {track_id}')

# 그래프에 레이블과 범례 추가
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Tracked Trajectories')
ax.legend()

# 그래프 표시
plt.show()