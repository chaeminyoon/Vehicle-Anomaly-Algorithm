import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_single_file(file_path):
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 기본 정보 출력
    print("데이터 형태:", df.shape)
    print("\n첫 몇 행:")
    print(df.head())
    print("\n데이터 정보:")
    df.info()
    print("\n기술 통계:")
    print(df.describe())
    
    # 결측치 확인
    print("\n결측치:")
    print(df.isnull().sum())
    
    # 시간 데이터 변환
    df['Time'] = pd.to_datetime(df['Time'])
    
    # TrackID별 데이터 포인트 수 확인
    track_counts = df['TrackID'].value_counts()
    print("\nTrackID별 데이터 포인트 수:")
    print(track_counts.describe())
    
    # 시각화: TrackID별 데이터 포인트 수 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(track_counts, bins=50)
    plt.title('Distribution of Data Points per TrackID')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Frequency')
    plt.show()
    
    # 시각화: X_center와 Y_center의 산점도
    plt.figure(figsize=(10, 6))
    plt.scatter(df['X_center'], df['Y_center'], alpha=0.1)
    plt.title('Scatter Plot of Vehicle Positions')
    plt.xlabel('X_center')
    plt.ylabel('Y_center')
    plt.show()
    
    return df

# 단일 파일 분석
file_path = 'D:/trajectory_project/trajectory_csv/track_history_11.csv'  # 실제 파일 경로로 수정 필요
df = load_and_analyze_single_file(file_path)

# 추가 전처리 단계 (필요에 따라)
# 1. 이상치 제거
# 2. 시간 순서로 정렬
# 3. 연속된 시간 간격 확인
# 4. 필요한 경우 데이터 정규화 또는 스케일링

# 예시: 시간 순서로 정렬
df_sorted = df.sort_values(['TrackID', 'Time'])

# 예시: TrackID별 시간 간격 확인
df_sorted['time_diff'] = df_sorted.groupby('TrackID')['Time'].diff()
print("\n시간 간격 통계:")
print(df_sorted['time_diff'].describe())

# 필요한 경우 추가 전처리 단계 구현