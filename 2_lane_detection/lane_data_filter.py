import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def calculate_ratio(group):
    if len(group) < 2:
        return float('inf')
    
    # 총 궤적 길이 계산
    total_length = np.sum(np.sqrt(np.diff(group['X'])**2 + np.diff(group['Y'])**2))
    
    # 시작점과 끝점 사이의 직선 거리 계산
    straight_distance = np.sqrt((group['X'].iloc[-1] - group['X'].iloc[0])**2 + 
                                (group['Y'].iloc[-1] - group['Y'].iloc[0])**2)
    
    # 비율 계산 (0으로 나누는 것을 방지)
    ratio = total_length / straight_distance if straight_distance > 0 else float('inf')
    
    return ratio

def filter_stationary(group, ratio_threshold=1.5):
    ratio = calculate_ratio(group)
    
    # 비율이 임계값보다 작으면 그룹 반환, 아니면 빈 DataFrame 반환
    return group if ratio < ratio_threshold else pd.DataFrame()

def process_file(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        filtered_df = df.groupby('TrackID').apply(filter_stationary).reset_index(drop=True)
        
        # 빈 DataFrame이 아닌 경우만 저장
        if not filtered_df.empty:
            filtered_df.to_csv(output_path, index=False)
            print(f"원본 행 수: {len(df)}, 필터링 후 행 수: {len(filtered_df)}")
        else:
            print(f"필터링 후 모든 데이터가 제거되었습니다: {input_path}")
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    input_dir = "D:/trajectory_project/video1/trajectory"
    output_dir = "D:/trajectory_project/video1/filtered_trajectory3"
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 처리할 파일 목록 생성
    files_to_process = [f"track_history_{i}.csv" for i in range(24, 51)]
    
    # tqdm을 사용하여 진행 상황 표시
    for filename in tqdm(files_to_process, desc="Processing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"filtered_{filename}")
        
        if os.path.exists(input_path):
            if process_file(input_path, output_path):
                print(f"처리가 완료되었습니다. {filename} 저장되었습니다.")
            else:
                print(f"파일 처리 실패: {filename}")
        else:
            print(f"파일이 존재하지 않습니다: {input_path}")

if __name__ == "__main__":
    main()