import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    required_columns = ['TrackID', 'Time', 'X', 'Y']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 파일에 필요한 열이 없습니다. 필요한 열: {required_columns}")
    return df[required_columns]

def preprocess_data(df):
    df = df.dropna(subset=['X', 'Y'])
    df = df[(~df['X'].isin([np.inf, -np.inf])) & (~df['Y'].isin([np.inf, -np.inf]))]
    
    x_range = df['X'].max() - df['X'].min()
    y_range = df['Y'].max() - df['Y'].min()
    
    if x_range == 0 or y_range == 0:
        raise ValueError("X 또는 Y의 범위가 0입니다. 데이터를 확인해주세요.")
    
    df['X'] = (df['X'] - df['X'].min()) / x_range * 100
    df['Y'] = (df['Y'] - df['Y'].min()) / y_range * 100
    df['Y'] = 100 - df['Y']
    return df

def estimate_road_count(df, y_value, y_tolerance=1, smooth_sigma=1, distance=1):
    filtered_df = df[(df['Y'] >= y_value - y_tolerance) & (df['Y'] <= y_value + y_tolerance)]
    
    hist, x_edges = np.histogram(filtered_df['X'], bins=100)
    smoothed_hist = gaussian_filter(hist, sigma=smooth_sigma)
    
    # 모든 첨두 찾기 (최소 거리만 설정)
    peaks, _ = find_peaks(smoothed_hist, distance=distance)
    
    # 피크의 너비 계산
    widths, _, _, _ = peak_widths(smoothed_hist, peaks, rel_height=0.5)

    return len(peaks), peaks, widths, smoothed_hist, x_edges

def plot_results(df, y_values, road_counts, peaks_list, widths_list, hist_list, x_edges_list, file_name='result'):
    fig, axs = plt.subplots(len(y_values), 1, figsize=(12, 5*len(y_values)), sharex=True)

    plt.subplots_adjust(left=0.059, bottom=0.064, right=0.988, top=0.924, wspace=0.198, hspace=0.295)

    for i, (y_value, road_count, peaks, widths, hist, x_edges) in enumerate(zip(y_values, road_counts, peaks_list, widths_list, hist_list, x_edges_list)):
        axs[i].plot(x_edges[:-1], hist, label='Smoothed Histogram')
        axs[i].plot(x_edges[:-1][peaks], hist[peaks], "x", color="red")
        for peak, width in zip(peaks, widths):
            axs[i].axvspan(x_edges[int(peak-width/2)], x_edges[int(peak+width/2)], alpha=0.2, color='red')
        axs[i].set_title(f'Y={y_value} (Estimated lanes: {road_count})')
        axs[i].set_ylabel('Density')
        axs[i].legend()

    axs[-1].set_xlabel('X')
    plt.tight_layout()
    
    # Save the figure
    save_dir = 'D:/trajectory_project/video1/results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{file_name}_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    y_values = [20, 50, 70, 90]
    
    for i in range(11, 51):
        file_name = f'filtered_track_history_{i}'
        file_path = f'D:/trajectory_project/video1/filtered_trajectory3/{file_name}.csv'
        
        try:
            data = load_data(file_path)
            preprocessed_data = preprocess_data(data)
            
            road_counts = []
            peaks_list = []
            widths_list = []
            hist_list = []
            x_edges_list = []
            
            for y_value in y_values:
                road_count, peaks, widths, hist, x_edges = estimate_road_count(preprocessed_data, y_value)
                road_counts.append(road_count)
                peaks_list.append(peaks)
                widths_list.append(widths)
                hist_list.append(hist)
                x_edges_list.append(x_edges)
            
            max_road_count = max(road_counts)
            max_road_count_y = y_values[road_counts.index(max_road_count)]
            
            print(f"File: filtered_track_history_{i}.csv")
            print(f"최대 추정 차선 수: {max_road_count} (Y={max_road_count_y})")
            print("각 Y 값에서의 추정 차선 수:")
            for y, count in zip(y_values, road_counts):
                print(f"  Y={y}: {count}")
            print()
            
            plot_results(preprocessed_data, y_values, road_counts, peaks_list, widths_list, hist_list, x_edges_list, file_name)
        
        except Exception as e:
            print(f"오류 발생 (filtered_track_history_{i}.csv): {str(e)}")
            print()