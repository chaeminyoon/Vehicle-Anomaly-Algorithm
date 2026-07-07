import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def extract_centerline(binary_road_image):
    # 도로의 골격(skeleton) 추출
    skeleton = cv2.ximgproc.thinning(binary_road_image)
    
    # 골격에서 중심선 좌표 추출
    y, x = np.where(skeleton > 0)
    
    # x, y 좌표를 정렬
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    
    return x, y

def calculate_curvature(x, y):
    # 스플라인 보간
    tck, u = splprep([x, y], s=0)
    
    # 더 조밀한 점들로 새로운 점 생성
    new_points = np.linspace(0, 1, num=1000)
    x_new, y_new = splev(new_points, tck)
    
    # 1차 미분
    dx = np.gradient(x_new)
    dy = np.gradient(y_new)
    
    # 2차 미분
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    # 곡률 계산
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
    
    return x_new, y_new, curvature

def process_road_image(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지 이진화 (실제로는 더 복잡한 전처리 필요)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 중심선 추출
    x, y = extract_centerline(binary)
    
    # 곡률 계산
    x_new, y_new, curvature = calculate_curvature(x, y)
    
    return img, x_new, y_new, curvature

def visualize_centerline(image, x, y):
    # 컬러 이미지로 변환
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 중심선 그리기
    for i in range(len(x) - 1):
        cv2.line(img_color, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), (0, 255, 0), 2)

    # 이미지 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Centerline Overlay on Road Image")
    plt.axis("off")
    plt.show()

# 사용 예
image_path = "D:/trajectory_project/photo/11.png"
image, x_new, y_new, curvature = process_road_image(image_path)

print(f"평균 곡률: {np.mean(curvature)}")
print(f"최대 곡률: {np.max(curvature)}")

# 중심선 시각화
visualize_centerline(image, x_new, y_new)
