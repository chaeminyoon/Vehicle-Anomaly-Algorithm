import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 데이터 정리
data = {
    'Location': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    'Actual': [9, 8, 9, 7, 7, 9, 6, 5, 7, 7, None, None, 6, 7, 7, 6, 11, 9, 8, 7, 8, 7, 7, 6, 5, 7, 7, 10, None, None, 8, 4, 9, 11, 4, 5, 11, 4, 8],
    'Filtered': [9, 7, 4, 6, 5, 7, 6, 3, 9, 9, 5, 5, 6, 8, 9, 7, 7, 5, 5, 7, 8, 9, 5, 8, 5, 7, 4, 7, 6, 11, 8, 8, 5, 7, 5, 6, 6, 8, 6],
    'Original': [8, 9, 8, 5, 9, 9, 7, 8, 8, 6, 5, 5, 6, 5, 9, 7, 8, 6, 4, 10, 9, 9, 6, 9, 9, 6, 5, 5, 6, 11, 8, 6, 6, 8, 6, 9, 8, 9, 7],
    'New': [8, 9, 8, 8, 10, 11, 6, 9, 7, 8, 9, None, 6, 7, 9, 10, 9, None, 10, 6, 9, 6, 8, 7, 6, 6, 11, 12, 4, None, 10, 9, 10, 9, 8, 5, 9, 9, 8]
}

df = pd.DataFrame(data)
df = df.dropna()  # None 값 제거

# 정확도 계산 함수
def calculate_accuracy(actual, predicted):
    exact_match = np.mean(actual == predicted) * 100
    within_one = np.mean(np.abs(actual - predicted) <= 1) * 100
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return exact_match, within_one, mae, rmse

# Filtered, Original, New에 대한 정확도 계산
filtered_accuracy = calculate_accuracy(df['Actual'], df['Filtered'])
original_accuracy = calculate_accuracy(df['Actual'], df['Original'])
new_accuracy = calculate_accuracy(df['Actual'], df['New'])

# 결과 출력
for name, accuracy in [("Filtered", filtered_accuracy), ("Original", original_accuracy), ("New", new_accuracy)]:
    print(f"\n{name} 추정 정확도:")
    print(f"정확히 일치: {accuracy[0]:.2f}%")
    print(f"±1 이내: {accuracy[1]:.2f}%")
    print(f"평균 절대 오차 (MAE): {accuracy[2]:.2f}")
    print(f"평균 제곱근 오차 (RMSE): {accuracy[3]:.2f}")

# 시각화
plt.figure(figsize=(12, 6))
plt.scatter(df['Location'], df['Actual'], label='Actual', color='blue')
plt.scatter(df['Location'], df['Filtered'], label='Filtered', color='red', marker='x')
plt.scatter(df['Location'], df['Original'], label='Original', color='green', marker='+')
plt.scatter(df['Location'], df['New'], label='New', color='purple', marker='*')
plt.xlabel('Location')
plt.ylabel('Lane Count')
plt.title('Actual vs Estimated Lane Counts')
plt.legend()
plt.grid(True)
plt.show()