import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# 데이터 로드
data = """\
지점수\t차선수\t평균속도\t통행량
11\t8\t22.34\t780
12\t8\t11.57\t31
13\t9\t27.87\t416
14\t8\t30.63\t362
15\t7\t32.95\t380
16\t9\t15.5\t439
17\t9\t11.9\t110
19\t7\t14.95\t337
20\t5\t35.49\t586
24\t8\t19.26\t431
25\t6\t12.93\t227
26\t8\t4.36\t211
27\t5\t16.26\t137
28\t10\t16.73\t315
31\t7\t18.45\t591
32\t5\t10.86\t228
33\t7\t27.19\t222
34\t9\t11.62\t700
35\t9\t28.5\t259
36\t5\t6.62\t585
37\t4\t36.83\t172
38\t6\t22.05\t474
39\t8\t19.73\t563
42\t9\t27.86\t408
43\t8\t9.55\t54
44\t8\t12.22\t83
45\t10\t13.42\t455
46\t10\t35.61\t87
47\t4\t29.98\t107
48\t11\t24.01\t634
49\t4\t17.8\t172
50\t8\t33.53\t640
"""
# 데이터프레임으로 변환
df = pd.read_csv(StringIO(data), sep='\t')

# 데이터 확인
print(df)

# 클러스터링에 사용할 피처 선택 (FeatureID 제외)
X = df[['차선수', '평균속도', '통행량']].values

# 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X

# 클러스터 수 설정
range_n_clusters = range(3, 11)
silhouette_scores = []
silhouette_stddevs = []  # 클러스터 내 실루엣 값 표준편차 저장용 리스트

# 클러스터 수별 실루엣 계수 계산
for n_clusters in range_n_clusters:
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    cluster_labels = gmm.predict(X_scaled)

    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)
    
    # 각 클러스터의 실루엣 값 분포 확인
    sil_values = silhouette_samples(X_scaled, cluster_labels)
    stddev = np.std(sil_values)  # 실루엣 값의 표준편차 계산
    silhouette_stddevs.append(stddev)

    print(f'Number of clusters: {n_clusters}, Silhouette Score: {score:.4f}, Silhouette StdDev: {stddev:.4f}')

# 실루엣 계수와 클러스터 편차 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x=list(range_n_clusters), y=silhouette_scores, marker='o', label='Silhouette Score')
sns.lineplot(x=list(range_n_clusters), y=silhouette_stddevs, marker='x', label='Cluster Size StdDev')
plt.title('Silhouette Score and silhouette StdDev by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Score / StdDev')
plt.xticks(range_n_clusters)
plt.legend()
plt.show()

# 실루엣 점수가 높고 군집 크기 편차가 낮은 최적의 클러스터 수 선택
# 실루엣 점수가 높은 클러스터들 중에서, 군집 크기 표준편차가 작은 것을 선택
optimal_idx = np.argmax(np.array(silhouette_scores) - np.array(silhouette_stddevs))  # 편차를 고려한 최적 선택
optimal_clusters = range_n_clusters[optimal_idx]
print(f'The optimal number of clusters is {optimal_clusters}.')

# PCA를 사용하여 2차원으로 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 최적의 클러스터 수로 GMM 적용
gmm_optimal = GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42)
gmm_optimal.fit(X_scaled)
labels_optimal = gmm_optimal.predict(X_scaled)

# 군집화 결과 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_optimal, palette='viridis', legend='full')
plt.title(f'GMM Clustering Result (Number of Clusters: {optimal_clusters})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 3D 시각화를 위한 모듈

# 최적의 클러스터 수로 GMM 적용
gmm_optimal = GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42)
gmm_optimal.fit(X_scaled)
labels_optimal = gmm_optimal.predict(X_scaled)

# 3D 시각화
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 각 클러스터에 따른 색상 지정
scatter = ax.scatter(
    X_scaled[:, 0],  # 차선수
    X_scaled[:, 1],  # 평균속도
    X_scaled[:, 2],  # 통행량
    c=labels_optimal,
    cmap='viridis',
    s=50,
    alpha=0.6
)

# 축 레이블 설정
ax.set_title(f'GMM Clustering Result  (Number of Clusters: {optimal_clusters})', fontsize=15)
ax.set_xlabel('Lanes', fontsize=12)
ax.set_ylabel('Avg_Speed', fontsize=12)
ax.set_zlabel('Traffic_Volume', fontsize=12)

# 범례 추가
legend1 = ax.legend(*scatter.legend_elements(),
                    title="Cluster",
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1))
ax.add_artist(legend1)

plt.show()
