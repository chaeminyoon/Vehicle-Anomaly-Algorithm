import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 데이터 입력 (예시 데이터 사용)
data = """
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

# 문자열 데이터를 pandas DataFrame으로 변환
from io import StringIO
df = pd.read_csv(StringIO(data), sep='\t')

# 필요한 피처 선택
X_features = df[['차선수', '평균속도', '통행량']]

# 데이터 스케일링 (표준화)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_features)

# GMM을 사용한 실루엣 분석 및 시각화 함수 정의
def visualize_silhouette_gmm(cluster_lists, X_features): 
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    cluster_labels_list = []

    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    for ind, n_cluster in enumerate(cluster_lists):
        # GMM 클러스터링 수행
        gmm = GaussianMixture(n_components=n_cluster, covariance_type='full', random_state=42)
        cluster_labels = gmm.fit_predict(X_features)
        cluster_labels_list.append(cluster_labels)
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        if n_cols == 1:
            ax = axs
        else:
            ax = axs[ind]
        ax.set_title('Number of Clusters: ' + str(n_cluster) + '\n' \
                      'Silhouette Score: ' + str(round(sil_avg, 3)))
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        ax.set_yticks([])  # y축 라벨 제거
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터별로 실루엣 계수를 막대 그래프로 표현
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels == i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        ax.axvline(x=sil_avg, color="red", linestyle="--")
    
    plt.tight_layout()
    plt.show()

    return cluster_labels_list

# 클러스터 개수를 3개부터 6개까지 적용하여 시각화
cluster_lists = [3, 4, 5, 6, 7, 8, 9, 10]
cluster_labels_list = visualize_silhouette_gmm(cluster_lists, X_scaled)

# 예를 들어, 7개의 클러스터에 대한 클러스터 레이블을 확인
cluster_labels_7 = cluster_labels_list[4]

# 각 클러스터에 속한 데이터 수를 확인
for i, cluster_labels in enumerate(cluster_labels_list):
    n_clusters = cluster_lists[i]  # 현재 클러스터 개수
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    print(f"{n_clusters}개의 클러스터별 데이터 수: {cluster_distribution}")

