import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 입력
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

# 특성 선택 (지점수를 제외한 나머지 컬럼)
X = df[['차선수', '평균속도', '통행량']]

# 데이터 스케일링 (표준화)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-거리 그래프를 위한 k 설정
k = 4
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# k번째 거리를 정렬
distances = np.sort(distances[:, k-1], axis=0)

# k-거리 그래프 그리기
plt.figure(figsize=(10,5))
plt.plot(distances)
plt.xlabel('데이터 포인트 정렬 순서')
plt.ylabel(f'{k}번째 최근접 거리')
plt.title('k-거리 그래프을 통해 ε 결정')
plt.grid(True)
plt.show()

# DBSCAN 군집화 수행
# eps와 min_samples 값을 조정하여 4개의 군집을 생성
dbscan = DBSCAN(eps=1.2, min_samples=2)
clusters = dbscan.fit_predict(X_scaled)

# 클러스터 결과를 데이터프레임에 추가
df['cluster'] = clusters

# 실루엣 스코어 계산 (클러스터가 1개 이하인 경우 계산 불가)
if len(set(clusters)) > 1:
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"실루엣 점수: {silhouette_avg:.2f}")
else:
    silhouette_avg = None
    print("실루엣 점수를 계산할 수 없습니다.")

# 각 군집별로 기초 통계량 계산
cluster_groups = df.groupby('cluster')

for cluster_num, group in cluster_groups:
    print(f"\n### 군집 {cluster_num} ###")
    print("지점 번호:", ', '.join(map(str, group['지점수'].values)))
    print("\n기초 통계:")
    print(group[['차선수', '평균속도', '통행량']].agg(['mean', 'std', 'max', 'min']))

# 결과 확인을 위해 데이터프레임 출력 (선택사항)
# print(df)
