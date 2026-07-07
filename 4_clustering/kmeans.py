import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

# '통행량'에 로그 변환 적용
df['통행량_log'] = np.log1p(df['통행량'])

# 특성 선택 ('차선수', '평균속도', '통행량_log' 사용)
X_features = df[['차선수', '평균속도', '통행량_log']]

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# KMeans 군집화 수행 (클러스터 개수: 5개)
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, random_state=0).fit(X_scaled)
df['cluster'] = kmeans.labels_

# 모든 개별 데이터에 실루엣 계수값을 구함
score_samples = silhouette_samples(X_scaled, df['cluster'])
df['silhouette_coeff'] = score_samples

# 모든 데이터의 평균 실루엣 계수값을 구함
average_score = silhouette_score(X_scaled, df['cluster'])
print('데이터셋 Silhouette Analysis Score: {0:.3f}'.format(average_score))

# 클러스터별 평균 실루엣 계수값 출력
print("\n클러스터별 평균 실루엣 계수:")
print(df.groupby('cluster')['silhouette_coeff'].mean())

# 각 군집별로 기초 통계량 계산 및 출력
cluster_groups = df.groupby('cluster')

for cluster_num, group in cluster_groups:
    print(f"\n### 군집 {cluster_num + 1} ###")
    print("지점 번호:", ', '.join(map(str, group['지점수'].values)))
    print("\n기초 통계:")
    print(group[['차선수', '평균속도', '통행량']].agg(['mean', 'std', 'max', 'min']))
