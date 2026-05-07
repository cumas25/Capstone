import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# 표고점 로드
elev = gpd.read_file('서울시 등고선/표고 5000/N3P_F002.shp')

# 좌표 및 높이 추출
coords = np.array([[geom.x, geom.y] for geom in elev.geometry])
heights = elev['HEIGHT'].values

# KNN (k=2: 자기자신 + 가장 가까운 점)
tree = cKDTree(coords)
distances, indices = tree.query(coords, k=2)

# 경사도 계산
nearest_dist = distances[:, 1]
height_diff = np.abs(heights - heights[indices[:, 1]])
slope_pct = (height_diff / nearest_dist) * 100
elev['slope'] = slope_pct

# 이상값 제거
elev = elev[elev['slope'] <= 100]

# 3단계 분류
def classify_slope(s):
    if s <= 5:   return 'Optimal'
    elif s <= 8: return 'Allowed'
    else:        return 'Impossible'

elev['slope_class'] = elev['slope'].apply(classify_slope)

# 결과 출력
print('=== 경사도 분포 ===')
print(elev['slope'].describe())
print()
print('=== 구간별 비율 ===')
print(elev['slope_class'].value_counts(normalize=True).mul(100).round(1))

# 시각화
color_map = {'Optimal': 'green', 'Allowed': 'orange', 'Impossible': 'red'}
elev['color'] = elev['slope_class'].map(color_map)

fig, ax = plt.subplots(figsize=(12, 12))
for cls, color in color_map.items():
    subset = elev[elev['slope_class'] == cls]
    subset.plot(ax=ax, color=color, markersize=0.5, label=cls)

plt.legend(title='Slope Grade', markerscale=5)
plt.title('Seoul Delivery Robot Accessible Area (Slope Analysis)')
plt.tight_layout()
plt.savefig('slope_map.png', dpi=150)
plt.show()