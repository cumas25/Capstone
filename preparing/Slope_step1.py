import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def clip_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))

def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0.0


# ── 1. 경사도 데이터 로드 ─────────────────────────
elev = gpd.read_file('서울시 등고선/표고 5000/N3P_F002.shp')
elev = elev.to_crs('EPSG:5179')

coords  = np.array([[geom.x, geom.y] for geom in elev.geometry])
heights = elev['HEIGHT'].values
tree    = cKDTree(coords)
distances, indices = tree.query(coords, k=2)
nearest_dist = distances[:, 1]
height_diff  = np.abs(heights - heights[indices[:, 1]])
slope_pct    = (height_diff / nearest_dist) * 100

elev['slope'] = slope_pct
elev = elev[elev['slope'] <= 100].copy()

def classify_slope(s):
    if s <= 5:   return 'Optimal'
    elif s <= 8: return 'Allowed'
    else:        return 'Impossible'

elev['slope_class'] = elev['slope'].apply(classify_slope)

# ── 2. 서울 경계 ──────────────────────────────────
boundary = gpd.read_file('N3A_G0110000/N3A_G0110000.shp')
seoul    = boundary[boundary['BJCD'].str.startswith('11')].dissolve()
print("서울 경계 CRS:", seoul.crs)

# ── 3. 500m × 500m 격자 생성 ─────────────────────
xmin, ymin, xmax, ymax = seoul.total_bounds
cols     = np.arange(xmin, xmax, 500)
rows     = np.arange(ymin, ymax, 500)
polygons = [box(x, y, x + 500, y + 500) for x in cols for y in rows]
grid     = gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:5179')

# ── 4. 서울 경계 클리핑 ───────────────────────────
grid          = gpd.clip(grid, seoul).reset_index(drop=True)
grid['grid_id'] = grid.index
print(f"전체 격자 수: {len(grid)}")

# ── 5. 경사도 포인트 → 격자 공간 조인 ────────────
joined = gpd.sjoin(elev, grid[['grid_id', 'geometry']], how='left', predicate='within')

# ── 6. 격자별 경사도 지표 산출 ───────────────────
def grid_slope_stats(g):
    total = len(g)
    if total == 0:
        return {'optimal_ratio': np.nan, 'mean_slope': np.nan, 'impossible_ratio': np.nan}
    return {
        'optimal_ratio':    (g['slope_class'] == 'Optimal').sum()    / total * 100,
        'mean_slope':        g['slope'].mean(),
        'impossible_ratio': (g['slope_class'] == 'Impossible').sum() / total * 100,
    }

slope_stats = (
    joined.dropna(subset=['grid_id'])
          .groupby('grid_id')
          .apply(grid_slope_stats)
          .apply(pd.Series)
)
grid = grid.join(slope_stats, on='grid_id')

# ── 7. 1차 필터: 불가 비율 50% 초과 → 운행불가 ──
grid['operable'] = grid['impossible_ratio'].fillna(100) <= 50

n_ok   = grid['operable'].sum()
n_out  = (~grid['operable']).sum()
print(f"운행 가능 격자: {n_ok}개")
print(f"운행 불가 격자: {n_out}개 ({n_out/len(grid)*100:.1f}%)")

# ── 8. 가능 격자만 H1 점수화 ─────────────────────
operable_mask = grid['operable']
grid['h1_score'] = np.nan
clipped = clip_iqr(grid.loc[operable_mask, 'optimal_ratio'])
grid.loc[operable_mask, 'h1_score'] = minmax(clipped)

print(grid.loc[operable_mask, ['grid_id', 'optimal_ratio', 'mean_slope', 'h1_score']].describe())

# ── 9. 저장 ──────────────────────────────────────
grid.to_file('grid_slope.gpkg', driver='GPKG')
print("저장 완료: grid_slope.gpkg")

# ── 10. 시각화 ───────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 구 단위 경계 준비
boundary_gu = gpd.read_file('N3A_G0110000/N3A_G0110000.shp')
boundary_gu['GU_CD'] = boundary_gu['BJCD'].str[:5]
gu = boundary_gu[boundary_gu['BJCD'].str.startswith('11')].dissolve(by='GU_CD').reset_index()
gu = gu.to_crs('EPSG:5179')

gu_names = {
    '11110': '종로구', '11140': '중구',     '11170': '용산구',  '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구',  '11290': '성북구',
    '11305': '강북구', '11320': '도봉구',   '11350': '노원구',  '11380': '은평구',
    '11410': '서대문구','11440': '마포구',  '11470': '양천구',  '11500': '강서구',
    '11530': '구로구', '11545': '금천구',   '11560': '영등포구','11590': '동작구',
    '11620': '관악구', '11650': '서초구',   '11680': '강남구',  '11710': '송파구',
    '11740': '강동구'
}
gu['GU_NM']   = gu['GU_CD'].map(gu_names)
gu['centroid'] = gu.geometry.centroid

fig, ax = plt.subplots(figsize=(12, 12))

# 운행불가 격자 → 진한 회색
grid[~grid['operable']].plot(ax=ax, color='#AAAAAA')

# 운행가능 격자 → 점수 색상
grid[grid['operable']].plot(
    column='h1_score', cmap='RdYlGn', vmin=0, vmax=1,
    legend=True,
    legend_kwds={'label': 'H1 경사도 점수 (0=불리, 1=최적)', 'shrink': 0.6},
    ax=ax
)

# 구 경계선
gu.boundary.plot(ax=ax, color='#222222', linewidth=1.5)

# 구 이름 레이블
for _, row in gu.iterrows():
    ax.annotate(
        text=row['GU_NM'],
        xy=(row['centroid'].x, row['centroid'].y),
        ha='center', va='center',
        fontsize=7, fontweight='bold', color='#111111',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, ec='none')
    )

ax.set_title('서울시 배달로봇 경사도 적합도\n(회색 = 운행불가 1차 제외)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('grid_slope_map.png', dpi=150)
plt.show()
print("저장 완료: grid_slope_map.png")