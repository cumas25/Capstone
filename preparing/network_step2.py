import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import matplotlib.pyplot as plt

# ── 도보 네트워크 CSV 로드 ────────────────────────
df = pd.read_csv('서울시 자치구별 도보 네트워크 공간정보.csv', encoding='cp949')

# ── 노드 / 링크 분리 ─────────────────────────────
nodes = df[df['노드링크 유형'] == 'NODE'].copy()
links = df[df['노드링크 유형'] == 'LINK'].copy()

# ── WKT → geometry 변환 + 좌표계 통일 ───────────
nodes['geometry'] = nodes['노드 WKT'].apply(wkt.loads)
links['geometry'] = links['링크 WKT'].apply(wkt.loads)

nodes_gdf = gpd.GeoDataFrame(nodes, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:5179')
links_gdf = gpd.GeoDataFrame(links, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:5179')

print("노드 수:", len(nodes_gdf))
print("링크 수:", len(links_gdf))
print("링크 횡단보도 있는 것:", links_gdf['횡단보도'].sum())

# ── 이전 단계 격자 불러오기 + 운행불가 제외 ──────
grid = gpd.read_file('grid_slope.gpkg')
grid = grid[grid['operable'] == True].copy().reset_index(drop=True)
print(f"분석 대상 격자 (운행가능): {len(grid)}개")

# ── H2: 격자별 교차로 밀도 + 연결성 ──────────────
nodes_joined = gpd.sjoin(nodes_gdf, grid[['grid_id', 'geometry']], how='left', predicate='within')
node_count   = nodes_joined.groupby('grid_id').size().rename('node_count')

links_joined = gpd.sjoin(links_gdf, grid[['grid_id', 'geometry']], how='left', predicate='intersects')
link_count   = links_joined.groupby('grid_id').size().rename('link_count')

h2 = pd.concat([node_count, link_count], axis=1).fillna(0)
h2['connectivity'] = h2['link_count'] / h2['node_count'].replace(0, float('nan'))

# ── H3: 격자별 횡단보도 수 ───────────────────────
crosswalk_links = links_gdf[links_gdf['횡단보도'] == 1]
cw_joined       = gpd.sjoin(crosswalk_links, grid[['grid_id', 'geometry']], how='left', predicate='intersects')
crosswalk_count = cw_joined.groupby('grid_id').size().rename('crosswalk_count')

# ── grid에 합치기 ────────────────────────────────
grid = grid.join(h2, on='grid_id').join(crosswalk_count, on='grid_id')
print(grid[['grid_id', 'node_count', 'link_count', 'connectivity', 'crosswalk_count']].describe())

# ── 결측값 평균 대체 ─────────────────────────────
for col in ['node_count', 'link_count', 'connectivity', 'crosswalk_count']:
    grid[col] = grid[col].fillna(grid[col].mean())

# ── 정규화 함수 ──────────────────────────────────
def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0.0
def minmax_inv(s): return 1 - minmax(s)

# ── IQR 이상치 클리핑 ────────────────────────────
def clip_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))

for col in ['node_count', 'connectivity', 'crosswalk_count']:
    grid[col] = clip_iqr(grid[col])

# ── H2 점수: 교차로 밀도(역) 0.5 + 연결성 0.5 ───
grid['h2_score'] = (
    0.5 * minmax_inv(grid['node_count']) +
    0.5 * minmax(grid['connectivity'])
)

# ── H3 점수: 횡단보도 부하(역) ───────────────────
grid['h3_score'] = minmax_inv(grid['crosswalk_count'])

print("\nH2 점수:"); print(grid['h2_score'].describe())
print("\nH3 점수:"); print(grid['h3_score'].describe())

# ── 저장 ─────────────────────────────────────────
grid.to_file('grid_h1h2h3.gpkg', driver='GPKG')
print("\n저장 완료: grid_h1h2h3.gpkg")

# ════════════════════════════════════════════════
# 시각화: H1 / H2 / H3 나란히 + 구 경계
# ════════════════════════════════════════════════
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 구 단위 경계 준비
boundary = gpd.read_file('N3A_G0110000/N3A_G0110000.shp')
boundary['GU_CD'] = boundary['BJCD'].str[:5]
gu = boundary[boundary['BJCD'].str.startswith('11')].dissolve(by='GU_CD').reset_index()
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
gu['GU_NM']    = gu['GU_CD'].map(gu_names)
gu['centroid'] = gu.geometry.centroid

seoul = boundary[boundary['BJCD'].str.startswith('11')].dissolve().to_crs('EPSG:5179')

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

configs = [
    ('h1_score', 'H1 경사도 적합도',  '경사 완만 ↑'),
    ('h2_score', 'H2 보행 네트워크',  '연결성 ↑ · 교차로 밀도 ↓'),
    ('h3_score', 'H3 횡단 환경',      '횡단보도 부하 ↓'),
]

for ax, (col, title, subtitle) in zip(axes, configs):
    grid.plot(column=col, cmap='RdYlGn', vmin=0, vmax=1,
              legend=False, missing_kwds={'color': 'lightgrey'}, ax=ax)
    gu.boundary.plot(ax=ax, color='#222222', linewidth=1.2)
    for _, row in gu.iterrows():
        ax.annotate(
            text=row['GU_NM'],
            xy=(row['centroid'].x, row['centroid'].y),
            ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='#111111',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, ec='none')
        )
    ax.set_title(f'{title}\n({subtitle})', fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')

sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label('점수  (0 = 불리  →  1 = 최적)', fontsize=11)

plt.suptitle('서울시 배달로봇 도입 적합도 — STEP 1~3 지표 (운행불가 격자 제외)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('step1_3_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장 완료: step1_3_scores.png")