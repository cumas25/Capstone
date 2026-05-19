import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.geometry import Point
import matplotlib.pyplot as plt

# ── 공통 함수 ─────────────────────────────────────
def clip_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))

def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0.0

def minmax_inv(s):
    return 1 - minmax(s)

# ── 1. 격자 로드 (H1 필터 통과 격자만) ───────────
grid = gpd.read_file('step2_h1.gpkg')
grid = grid[grid['operable'] == True].copy().reset_index(drop=True)
print(f"[STEP2] 운행 가능 격자: {len(grid)}개")

# ── 2. 도보 네트워크 로드 ─────────────────────────
df = pd.read_csv('서울시 자치구별 도보 네트워크 공간정보.csv', encoding='cp949')

nodes_df = df[df['노드링크 유형'] == 'NODE'].copy()
links_df = df[df['노드링크 유형'] == 'LINK'].copy()

nodes_df['geometry'] = nodes_df['노드 WKT'].apply(wkt.loads)
links_df['geometry'] = links_df['링크 WKT'].apply(wkt.loads)

nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:5179')
links_gdf = gpd.GeoDataFrame(links_df, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:5179')

print(f"[네트워크] 노드: {len(nodes_gdf)}개 | 링크: {len(links_gdf)}개")

# ── 3. H2 — 보행 네트워크 지표 산출 ──────────────
# 격자 면적 (km²)
grid['area_km2'] = grid.geometry.area / 1e6

# 격자별 노드 수
nodes_joined = gpd.sjoin(
    nodes_gdf[['geometry']],
    grid[['grid_id', 'geometry']],
    how='inner', predicate='within'
)
node_count = nodes_joined.groupby('grid_id').size().rename('node_count')

# 격자별 링크 수
links_joined = gpd.sjoin(
    links_gdf[['geometry']],
    grid[['grid_id', 'geometry']],
    how='inner', predicate='intersects'
)
link_count = links_joined.groupby('grid_id').size().rename('link_count')

grid = grid.join(node_count, on='grid_id').join(link_count, on='grid_id')
grid['node_count'] = grid['node_count'].fillna(0)
grid['link_count'] = grid['link_count'].fillna(0)

# 링크 연결도 = 링크 수 / 노드 수 (정방향: 높을수록 우회 경로 풍부)
grid['connectivity'] = (
    grid['link_count'] / grid['node_count'].replace(0, np.nan)
).fillna(0)

# 네트워크 복잡도 = 노드 수 / 면적 km² (역방향: 높을수록 의사결정 포인트 많음)
grid['complexity'] = grid['node_count'] / grid['area_km2']

# H2 점수
grid['h2_score'] = (
    0.5 * minmax(clip_iqr(grid['connectivity'])) +
    0.5 * minmax_inv(clip_iqr(grid['complexity']))
)

print(f"\n[H2] 연결도 평균: {grid['connectivity'].mean():.2f} | 복잡도 평균: {grid['complexity'].mean():.1f}")
print(f"[H2] 점수 분포:")
print(grid['h2_score'].describe().round(3))

# ── 4. H3 — 횡단 환경 지표 산출 ──────────────────
# 횡단보도 링크 (정방향: 많을수록 목적지 접근 경로 다양)
crosswalk_links = links_gdf[links_gdf['횡단보도'] == 1].copy()
cw_joined = gpd.sjoin(
    crosswalk_links[['geometry']],
    grid[['grid_id', 'geometry']],
    how='inner', predicate='intersects'
)
crosswalk_count = cw_joined.groupby('grid_id').size().rename('crosswalk_count')
grid = grid.join(crosswalk_count, on='grid_id')
grid['crosswalk_count'] = grid['crosswalk_count'].fillna(0)

# 신호등 (역방향: 많을수록 대기 빈도 높아 이동 예측 어려움)
# 좌표계: EPSG:5174 → 5179 변환
sig_df = pd.read_csv('서울특별시_보행자 신호등 분포도.csv', encoding='cp949')
sig_gdf = gpd.GeoDataFrame(
    sig_df,
    geometry=[Point(x, y) for x, y in zip(sig_df['X좌표'], sig_df['Y좌표'])],
    crs='EPSG:5186'
).to_crs('EPSG:5179')
print(f"\n[신호등] {len(sig_gdf)}개 로드 완료 (EPSG:5186 → 5179)")

sig_joined = gpd.sjoin(
    sig_gdf[['geometry']],
    grid[['grid_id', 'geometry']],
    how='inner', predicate='within'
)
signal_count = sig_joined.groupby('grid_id').size().rename('signal_count')
grid = grid.join(signal_count, on='grid_id')
grid['signal_count'] = grid['signal_count'].fillna(0)

# 신호등 밀도 = 신호등 수 / 면적 km²
grid['signal_density'] = grid['signal_count'] / grid['area_km2']

# H3 점수 (출입구 데이터 미확보 → 횡단보도 0.5 + 신호등 0.5)
# ⚠️ 출입구 데이터 확보 시 가중치 재조정 필요 (횡단보도 0.4 / 신호등 0.4 / 출입구 0.2)
grid['h3_score'] = (
    0.5 * minmax(clip_iqr(grid['crosswalk_count'])) +
    0.5 * minmax_inv(clip_iqr(grid['signal_density']))
)

print(f"[H3] 횡단보도 평균: {grid['crosswalk_count'].mean():.1f} | 신호등 평균: {grid['signal_count'].mean():.1f}")
print(f"[H3] 점수 분포:")
print(grid['h3_score'].describe().round(3))

# ── 5. 저장 ───────────────────────────────────────
grid.to_file('step3_h2_h3.gpkg', driver='GPKG')
print("\n저장 완료: step3_h2_h3.gpkg")

# ── 6. 시각화 ─────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

boundary = gpd.read_file('N3A_G0110000/N3A_G0110000.shp').to_crs('EPSG:5179')
seoul_gu = boundary[boundary['BJCD'].str.startswith('11')].copy()
seoul_gu['GU_CD'] = seoul_gu['BJCD'].str[:5]
gu = seoul_gu.dissolve(by='GU_CD').reset_index()
gu['centroid'] = gu.geometry.centroid

GU_NAMES = {
    '11110':'종로구','11140':'중구',    '11170':'용산구', '11200':'성동구',
    '11215':'광진구','11230':'동대문구','11260':'중랑구', '11290':'성북구',
    '11305':'강북구','11320':'도봉구',  '11350':'노원구', '11380':'은평구',
    '11410':'서대문구','11440':'마포구','11470':'양천구', '11500':'강서구',
    '11530':'구로구','11545':'금천구',  '11560':'영등포구','11590':'동작구',
    '11620':'관악구','11650':'서초구',  '11680':'강남구', '11710':'송파구',
    '11740':'강동구'
}
gu['GU_NM'] = gu['GU_CD'].map(GU_NAMES)

def add_gu_labels(ax):
    gu.boundary.plot(ax=ax, color='#333333', linewidth=1.2)
    for _, row in gu.iterrows():
        if pd.notna(row.get('GU_NM')):
            ax.annotate(row['GU_NM'], xy=(row['centroid'].x, row['centroid'].y),
                        ha='center', va='center', fontsize=6, fontweight='bold',
                        color='#111111',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, ec='none'))

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

for ax, col, title in [
    (axes[0], 'h2_score', 'H2 보행 네트워크\n(연결도↑ · 복잡도↓)'),
    (axes[1], 'h3_score', 'H3 횡단 환경\n(횡단보도↑ · 신호등↓)'),
]:
    grid.plot(column=col, cmap='RdYlGn', vmin=0, vmax=1,
              legend=True,
              legend_kwds={'label': f'{col} (0=불리, 1=최적)', 'shrink': 0.6},
              ax=ax)
    add_gu_labels(ax)
    ax.set_title(title, fontsize=13)
    ax.axis('off')

plt.suptitle('STEP 3 — H2·H3 보행 환경 분석 (운행 가능 격자 기준)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('step3_h2_h3.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장 완료: step3_h2_h3.png")
