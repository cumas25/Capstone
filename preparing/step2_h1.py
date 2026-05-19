import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. 격자 로드 (step1 결과) ──────────────────────
grid = gpd.read_file('step1_grid.gpkg')
print(f"[STEP1] 격자: {len(grid)}개")

# ── 2. 표고점 로드 (EPSG:5174 → 5179) ─────────────
elev = gpd.read_file('서울시 경사도/표고 5000/N3P_F002.shp')
print(f"[표고점] 원본 CRS: {elev.crs} | {len(elev)}개")
elev = elev.to_crs('EPSG:5179')

coords_elev = np.array([[g.x, g.y] for g in elev.geometry])
heights     = elev['HEIGHT'].values
tree        = cKDTree(coords_elev)
print(f"[표고점] EPSG:5179 변환 완료 → KDTree 구축")

# ── 3. 도보 네트워크 로드 → 링크만 추출 ───────────
df       = pd.read_csv('서울시 자치구별 도보 네트워크 공간정보.csv', encoding='cp949')
links_df = df[df['노드링크 유형'] == 'LINK'].copy()
links_df['geometry'] = links_df['링크 WKT'].apply(wkt.loads)
links_gdf = gpd.GeoDataFrame(links_df, geometry='geometry', crs='EPSG:4326')
links_gdf = links_gdf.to_crs('EPSG:5179').reset_index(drop=True)
print(f"[도보 링크] {len(links_gdf)}개 로드 완료")

# ── 4. 링크별 경사도 계산 (벡터 연산) ─────────────
# 링크 양 끝점 → 가장 가까운 표고점 HEIGHT → 경사도(%)
all_starts = np.array([[list(g.coords)[0][0],  list(g.coords)[0][1]]  for g in links_gdf.geometry])
all_ends   = np.array([[list(g.coords)[-1][0], list(g.coords)[-1][1]] for g in links_gdf.geometry])

_, idx_s = tree.query(all_starts)
_, idx_e = tree.query(all_ends)

links_gdf['link_length'] = links_gdf.geometry.length
links_gdf['slope'] = (
    np.abs(heights[idx_s] - heights[idx_e]) / links_gdf['link_length'].values * 100
)

# ── 5. 이상값 제거 ────────────────────────────────
n_before = len(links_gdf)
links_gdf = links_gdf[
    (links_gdf['link_length'] >= 1) &   # 1m 미만 링크 제외
    (links_gdf['slope'] <= 100)          # 경사도 100% 초과 제외
].copy()
print(f"[품질] 이상값 제거: {n_before - len(links_gdf)}개 → 남은 링크: {len(links_gdf)}개")

# ── 6. BF 인증 기준 3단계 분류 ────────────────────
# 최적(≤5%): BF 권장 1/18(≈5.6%)보다 보수적 적용
# 허용(5~8%): BF 허용 상한 1/12(≈8.3%) 미만
# 불가(>8%): BF 최대 허용 초과 → 보행 약자도 통행 불가
def classify(s):
    if s <= 5:   return 'optimal'
    elif s <= 8: return 'allowed'
    else:        return 'impossible'

links_gdf['slope_class'] = links_gdf['slope'].apply(classify)

total_km = links_gdf['link_length'].sum() / 1000
print(f"\n=== 링크 경사도 분포 (전체 {total_km:.1f} km) ===")
for cls in ['optimal', 'allowed', 'impossible']:
    mask    = links_gdf['slope_class'] == cls
    n       = mask.sum()
    len_km  = links_gdf.loc[mask, 'link_length'].sum() / 1000
    print(f"  {cls:12s}: {n:6d}개 | {len_km:7.1f} km ({len_km/total_km*100:.1f}%)")

# ── 7. 격자별 집계 (링크 길이 가중) ──────────────
links_gdf['imp_len'] = links_gdf['link_length'] * (links_gdf['slope_class'] == 'impossible')
links_gdf['opt_len'] = links_gdf['link_length'] * (links_gdf['slope_class'] == 'optimal')

joined = gpd.sjoin(
    links_gdf[['link_length', 'imp_len', 'opt_len', 'geometry']],
    grid[['grid_id', 'geometry']],
    how='inner', predicate='intersects'
)

agg = joined.groupby('grid_id').agg(
    total_link_len=('link_length', 'sum'),
    imp_link_len  =('imp_len',     'sum'),
    opt_link_len  =('opt_len',     'sum'),
).reset_index()

agg['impossible_ratio'] = agg['imp_link_len'] / agg['total_link_len'] * 100
agg['optimal_ratio']    = agg['opt_link_len'] / agg['total_link_len'] * 100

grid = grid.merge(agg[['grid_id', 'total_link_len', 'impossible_ratio', 'optimal_ratio']],
                  on='grid_id', how='left')

# 도보 링크 없는 격자: 운행 불가 처리
grid['impossible_ratio'] = grid['impossible_ratio'].fillna(100)
grid['optimal_ratio']    = grid['optimal_ratio'].fillna(0)
grid['total_link_len']   = grid['total_link_len'].fillna(0)

# ── 8. H1 필터: 불가 비율 50% 초과 → 운행불가 ────
grid['operable'] = grid['impossible_ratio'] <= 50

n_ok  = grid['operable'].sum()
n_out = (~grid['operable']).sum()
print(f"\n[H1 필터] 운행 가능: {n_ok}개 | 운행 불가: {n_out}개 ({n_out/len(grid)*100:.1f}%)")

# ── 9. H1 점수화 (운행 가능 격자만) ───────────────
def clip_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))

def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0.0

grid['h1_score'] = np.nan
mask = grid['operable']
grid.loc[mask, 'h1_score'] = minmax(clip_iqr(grid.loc[mask, 'optimal_ratio']))

print(f"\n[H1 점수] 운행 가능 격자 {n_ok}개 기준")
print(grid.loc[mask, 'h1_score'].describe().round(3))

# ── 10. 저장 ──────────────────────────────────────
grid.to_file('step2_h1.gpkg', driver='GPKG')
print("\n저장 완료: step2_h1.gpkg")

# ── 11. 시각화 ────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 구 경계
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
        if pd.notna(row['GU_NM']):
            ax.annotate(row['GU_NM'], xy=(row['centroid'].x, row['centroid'].y),
                        ha='center', va='center', fontsize=6,
                        fontweight='bold', color='#111111',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, ec='none'))

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# 왼쪽: H1 필터 결과
ax = axes[0]
grid[~grid['operable']].plot(ax=ax, color='#E53935', alpha=0.8)
grid[grid['operable']].plot(ax=ax, color='#43A047', alpha=0.8)
add_gu_labels(ax)
ax.set_title(f'H1 경사도 필터 결과\n운행 가능 {n_ok}개(녹) | 운행 불가 {n_out}개(빨)', fontsize=13)
ax.axis('off')
patches = [mpatches.Patch(color='#43A047', label=f'운행 가능 ({n_ok}개)'),
           mpatches.Patch(color='#E53935', label=f'운행 불가 ({n_out}개)')]
ax.legend(handles=patches, loc='lower right', fontsize=10)

# 오른쪽: H1 점수 (운행 가능 격자만)
ax2 = axes[1]
grid[~grid['operable']].plot(ax=ax2, color='#CCCCCC')
grid[grid['operable']].plot(
    column='h1_score', cmap='RdYlGn', vmin=0, vmax=1,
    legend=True, legend_kwds={'label': 'H1 점수 (0=불리, 1=최적)', 'shrink': 0.6},
    ax=ax2
)
add_gu_labels(ax2)
ax2.set_title('H1 경사도 점수\n(회색 = 운행불가 제외)', fontsize=13)
ax2.axis('off')

plt.suptitle('STEP 2 — H1 경사도 분석 (BF 인증 기준 / 도보 링크 기반)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('step2_h1.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장 완료: step2_h1.png")
