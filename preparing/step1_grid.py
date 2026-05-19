import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.validation import make_valid

CELL_SIZE = 500   # m (EPSG:5179 단위)
MIN_AREA_RATIO = 0.1  # 격자 면적이 전체의 10% 미만이면 경계 파편으로 제거

# ── 1. 서울 행정경계 로드 ──────────────────────────
boundary = gpd.read_file('N3A_G0110000/N3A_G0110000.shp')
print(f"[원본] CRS: {boundary.crs} | 행 수: {len(boundary)}")

# EPSG:5179 통일 (단위: m)
if boundary.crs is None or boundary.crs.to_epsg() != 5179:
    boundary = boundary.to_crs('EPSG:5179')
    print("  → EPSG:5179 변환 완료")

# 서울시 행정동만 필터링 (BJCD 앞 2자리 '11')
seoul_dong = boundary[boundary['BJCD'].str.startswith('11')].copy()
print(f"[서울] 행정동 수: {len(seoul_dong)}")

# ── 2. Geometry 유효성 검사 및 수정 ───────────────
invalid_mask = ~seoul_dong.geometry.is_valid
if invalid_mask.any():
    print(f"  유효하지 않은 geometry {invalid_mask.sum()}개 → make_valid() 수정")
    seoul_dong.loc[invalid_mask, 'geometry'] = (
        seoul_dong.loc[invalid_mask, 'geometry'].apply(make_valid)
    )

# ── 3. 서울 전체 경계 (단일 polygon) ──────────────
seoul = seoul_dong.dissolve()
area_km2 = seoul.geometry.area.values[0] / 1e6
print(f"[서울 전체] 면적: {area_km2:.1f} km²  (기준값: 약 605 km²)")
if not (580 < area_km2 < 630):
    print("  ⚠️  면적이 예상 범위(580~630 km²)를 벗어남 — CRS 또는 경계 데이터 확인 필요")

# ── 4. 500m × 500m 격자 생성 ──────────────────────
xmin, ymin, xmax, ymax = seoul.total_bounds
print(f"[Bounding Box] x: {xmin:.0f} ~ {xmax:.0f} | y: {ymin:.0f} ~ {ymax:.0f}")

xs = np.arange(xmin, xmax + CELL_SIZE, CELL_SIZE)
ys = np.arange(ymin, ymax + CELL_SIZE, CELL_SIZE)

polygons = [
    box(x, y, x + CELL_SIZE, y + CELL_SIZE)
    for x in xs[:-1]
    for y in ys[:-1]
]
grid_raw = gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:5179')
print(f"[격자] 클리핑 전: {len(grid_raw)}개")

# ── 5. 서울 경계 클리핑 ───────────────────────────
grid = gpd.clip(grid_raw, seoul).reset_index(drop=True)
print(f"[격자] 클리핑 후: {len(grid)}개")

# ── 6. 경계 파편 제거 (면적 비율 10% 미만) ────────
grid['area_m2']    = grid.geometry.area
grid['area_ratio'] = grid['area_m2'] / (CELL_SIZE ** 2)

n_before   = len(grid)
grid       = grid[grid['area_ratio'] >= MIN_AREA_RATIO].copy()
n_removed  = n_before - len(grid)
print(f"[품질] 경계 파편 제거: {n_removed}개 → 남은 격자: {len(grid)}개")

# ── 7. grid_id 부여 및 중심 좌표 저장 ─────────────
grid = grid.reset_index(drop=True)
grid['grid_id'] = grid.index

centroids        = grid.geometry.centroid
grid['cx_5179']  = centroids.x
grid['cy_5179']  = centroids.y

# ── 8. 최종 검증 ──────────────────────────────────
assert grid['grid_id'].is_unique,     "grid_id 중복 존재"
assert grid.geometry.is_valid.all(),  "유효하지 않은 geometry 존재"
assert grid.crs.to_epsg() == 5179,   "CRS가 EPSG:5179가 아님"

print(f"\n=== 최종 격자 요약 ===")
print(f"  격자 수       : {len(grid)}")
print(f"  격자 크기     : {CELL_SIZE}m × {CELL_SIZE}m")
print(f"  면적 범위     : {grid['area_m2'].min():.0f} ~ {grid['area_m2'].max():.0f} m²")
print(f"  CRS           : EPSG:{grid.crs.to_epsg()}")

# ── 9. 저장 ───────────────────────────────────────
out_cols = ['grid_id', 'area_m2', 'area_ratio', 'cx_5179', 'cy_5179', 'geometry']
grid[out_cols].to_file('step1_grid.gpkg', driver='GPKG')
print("\n저장 완료: step1_grid.gpkg")

# ── 10. 시각화 ────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 왼쪽: 전체 격자 (면적 비율로 색 구분)
ax = axes[0]
grid.plot(
    column='area_ratio', cmap='YlGn', vmin=0, vmax=1,
    edgecolor='grey', linewidth=0.2, ax=ax
)
seoul.boundary.plot(ax=ax, color='black', linewidth=1.5)
ax.set_title(f'서울시 500m 격자 (총 {len(grid)}개)\n면적 비율 (1.0 = 완전 격자)', fontsize=13)
ax.axis('off')

sm = plt.cm.ScalarMappable(cmap='YlGn', norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.6, label='면적 비율')

# 오른쪽: 완전 격자 vs 경계 격자 구분
ax2 = axes[1]
full   = grid[grid['area_ratio'] == 1.0]
edge   = grid[grid['area_ratio'] <  1.0]
full.plot(ax=ax2, color='#4CAF50', edgecolor='grey', linewidth=0.2)
edge.plot(ax=ax2, color='#FF9800', edgecolor='grey', linewidth=0.2)
seoul.boundary.plot(ax=ax2, color='black', linewidth=1.5)
ax2.set_title(
    f'완전 격자: {len(full)}개 (녹색)\n경계 격자: {len(edge)}개 (주황)',
    fontsize=13
)
ax2.axis('off')
patches = [
    mpatches.Patch(color='#4CAF50', label=f'완전 격자 ({len(full)}개)'),
    mpatches.Patch(color='#FF9800', label=f'경계 격자 ({len(edge)}개)'),
]
ax2.legend(handles=patches, loc='lower right', fontsize=10)

plt.suptitle('STEP 1 — 서울시 500m × 500m 격자 생성 결과', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('step1_grid.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장 완료: step1_grid.png")
