import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import time
import os
import matplotlib.pyplot as plt
from pyproj import Transformer

def _load_env(path='.env.local'):
    if not os.path.exists(path):
        return {}
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env

_env = _load_env()
KAKAO_API_KEY = _env.get('KAKAO_API_KEY') or os.environ.get('KAKAO_API_KEY', '')

# 500m 격자 → 2×2 서브셀 분할 파라미터
# 서브셀 중심: 격자 중심에서 ±125m 오프셋 (EPSG:5179 기준)
# 서브셀 반경: 250m 정사각형 외접원 ≈ 177m → 180m
_SUB_OFFSET  = 125
_SUB_RADIUS  = 180
_SUB_OFFSETS = [-_SUB_OFFSET, _SUB_OFFSET]
_to_wgs84    = Transformer.from_crs('EPSG:5179', 'EPSG:4326', always_xy=True)

GU_NAMES = {
    '11110':'종로구','11140':'중구',    '11170':'용산구', '11200':'성동구',
    '11215':'광진구','11230':'동대문구','11260':'중랑구', '11290':'성북구',
    '11305':'강북구','11320':'도봉구',  '11350':'노원구', '11380':'은평구',
    '11410':'서대문구','11440':'마포구','11470':'양천구', '11500':'강서구',
    '11530':'구로구','11545':'금천구',  '11560':'영등포구','11590':'동작구',
    '11620':'관악구','11650':'서초구',  '11680':'강남구', '11710':'송파구',
    '11740':'강동구'
}

# ── 공통 함수 ─────────────────────────────────────
def clip_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))

def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0.0

# ── 카카오 API 함수 ───────────────────────────────
def _fetch_ids(lon, lat, category_group_code, max_retries=4):
    """카카오 카테고리 검색 — 반경 내 장소 ID 집합 반환 (최대 45개)"""
    url     = 'https://dapi.kakao.com/v2/local/search/category.json'
    headers = {'Authorization': f'KakaoAK {KAKAO_API_KEY}'}
    ids     = set()
    for page in range(1, 4):
        params = {
            'category_group_code': category_group_code,
            'x': lon, 'y': lat,
            'radius': _SUB_RADIUS,
            'page': page,
            'size': 15,
        }
        data = None
        for attempt in range(max_retries):
            try:
                res  = requests.get(url, headers=headers, params=params, timeout=20)
                data = res.json()
                break
            except Exception as e:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                if attempt < max_retries - 1:
                    print(f'  요청 오류 → {wait}s 후 재시도 ({attempt+1}/{max_retries}): {e}')
                    time.sleep(wait)
                else:
                    print(f'  요청 실패 (재시도 {max_retries}회 초과): {e}')
        if data is None:
            break
        if 'meta' not in data:
            print(f'  API 오류: {data}')
            break
        for doc in data.get('documents', []):
            ids.add(doc['id'])
        if data['meta']['is_end']:
            break
        time.sleep(0.05)
    return ids

def kakao_count_tiled(cx_5179, cy_5179, category_group_code):
    """500m 격자를 2×2 서브셀로 분할해 장소 ID 중복 제거 후 카운트"""
    all_ids = set()
    for dx in _SUB_OFFSETS:
        for dy in _SUB_OFFSETS:
            lon, lat = _to_wgs84.transform(cx_5179 + dx, cy_5179 + dy)
            all_ids.update(_fetch_ids(lon, lat, category_group_code))
            time.sleep(0.05)
    return len(all_ids)

# ── 1. 격자 로드 (H2·H3 결과) ─────────────────────
grid = gpd.read_file('step3_h2_h3.gpkg')
print(f"[STEP3] 격자: {len(grid)}개")

# ── 2. 카카오 API — 음식점·편의점 수집 ───────────
CHECKPOINT = 'step4_checkpoint.csv'
SAVE_EVERY = 50

if os.path.exists(CHECKPOINT):
    ckpt     = pd.read_csv(CHECKPOINT, index_col='grid_id')
    done_ids = set(ckpt.index)
    print(f"[체크포인트] {len(done_ids)}개 격자 이미 완료, 이어서 진행")
else:
    ckpt     = pd.DataFrame(columns=['grid_id', 'restaurant_count', 'cvs_count']).set_index('grid_id')
    done_ids = set()

restaurant_map = dict(zip(ckpt.index, ckpt['restaurant_count']))
cvs_map        = dict(zip(ckpt.index, ckpt['cvs_count']))
buffer         = {}
total          = len(grid)

for i, (_, row) in enumerate(grid.iterrows()):
    gid = int(row['grid_id'])
    if gid in done_ids:
        continue

    if i % 50 == 0:
        print(f"  진행: {i}/{total} ({i/total*100:.1f}%)")

    r = kakao_count_tiled(row['cx_5179'], row['cy_5179'], 'FD6')
    c = kakao_count_tiled(row['cx_5179'], row['cy_5179'], 'CS2')
    restaurant_map[gid] = r
    cvs_map[gid]        = c
    buffer[gid]         = {'restaurant_count': r, 'cvs_count': c}
    time.sleep(0.1)

    if len(buffer) >= SAVE_EVERY:
        new_rows = pd.DataFrame.from_dict(buffer, orient='index')
        new_rows.index.name = 'grid_id'
        ckpt = pd.concat([ckpt, new_rows])
        ckpt.to_csv(CHECKPOINT)
        buffer.clear()
        print(f"  체크포인트 저장 ({len(ckpt)}개)")

if buffer:
    new_rows = pd.DataFrame.from_dict(buffer, orient='index')
    new_rows.index.name = 'grid_id'
    ckpt = pd.concat([ckpt, new_rows])
    ckpt.to_csv(CHECKPOINT)

grid['restaurant_count'] = grid['grid_id'].map(restaurant_map)
grid['cvs_count']        = grid['grid_id'].map(cvs_map)
grid['demand_total']     = grid['restaurant_count'] + grid['cvs_count']

print(f"\n[수요] 음식점+편의점 분포:")
print(grid[['restaurant_count', 'cvs_count', 'demand_total']].describe().round(1))

# ── 3. 1인가구 데이터 로드 및 격자 결합 ──────────
df = pd.read_csv('1인가구(연령별)_20260427184906.csv', encoding='utf-8-sig', header=None)
df.columns = ['지역1', '지역2', '성별', '합계'] + [f'age_{i}' for i in range(15)]
df = df.iloc[2:].reset_index(drop=True)

gu_data = df[
    (df['성별'] == '계') &
    (df['지역1'] == '합계') &
    (df['지역2'] != '소계')
][['지역2', '합계']].copy()
gu_data.columns = ['GU_NM', 'single_hh']
gu_data['single_hh'] = pd.to_numeric(gu_data['single_hh'].str.replace(',', ''), errors='coerce')
gu_data = gu_data.reset_index(drop=True)

boundary = gpd.read_file('N3A_G0110000/N3A_G0110000.shp')
boundary['GU_CD'] = boundary['BJCD'].str[:5]
gu = boundary[boundary['BJCD'].str.startswith('11')].dissolve(by='GU_CD').reset_index()
gu = gu.to_crs('EPSG:5179')
gu['GU_NM'] = gu['GU_CD'].map(GU_NAMES)
gu = gu.merge(gu_data, on='GU_NM', how='left')

grid_centroids = grid[['grid_id', 'geometry']].copy()
grid_centroids['geometry'] = grid_centroids.geometry.centroid
grid_with_gu = gpd.sjoin(
    grid_centroids,
    gu[['GU_NM', 'single_hh', 'geometry']],
    how='left', predicate='within'
)
grid['single_hh'] = grid_with_gu['single_hh'].reindex(grid.index)

# ── 4. H4 점수 산출 ───────────────────────────────
# 음식점+편의점 밀도 (70%) + 1인가구 수 (30%)
grid['demand_score']  = minmax(clip_iqr(grid['demand_total']))
grid['single_score']  = minmax(clip_iqr(grid['single_hh']))
grid['h4_score']      = 0.6 * grid['demand_score'] + 0.4 * grid['single_score'].fillna(0)

print(f"\n[H4] 점수 분포:")
print(grid['h4_score'].describe().round(3))

# ── 5. 저장 ───────────────────────────────────────
grid.to_file('step4_h4.gpkg', driver='GPKG')
print("\n저장 완료: step4_h4.gpkg")

# ── 6. 시각화 ─────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

boundary_vis = gpd.read_file('N3A_G0110000/N3A_G0110000.shp').to_crs('EPSG:5179')
seoul_gu     = boundary_vis[boundary_vis['BJCD'].str.startswith('11')].copy()
seoul_gu['GU_CD'] = seoul_gu['BJCD'].str[:5]
gu_vis       = seoul_gu.dissolve(by='GU_CD').reset_index()
gu_vis['centroid'] = gu_vis.geometry.centroid
gu_vis['GU_NM']    = gu_vis['GU_CD'].map(GU_NAMES)

def add_gu_labels(ax):
    gu_vis.boundary.plot(ax=ax, color='#333333', linewidth=1.2)
    for _, row in gu_vis.iterrows():
        if pd.notna(row.get('GU_NM')):
            ax.annotate(row['GU_NM'], xy=(row['centroid'].x, row['centroid'].y),
                        ha='center', va='center', fontsize=6, fontweight='bold',
                        color='#111111',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, ec='none'))

fig, axes = plt.subplots(1, 3, figsize=(28, 9))

for ax, col, title in [
    (axes[0], 'demand_score', '음식점·편의점 밀도 점수'),
    (axes[1], 'single_score', '1인가구 비율 점수'),
    (axes[2], 'h4_score',    'H4 배달 수요 점수\n(음식점·편의점 60% + 1인가구 40%)'),
]:
    grid.plot(column=col, cmap='RdYlGn', vmin=0, vmax=1,
              legend=True,
              legend_kwds={'label': f'{col} (0=낮음, 1=높음)', 'shrink': 0.6},
              ax=ax)
    add_gu_labels(ax)
    ax.set_title(title, fontsize=13)
    ax.axis('off')

plt.suptitle('STEP 4 — H4 배달 수요 분석 (카카오 API 2×2 타일링)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('step4_h4.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장 완료: step4_h4.png")
