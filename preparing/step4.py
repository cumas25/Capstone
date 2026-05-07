import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt

KAKAO_API_KEY = "1a020a0029ec66458a99b580dd79698a"

GU_NAMES = {
    '11110': '종로구', '11140': '중구',     '11170': '용산구',  '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구',  '11290': '성북구',
    '11305': '강북구', '11320': '도봉구',   '11350': '노원구',  '11380': '은평구',
    '11410': '서대문구','11440': '마포구',  '11470': '양천구',  '11500': '강서구',
    '11530': '구로구', '11545': '금천구',   '11560': '영등포구','11590': '동작구',
    '11620': '관악구', '11650': '서초구',   '11680': '강남구',  '11710': '송파구',
    '11740': '강동구'
}

def clip_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))

def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0.0


# ── 운행가능 격자 불러오기 ────────────────────────
grid = gpd.read_file('grid_h1h2h3.gpkg')
print(f"분석 대상 격자: {len(grid)}개")

# ── 격자 중심점: 5179에서 중심 계산 후 WGS84 변환 ─
centroids_5179 = grid.geometry.centroid
centroids_gdf  = gpd.GeoDataFrame(geometry=centroids_5179, crs='EPSG:5179').to_crs('EPSG:4326')
grid['cx'] = centroids_gdf.geometry.x
grid['cy'] = centroids_gdf.geometry.y

# ── 카카오 로컬 API 호출 함수 ─────────────────────
def kakao_count(lon, lat, category_group_code, radius=350):
    url     = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    total   = 0
    for page in range(1, 4):
        params = {
            "category_group_code": category_group_code,
            "x": lon, "y": lat,
            "radius": radius,
            "page": page,
            "size": 15
        }
        res  = requests.get(url, headers=headers, params=params)
        data = res.json()

        if 'meta' not in data:
            print(f"  API 오류: {data}")
            break

        total += len(data.get('documents', []))
        if data['meta']['is_end']:
            break
        time.sleep(0.05)
    return total

# ── 격자별 음식점(FD6) + 편의점(CS2) 수집 ─────────
restaurant_counts = []
cvs_counts        = []
total             = len(grid)

for i, row in grid.iterrows():
    if i % 100 == 0:
        print(f"진행 중: {i}/{total} ({i/total*100:.1f}%)")

    r = kakao_count(row['cx'], row['cy'], 'FD6')
    c = kakao_count(row['cx'], row['cy'], 'CS2')
    restaurant_counts.append(r)
    cvs_counts.append(c)
    time.sleep(0.1)

grid['restaurant_count'] = restaurant_counts
grid['cvs_count']        = cvs_counts
grid['demand_total']     = grid['restaurant_count'] + grid['cvs_count']

print("\n수요 데이터:")
print(grid[['restaurant_count', 'cvs_count', 'demand_total']].describe())

# ── 음식점+편의점 점수 정규화 (70%) ───────────────
grid['restaurant_score'] = minmax(clip_iqr(grid['demand_total']))

# ── 1인 가구 데이터 로드 및 격자 결합 (30%) ───────
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

grid_with_gu = gpd.sjoin(grid, gu[['GU_NM', 'single_hh', 'geometry']], how='left', predicate='within')
grid['single_hh'] = grid_with_gu['single_hh'].reindex(grid.index)
grid['single_hh_score'] = minmax(clip_iqr(grid['single_hh']))

# ── H4 최종 점수: 음식점+편의점 70% + 1인가구 30% ─
grid['h4_score'] = 0.7 * grid['restaurant_score'] + 0.3 * grid['single_hh_score']

print("\nH4 점수:"); print(grid['h4_score'].describe())

# ── 저장 ─────────────────────────────────────────
grid.to_file('grid_h1h2h3h4.gpkg', driver='GPKG')
print("\n저장 완료: grid_h1h2h3h4.gpkg")

# ── 시각화 ───────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

gu['centroid'] = gu.geometry.centroid

fig, ax = plt.subplots(figsize=(12, 12))
grid.plot(column='h4_score', cmap='RdYlGn', vmin=0, vmax=1,
          legend=True,
          legend_kwds={'label': 'H4 배달 수요 점수 (0=낮음, 1=높음)', 'shrink': 0.6},
          ax=ax)
gu.boundary.plot(ax=ax, color='#222222', linewidth=1.5)
for _, row in gu.iterrows():
    ax.annotate(
        text=row['GU_NM'],
        xy=(row['centroid'].x, row['centroid'].y),
        ha='center', va='center',
        fontsize=7, fontweight='bold', color='#111111',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, ec='none')
    )
ax.set_title('서울시 배달 수요 분포 (음식점+편의점 70% + 1인가구 30%)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('step3_h4_demand.png', dpi=150)
plt.show()
print("저장 완료: step3_h4_demand.png")