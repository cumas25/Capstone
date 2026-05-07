import pandas as pd
import geopandas as gpd
import numpy as np

# ── 1인 가구 데이터 정리 ──────────────────────────
df = pd.read_csv('1인가구(연령별)_20260427184906.csv', encoding='utf-8-sig', header=None)

# 실제 데이터는 2행부터, 컬럼명 직접 지정
df.columns = ['지역1', '지역2', '성별', '합계'] + [f'age_{i}' for i in range(15)]
df = df.iloc[2:].reset_index(drop=True)  # 헤더 2줄 제거

# 구별 합계만 추출 (성별 = '계', 지역2 = 소계 제외)
gu_data = df[
    (df['성별'] == '계') &
    (df['지역1'] == '합계') &
    (df['지역2'] != '소계')
][['지역2', '합계']].copy()

gu_data.columns = ['GU_NM', 'single_hh']
gu_data['single_hh'] = pd.to_numeric(gu_data['single_hh'].str.replace(',', ''), errors='coerce')
gu_data = gu_data.reset_index(drop=True)
print(gu_data)

# ── 구 경계에 조인 ────────────────────────────────
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
gu['GU_NM'] = gu['GU_CD'].map(gu_names)

# 구 이름으로 조인
gu = gu.merge(gu_data, on='GU_NM', how='left')
print(gu[['GU_NM', 'single_hh']])

# ── 격자에 구 단위 1인 가구 수 붙이기 ────────────
grid = gpd.read_file('grid_h1h2h3.gpkg')
grid_with_gu = gpd.sjoin(grid, gu[['GU_NM', 'single_hh', 'geometry']],
                          how='left', predicate='within')
grid['single_hh'] = grid_with_gu['single_hh']

print("\n1인 가구 격자 결합 결과:")
print(grid['single_hh'].describe())