# 서울시 배달로봇 도입 최적 지역 선정

> 캡스톤 디자인 프로젝트 | 2026.04  
> GIS 공간 분석 기반 다기준 입지 평가 파이프라인

---

## 프로젝트 개요

2024년 배달로봇 전국 보도 통행이 법적으로 허가되었으나, 실제 운영은 강남권에 집중되어 있습니다.  
본 프로젝트는 서울시 전체를 **500m × 500m 격자** 단위로 분석하여, **다음 확장 후보 지역을 데이터 기반으로 선정**하는 것을 목표로 합니다.

| 항목 | 내용 |
|---|---|
| 분석 대상 | 서울시 전체 (25개 자치구) |
| 격자 단위 | 500m × 500m |
| 좌표계 | EPSG:5179 (한국 표준) |
| 주요 기법 | GIS 공간 분석, 다기준 점수화, 클러스터링 |

---

## 가설 및 평가 지표

| 가설 | 내용 | 가중치 |
|---|---|---|
| H1 — 지형 조건 | 경사도 5% 이하 구간 비율이 높을수록 적합 | 35% |
| H2 — 보행 네트워크 | 교차로 밀도 낮고 경로 연결성 높을수록 적합 | 25% |
| H3 — 횡단 환경 | 횡단보도·신호등 밀도가 적정할수록 적합 | 20% |
| H4 — 배달 수요 | 음식점·편의점 밀도가 높을수록 적합 | 20% |

경사도 분류 기준: **최적 ≤ 5% / 허용 5~8% / 불가 > 8%** (로봇 하드웨어 등판 한계 기준)

---

## 분석 파이프라인

```
STEP 1  전처리        → 500m 격자 생성, 공간 조인, 정규화
STEP 2  경사도 분석   → 표고점 KNN 기반 경사도 계산 및 H1 점수화
STEP 3  보행 환경     → 교차로 밀도, 연결성, 횡단보도, 신호등 지표 산출
STEP 4  점수화        → 가중 합산으로 격자별 적합도 지수 산출
STEP 5  클러스터링    → HDBSCAN / K-Means로 후보 구 Top 5 선정
STEP 6  검증         → 실증 운행 지역 대조, Spearman 상관분석, 로드뷰 육안 검증
STEP 7  시각화        → 서울시 히트맵, 후보 구 레이더 차트
```

---

## 데이터 출처

| 데이터셋 | 가설 | 출처 |
|---|---|---|
| 서울시 표고점 (shp) | H1 | [서울 열린데이터광장](https://data.seoul.go.kr) |
| 서울시 도보 네트워크 | H2 | [서울 열린데이터광장](https://data.seoul.go.kr) |
| 서울시 횡단보도 위치정보 | H3 | [서울 열린데이터광장](https://data.seoul.go.kr) |
| 서울시 보행자 신호등 분포 | H3 | [서울 열린데이터광장](https://data.seoul.go.kr) |
| 서울시 보행자 출입구 정보 | H3 | [서울 열린데이터광장](https://data.seoul.go.kr) |
| 음식점·편의점 인허가 데이터 | H4 | [행안부 localdata.go.kr](https://www.localdata.go.kr) |

---

## 프로젝트 구조

```
Capstone/
├── preparing/
│   ├── Slope_step1.py        # STEP 2: 경사도 분석 및 H1 점수화
│   ├── network_step2.py      # STEP 3: 보행 네트워크 분석
│   ├── step4.py              # STEP 4: 다기준 가중 합산
│   ├── 1인가구.py             # 1인 가구 수 분석 (보조 지표)
│   ├── grid_slope.gpkg       # 경사도 분석 결과 (격자)
│   ├── grid_h1h2h3.gpkg      # H1~H3 점수 통합 결과
│   ├── grid_h1h2h3h4.gpkg    # H1~H4 점수 통합 결과 (최종)
│   ├── slope_map.png         # 경사도 시각화
│   └── grid_slope_map.png    # 격자 경사도 적합도 지도
└── README.md
```

---

## 기술 스택

- **공간 분석**: `geopandas`, `shapely`
- **수치 계산**: `numpy`, `scipy` (cKDTree, KNN)
- **시각화**: `matplotlib`, `folium`

---

## 설치 및 실행

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install geopandas shapely numpy scipy matplotlib folium

# 분석 실행 (순서대로)
cd preparing
python Slope_step1.py     # STEP 2: 경사도
python network_step2.py   # STEP 3: 보행 환경
python step4.py           # STEP 4: 점수 합산
```

> 데이터 파일(shp, csv)은 저작권 이슈로 저장소에 포함되지 않습니다.  
> 위 데이터 출처에서 직접 다운로드 후 `preparing/` 하위에 위치시켜 주세요.

---

## 기대효과

| 대상 | 기대효과 |
|---|---|
| 민간 | 로봇 배달 시범 지역 확대 시 우선순위 근거 제공 |
| 공공 | 스마트 모빌리티 정책 수립·보도 정비 우선순위 지원 |
| 학술 | 도시 환경 × 배달 수요 결합 입지 분석 방법론 제시 |

---

## 참고 문헌

- [과기정통부 — 자율주행 배달로봇 운행 지역 전국 보도로 확대](https://idsn.co.kr/news/view/1065590572178971)
- [뉴빌리티 — 2025년 상용화 성과 공개](https://www.neubility.co.kr/ko/discover/detail?id=152)
- [로봇신문 — 배민 배달로봇 차세대 모델 운행안전인증 획득](https://www.irobotnews.com/news/articleView.html?idxno=40703)
- [국가공간정보포털 — 수치표고모델(DEM)](https://www.nsdi.go.kr)
- [국가법령정보센터 — 지능형 로봇 개발 및 보급 촉진법 제40조의2](https://www.law.go.kr/LSW//lsInfoP.do?lsiSeq=276607)
