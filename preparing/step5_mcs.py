import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN

# MCS 가중치 (H1은 필터, 점수 합산 제외)
W_H2 = 0.25
W_H3 = 0.25
W_H4 = 0.50

GU_NAMES = {
    '11110':'종로구','11140':'중구',    '11170':'용산구', '11200':'성동구',
    '11215':'광진구','11230':'동대문구','11260':'중랑구', '11290':'성북구',
    '11305':'강북구','11320':'도봉구',  '11350':'노원구', '11380':'은평구',
    '11410':'서대문구','11440':'마포구','11470':'양천구', '11500':'강서구',
    '11530':'구로구','11545':'금천구',  '11560':'영등포구','11590':'동작구',
    '11620':'관악구','11650':'서초구',  '11680':'강남구', '11710':'송파구',
    '11740':'강동구'
}

# ── 1. 격자 로드 ───────────────────────────────────
grid = gpd.read_file('step4_h4.gpkg')
print(f"[STEP4] 격자: {len(grid)}개")

# ── 2. MCS 점수 계산 ───────────────────────────────
grid['mcs_score'] = (
    W_H2 * grid['h2_score'].fillna(0) +
    W_H3 * grid['h3_score'].fillna(0) +
    W_H4 * grid['h4_score'].fillna(0)
)

print(f"\n[MCS] 점수 분포:")
print(grid['mcs_score'].describe().round(3))

# ── 3. 5분위 등급 부여 ─────────────────────────────
grid['mcs_grade'] = pd.qcut(
    grid['mcs_score'], q=5,
    labels=['5등급(최저)', '4등급', '3등급', '2등급', '1등급(최고)']
)
print(f"\n[등급 분포]")
print(grid['mcs_grade'].value_counts().sort_index())

# ── 4. 구 경계 및 구별 집계 ───────────────────────
boundary = gpd.read_file('N3A_G0110000/N3A_G0110000.shp').to_crs('EPSG:5179')
seoul_gu = boundary[boundary['BJCD'].str.startswith('11')].copy()
seoul_gu['GU_CD'] = seoul_gu['BJCD'].str[:5]
gu = seoul_gu.dissolve(by='GU_CD').reset_index()
gu['GU_NM'] = gu['GU_CD'].map(GU_NAMES)
gu['centroid'] = gu.geometry.centroid

grid_centroids = grid[['grid_id', 'mcs_score', 'geometry']].copy()
grid_centroids['geometry'] = grid_centroids.geometry.centroid

grid_gu = gpd.sjoin(
    grid_centroids,
    gu[['GU_NM', 'geometry']],
    how='left', predicate='within'
)

top30_thr = grid['mcs_score'].quantile(0.70)
gu_rank = (
    grid_gu
    .groupby('GU_NM')
    .agg(
        grid_count  = ('grid_id',    'count'),
        mcs_mean    = ('mcs_score',  'mean'),
        mcs_max     = ('mcs_score',  'max'),
        top30_count = ('mcs_score',  lambda x: (x >= top30_thr).sum()),
    )
    .reset_index()
)
gu_rank['top30_ratio'] = gu_rank['top30_count'] / gu_rank['grid_count'] * 100
gu_rank = gu_rank.sort_values('mcs_mean', ascending=False).reset_index(drop=True)
gu_rank.index += 1

print(f"\n[구별 MCS 순위 Top 10]")
print(
    gu_rank[['GU_NM', 'grid_count', 'mcs_mean', 'mcs_max', 'top30_ratio']]
    .head(10)
    .rename(columns={
        'GU_NM': '자치구', 'grid_count': '격자수',
        'mcs_mean': 'MCS평균', 'mcs_max': 'MCS최고', 'top30_ratio': 'Top30%비율(%)'
    })
    .to_string(index=True)
)

# ── 5. 공간 클러스터링 (Top 30% 격자 대상 DBSCAN) ─
top_mask  = grid['mcs_score'] >= top30_thr
top_grid  = grid[top_mask].copy().reset_index(drop=True)
coords    = np.column_stack([top_grid['cx_5179'], top_grid['cy_5179']])

db = DBSCAN(eps=1000, min_samples=3).fit(coords)   # 1 km 반경
top_grid['cluster'] = db.labels_

n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise    = (db.labels_ == -1).sum()
print(f"\n[DBSCAN] 클러스터: {n_clusters}개 | 노이즈(고립 격자): {n_noise}개")

# 클러스터별 MCS 평균 → 최소 10개 격자 이상인 클러스터만 상위 5개 추출
cluster_stats = (
    top_grid[top_grid['cluster'] >= 0]
    .groupby('cluster')
    .agg(size=('mcs_score', 'count'), mcs_mean=('mcs_score', 'mean'))
    .query('size >= 10')
    .sort_values('mcs_mean', ascending=False)
    .head(5)
)
print(f"\n[Top 5 클러스터]")
print(cluster_stats.round(3))

top5_ids = cluster_stats.index.tolist()
grid['top5_cluster'] = -1
for rank, cid in enumerate(top5_ids, 1):
    mask = top_grid['cluster'] == cid
    gids = top_grid.loc[mask, 'grid_id'].values
    grid.loc[grid['grid_id'].isin(gids), 'top5_cluster'] = rank

# 클러스터별 주요 자치구 구성 출력
print(f"\n[클러스터별 주요 자치구]")
top_grid_centroids = top_grid[['grid_id', 'cluster', 'mcs_score', 'geometry']].copy()
top_grid_centroids['geometry'] = top_grid_centroids.geometry.centroid
top_grid_gu = gpd.sjoin(
    top_grid_centroids,
    gu[['GU_NM', 'geometry']],
    how='left', predicate='within'
)
for rank, cid in enumerate(top5_ids, 1):
    sub = top_grid_gu[top_grid_gu['cluster'] == cid]
    gu_counts = sub['GU_NM'].value_counts().head(3)
    gu_str = ', '.join([f"{g}({n}개)" for g, n in gu_counts.items()])
    print(f"  클러스터 {rank}위 (ID={cid}, {len(sub)}개): {gu_str}")

# ── 6. 저장 ───────────────────────────────────────
out_cols = [c for c in grid.columns if c != 'mcs_grade'] + ['mcs_grade']
grid.to_file('step5_mcs.gpkg', driver='GPKG')
gu_rank.to_csv('step5_gu_rank.csv', index=True, encoding='utf-8-sig')
print("\n저장 완료: step5_mcs.gpkg | step5_gu_rank.csv")

# ── 7. 시각화 ─────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

gu_vis = gu.copy()

def add_gu_labels(ax):
    gu_vis.boundary.plot(ax=ax, color='#333333', linewidth=1.2)
    for _, row in gu_vis.iterrows():
        if pd.notna(row.get('GU_NM')):
            ax.annotate(row['GU_NM'], xy=(row['centroid'].x, row['centroid'].y),
                        ha='center', va='center', fontsize=6, fontweight='bold',
                        color='#111111',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, ec='none'))

fig, axes = plt.subplots(1, 3, figsize=(24, 9))

# ── 왼쪽: MCS 전체 분포 ──
ax1 = axes[0]
grid.plot(column='mcs_score', cmap='RdYlGn', vmin=0, vmax=1,
          legend=True,
          legend_kwds={'label': 'MCS 점수 (0=낮음, 1=높음)', 'shrink': 0.6},
          ax=ax1)
add_gu_labels(ax1)
ax1.set_title('MCS 종합 적합도\n(H2×0.25 + H3×0.25 + H4×0.50)', fontsize=13)
ax1.axis('off')

# ── 가운데: Top 30% + Top 클러스터 ──
ax2 = axes[1]
cluster_colors = ['#D32F2F', '#F57C00', '#388E3C', '#7B1FA2', '#0288D1']
cluster_labels = ['1위', '2위', '3위', '4위', '5위']

grid[~top_mask].plot(ax=ax2, color='#E0E0E0')
grid[top_mask & (grid['top5_cluster'] < 0)].plot(ax=ax2, color='#90CAF9')
for rank, (color, label) in enumerate(zip(cluster_colors, cluster_labels), 1):
    sub = grid[grid['top5_cluster'] == rank]
    if len(sub):
        sub.plot(ax=ax2, color=color, label=f'클러스터 {label}')

add_gu_labels(ax2)
ax2.set_title(f'Top 30% 격자 (임계값 {top30_thr:.3f})\n+ Top 클러스터', fontsize=13)
ax2.axis('off')

patches = [mpatches.Patch(color='#E0E0E0', label='하위 70%'),
           mpatches.Patch(color='#90CAF9', label='Top 30% (클러스터 외)')]
patches += [mpatches.Patch(color=c, label=f'클러스터 {l}')
            for c, l in zip(cluster_colors, cluster_labels)]
ax2.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.8)

# ── 오른쪽: 구별 MCS 평균 막대그래프 ──
ax3 = axes[2]
top10 = gu_rank.head(10).copy()
colors = ['#D32F2F' if i < 1 else '#F57C00' if i < 3 else '#1565C0'
          for i in range(len(top10))]
bars = ax3.barh(top10['GU_NM'][::-1], top10['mcs_mean'][::-1], color=colors[::-1])
ax3.set_xlabel('MCS 평균 점수', fontsize=11)
ax3.set_title('자치구별 MCS 평균\n(Top 10)', fontsize=13)
ax3.set_xlim(0, top10['mcs_mean'].max() * 1.15)
for bar, val in zip(bars, top10['mcs_mean'][::-1]):
    ax3.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=9)
ax3.spines[['top', 'right']].set_visible(False)

plt.suptitle('STEP 5 — MCS 배달로봇 도입 최적 지역 종합 분석',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('step5_mcs.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장 완료: step5_mcs.png")
