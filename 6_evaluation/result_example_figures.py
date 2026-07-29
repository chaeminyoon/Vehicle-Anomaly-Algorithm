# README "결과 예시" 그림 3종 생성 — 실제 지점 11 데이터와 채택 파이프라인(A2+D3+⑪) 산출물 사용
#   docs/images/example_trajectories_lanes.png  궤적 + 추정 차선 중심선 + 2D 방향장 + 횡밀도 히스토그램
#   docs/images/example_anomaly_types.png       합성 이상 4유형 (배경: 실제 정상 궤적)
#   docs/images/example_detection.png           하이브리드 점수 공간에서의 판별 (⑬ 점수 덤프 재사용)
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# 데이터/도로모델 준비 (실험 ⑩ 스크립트의 파이프라인 재사용)
exec(open('lane_cross_features_eval.py').read().split("FEATURE_SETS = {")[0])

NAVY = '#1E3A5F'; STEEL = '#4C7FA8'; TEAL = '#2E8B7A'; GRAY = '#9AA3AE'
CRIMSON = '#C0392B'; AMBER = '#C4762E'; GREEN = '#2F7D5C'
LOC = 11
IMG_DIR = "/Volumes/T7/1. 논문정리/github/vehicle-trajectory-anomaly-detection/docs/images"

lm = lane_models_[LOC]
f = dir_fields_[LOC]
tkeys = [k for k in train_keys if k[0] == LOC]

# ---------- 그림 1: 궤적 + 차선 모델 + 방향장 ----------
fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={'width_ratios': [1.5, 1]})
ax = axs[0]
sign = {k: np.sign(((T[k] - lm['center']) @ lm['u'])[-1] - ((T[k] - lm['center']) @ lm['u'])[0])
        for k in tkeys}
for k in tkeys:
    c = STEEL if sign[k] > 0 else TEAL
    ax.plot(T[k][:, 0], T[k][:, 1], color=c, lw=0.5, alpha=0.16, zorder=1)
s_rng = np.percentile(np.concatenate([(T[k] - lm['center']) @ lm['u'] for k in tkeys]), [2, 98])
ss = np.linspace(s_rng[0], s_rng[1], 50)
for c_off in lm['centers']:
    line = lm['center'] + np.outer(ss, lm['u']) + c_off * lm['v']
    ax.plot(line[:, 0], line[:, 1], '--', color=NAVY, lw=1.3, alpha=0.85, zorder=3)
step = 4
gx = np.linspace(f['x0'], f['x1'], f['G']); gy = np.linspace(f['y0'], f['y1'], f['G'])
GX, GY = np.meshgrid(gx, gy)
mask = f['coherence'] > np.percentile(f['coherence'], 75)
ax.quiver(GX[::step, ::step][mask[::step, ::step]], GY[::step, ::step][mask[::step, ::step]],
          f['ux'][::step, ::step][mask[::step, ::step]], -f['uy'][::step, ::step][mask[::step, ::step]],
          color='#6B7280', scale=38, width=0.0022, alpha=0.75, zorder=2)
ax.invert_yaxis()
ax.set_title(f'Site {LOC} - {len(tkeys)} normal trajectories, estimated lane centerlines (dashed)\n'
             'and the 2D direction field (arrows) used by the wrong-way / cross_flow rules')
ax.set_xlabel('image x (px)'); ax.set_ylabel('image y (px)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=STEEL, lw=2, label='direction A'),
                   Line2D([], [], color=TEAL, lw=2, label='direction B'),
                   Line2D([], [], color=NAVY, lw=1.5, ls='--', label='lane centerline'),
                   Line2D([], [], color='#6B7280', marker=r'$\rightarrow$', ls='', markersize=12,
                          label='direction field')],
          fontsize=8.5, loc='lower right')

ax = axs[1]
d_all = np.concatenate([(T[k] - lm['center']) @ lm['v'] for k in tkeys])
hist, edges = np.histogram(d_all, bins=120)
centers_x = (edges[:-1] + edges[1:]) / 2
smoo = gaussian_filter1d(hist.astype(float), 2)
ax.fill_between(centers_x, smoo, color='#D6E0EC', zorder=1)
ax.plot(centers_x, smoo, color=STEEL, lw=1.5, zorder=2)
for i, c_off in enumerate(lm['centers']):
    ax.axvline(c_off, color=NAVY, ls='--', lw=1.2, alpha=0.85)
ax.set_title('Lateral density histogram along the road axis:\neach peak is a detected lane centerline')
ax.set_xlabel('lateral offset from road center (px)'); ax.set_ylabel('trajectory points')
ax.annotate(f'{len(lm["centers"])} lanes detected\n(spacing {lm["spacing"]:.0f} px)',
            xy=(0.03, 0.94), xycoords='axes fraction', fontsize=9.5, color=NAVY, va='top')
for a in axs:
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/example_trajectories_lanes.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: example_trajectories_lanes.png")

# ---------- 그림 2: 합성 이상 4유형 ----------
rngv = np.random.RandomState(7)
bg_keys = [k for k in test_keys if k[0] == LOC][:45]
base_pool = [k for k in test_keys if k[0] == LOC and len(tracks['bottom'][k]) > 80]
TITLES = {'wrong_way': 'wrong-way: same geometry, reversed time\n(only the direction field can see it)',
          'lane_cross': 'lane-cross: sigmoid lateral drift\nacross lane boundaries',
          'sudden_stop': 'sudden-stop: position frozen\nfor the middle 30%',
          'zigzag': 'zigzag: high-frequency lateral\noscillation (3 cycles)'}
fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.2))
for ax, (tname, fn) in zip(axs.ravel(), ANOM_TYPES.items()):
    for k in bg_keys:
        ax.plot(T[k][:, 0], T[k][:, 1], color=GRAY, lw=0.5, alpha=0.28, zorder=1)
    picks = rngv.choice(len(base_pool), size=3, replace=False)
    for j, pi in enumerate(picks):
        arr = smooth_track(fn(tracks['bottom'][base_pool[pi]], loc_range[LOC]))
        ax.plot(arr[:, 0], arr[:, 1], color=CRIMSON, lw=1.7, alpha=0.95, zorder=3)
        ax.scatter(*arr[0], color=NAVY, s=32, zorder=4, marker='o')
        ax.scatter(*arr[-1], color=CRIMSON, s=44, zorder=4, marker='>')
        idx = np.linspace(8, len(arr) - 9, 5).astype(int)
        d = arr[idx + 6] - arr[idx]
        ax.quiver(arr[idx, 0], arr[idx, 1], d[:, 0], -d[:, 1], color=CRIMSON,
                  scale=None, width=0.004, zorder=4, alpha=0.9)
    ax.invert_yaxis()
    ax.set_title(TITLES[tname], fontsize=10.5, color=NAVY)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#C9CED6')
fig.suptitle(f'Synthetic anomaly types on real site-{LOC} trajectories '
             '(gray: normal traffic - navy dot: start, red arrow: travel direction)',
             fontsize=11.5, color=NAVY, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{IMG_DIR}/example_anomaly_types.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: example_anomaly_types.png")

# ---------- 그림 3: 하이브리드 점수 공간에서의 판별 (⑬ 점수 덤프 재사용) ----------
D = np.load(f"{OUT}/pred_eval_scores.npz", allow_pickle=True)
y, types_, locs_ = D['y'], D['types'], D['locs'].astype(int)
rz, az = D['rz'], D['az']
hm = (rz + az) / 2
thr = {loc: float(np.percentile([(r + a) / 2 for r, a, p in D[f"train_{loc}"]], 95))
       for loc in LOCATIONS}
detected = np.array([s > thr[l] for s, l in zip(hm, locs_)])

CATS = ['normal', 'wrong_way', 'lane_cross', 'sudden_stop', 'zigzag']
fig, ax = plt.subplots(figsize=(12.5, 5.2))
rngj = np.random.RandomState(0)
for i, cat in enumerate(CATS):
    idx = types_ == cat if cat != 'normal' else (y == 0)
    xs = i + rngj.uniform(-0.17, 0.17, idx.sum())
    vals = np.clip(hm[idx], -2, 40)
    det = detected[idx]
    if cat == 'normal':
        ax.scatter(xs[~det], vals[~det], s=13, color=GRAY, alpha=0.55, zorder=2, label='normal (correct)')
        ax.scatter(xs[det], vals[det], s=26, facecolor='none', edgecolor=AMBER, lw=1.2,
                   zorder=3, label='false positive')
    else:
        ax.scatter(xs[det], vals[det], s=15, color=GREEN, alpha=0.8, zorder=2,
                   label='detected' if i == 1 else None)
        ax.scatter(xs[~det], vals[~det], s=26, facecolor='none', edgecolor=CRIMSON, lw=1.2,
                   zorder=3, label='missed' if i == 1 else None)
    if cat != 'normal':
        rate = float(np.mean(detected[idx]))
        ax.text(i, 41.5, f"{rate:.0%}", ha='center', fontsize=10.5, color=NAVY, fontweight='bold')
band = (min(thr.values()), max(thr.values()))
ax.axhspan(band[0], band[1], color=NAVY, alpha=0.10, zorder=1)
ax.axhline(np.mean(list(thr.values())), color=NAVY, ls='--', lw=1.2)
ax.text(-0.42, band[1] + 1.2, 'per-site q95 thresholds\n(calibrated on train normals)',
        fontsize=8.5, color=NAVY, ha='left')
ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS)
ax.set_ylabel('hybrid anomaly score (rule z + AE z mean, clipped at 40)')
ax.set_title('Adopted config (exp.11) on the enlarged benchmark - per-type detection at the q95 operating point\n'
             f'(F1 0.85 - wrong-way and sudden-stop separate cleanly; low-amplitude lane-cross overlaps normal traffic)')
ax.set_ylim(-3, 46)
ax.legend(fontsize=8.5, loc='upper left', ncol=2, bbox_to_anchor=(0.01, 0.88))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.25, axis='y')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/example_detection.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: example_detection.png")
print("EXAMPLES_OK")
