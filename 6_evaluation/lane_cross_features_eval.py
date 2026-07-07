# 개선 실험 ⑩: 차로횡단 전용 특징 — 최대 병목(확대셋 탐지율 11%) 공략
# 진단: 기존 오프셋 특징은 "최근접 차선 중심 거리"라 0.5차로에서 접힘(fold) —
#       여러 차로를 가로질러 새 차선에 정착하면 오프셋이 다시 0이 되어 신호 소실.
# 추가 특징 (접히지 않는 횡이동 량):
#   net_shift — 시작<->끝 횡위치 순변화량(차로 단위): 몇 차로를 건넜는가
#   n_cross   — 중앙값 필터된 차선 인덱스의 전이 횟수: 차선 경계 통과 횟수
#   osc       — 횡방향 2차 차분 에너지(지그재그 겨냥 보너스)
# 구성: A2(이미지 좌표 + 하단중앙점 + 스무딩) 고정, 특징만 누적 추가 (D0~D3)
# 평가: 확대 평가셋 (지점·유형당 24개, 시드 43 — 실험 ⑨와 동일)
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# ---------- 데이터/전처리 준비 (실험 ⑦ 스크립트 재사용: 이미지 좌표만 필요) ----------
exec(open('bottom_center_eval.py').read().split("# ---------- 규칙 기반 파이프라인")[0])

# ---------- 확대 평가셋 재생성 (실험 ⑨와 동일 시드 43) ----------
N_PER_TYPE = 24
rng2 = np.random.RandomState(43)
picks = {}
for loc in LOCATIONS:
    pool = test_by_loc[loc]
    for tname in ANOM_TYPES:
        idx = rng2.choice(len(pool), size=min(N_PER_TYPE, len(pool)), replace=False)
        picks[(loc, tname)] = [pool[i] for i in idx]
synth_big = {}
for (loc, tname), pks in picks.items():
    fn = ANOM_TYPES[tname]
    for i, k in enumerate(pks):
        synth_big[(loc, f"{tname}_{i}")] = fn(tracks['bottom'][k], loc_range[loc])
print(f"확대 평가셋: 합성이상 {len(synth_big)}개 (지점·유형당 최대 {N_PER_TYPE})")

# ---------- 도로 모델 (A2와 동일: 이미지 좌표 + 스무딩) ----------
T = {k: smooth_track(tracks['bottom'][k]) for k in tracks['bottom']}
S = {sk: smooth_track(arr) for sk, arr in synth_big.items()}

lane_models_, dir_fields_ = {}, {}
for loc in LOCATIONS:
    tkeys = [k for k in train_keys if k[0] == loc]
    P = np.vstack([T[k] for k in tkeys])
    center = P.mean(axis=0)
    cov = np.cov((P - center).T)
    _, eigvec = np.linalg.eigh(cov)
    u = eigvec[:, -1]; v = np.array([-u[1], u[0]])
    track_sign = {k: np.sign(((T[k] - center) @ u)[-1] - ((T[k] - center) @ u)[0]) for k in tkeys}
    centers_list, dir_list = [], []
    for sgn in [1.0, -1.0]:
        dkeys = [k for k in tkeys if track_sign[k] == sgn]
        if len(dkeys) < 5:
            continue
        d_dir = np.concatenate([(T[k] - center) @ v for k in dkeys])
        hist, edges = np.histogram(d_dir, bins=100)
        smoo = gaussian_filter1d(hist.astype(float), 2)
        bin_w = edges[1] - edges[0]
        min_sep = 0.02 * (d_dir.max() - d_dir.min())
        peaks, _ = find_peaks(smoo, distance=max(int(min_sep / bin_w), 1), height=smoo.max() * 0.05)
        centers_list.extend((edges[peaks] + edges[peaks + 1]) / 2)
        dir_list.extend([sgn] * len(peaks))
    order = np.argsort(centers_list)
    centers = np.array(centers_list)[order]
    spacing = np.median(np.diff(centers)) if len(centers) > 1 else \
        np.concatenate([(T[k] - center) @ v for k in tkeys]).std()
    lane_models_[loc] = dict(center=center, u=u, v=v, centers=centers, spacing=spacing)

    x0, x1 = P[:, 0].min(), P[:, 0].max(); y0, y1 = P[:, 1].min(), P[:, 1].max()
    G = 48; w_ = 10
    vx = np.zeros((G, G)); vy = np.zeros((G, G)); cnt = np.zeros((G, G))
    for k in tkeys:
        a = T[k]
        if len(a) <= w_:
            continue
        disp = a[w_:] - a[:-w_]
        nrm = np.linalg.norm(disp, axis=1) + 1e-9
        unit = disp / nrm[:, None]
        mid = (a[w_:] + a[:-w_]) / 2
        ix = np.clip(((mid[:, 0] - x0) / (x1 - x0 + 1e-9) * G).astype(int), 0, G - 1)
        iy = np.clip(((mid[:, 1] - y0) / (y1 - y0 + 1e-9) * G).astype(int), 0, G - 1)
        np.add.at(vx, (iy, ix), unit[:, 0]); np.add.at(vy, (iy, ix), unit[:, 1])
        np.add.at(cnt, (iy, ix), 1)
    vx, vy, cnt = gaussian_filter(vx, 1.5), gaussian_filter(vy, 1.5), gaussian_filter(cnt, 1.5)
    mag = np.hypot(vx, vy)
    dir_fields_[loc] = dict(x0=x0, x1=x1, y0=y0, y1=y1, G=G,
                            ux=np.where(mag > 0, vx / (mag + 1e-9), 0),
                            uy=np.where(mag > 0, vy / (mag + 1e-9), 0),
                            coherence=mag / (cnt + 1e-9))

def lane_feats(arr, loc):
    lm = lane_models_[loc]
    s = (arr - lm['center']) @ lm['u']
    d = (arr - lm['center']) @ lm['v']
    dd = np.diff(d); d1 = d[1:]
    li = np.argmin(np.abs(d1[:, None] - lm['centers'][None, :]), axis=1)
    off = (d1 - lm['centers'][li]) / lm['spacing']
    ds = np.diff(s)
    return off, np.hypot(ds, dd) / lm['spacing'], d1 / lm['spacing'], li

def ww_score(arr, loc, w=10):
    f = dir_fields_[loc]; G = f['G']
    if len(arr) <= w:
        return 0.0
    disp = arr[w:] - arr[:-w]
    nrm = np.linalg.norm(disp, axis=1) + 1e-9
    mid = (arr[w:] + arr[:-w]) / 2
    ix = np.clip(((mid[:, 0] - f['x0']) / (f['x1'] - f['x0'] + 1e-9) * G).astype(int), 0, G - 1)
    iy = np.clip(((mid[:, 1] - f['y0']) / (f['y1'] - f['y0'] + 1e-9) * G).astype(int), 0, G - 1)
    align = (disp[:, 0] * f['ux'][iy, ix] + disp[:, 1] * f['uy'][iy, ix]) / nrm
    conf = (f['coherence'][iy, ix] > 0.7) & (nrm > np.percentile(nrm, 20))
    return -align[conf].mean() if conf.any() else 0.0

speed_q_ = {loc: np.percentile(np.concatenate(
    [lane_feats(T[k], loc)[1] for k in train_keys if k[0] == loc]), 5) for loc in LOCATIONS}

# ---------- 확장 특징 ----------
def net_shift(dl):
    # 시작/끝 10프레임 중앙값 차 (차로 단위) — 최근접 접힘 없이 "몇 차로 건넜나"
    # 한계: 직선 PCA 축 기준이라 곡선·원근 기하 드리프트가 섞임 (진단용으로 유지)
    n = max(min(10, len(dl) // 4), 1)
    return abs(np.median(dl[-n:]) - np.median(dl[:n]))

def n_cross(li):
    # 중앙값 필터로 순간 잡음 제거 후 차선 인덱스 전이 횟수
    if len(li) < 9:
        return 0.0
    lf = medfilt(li.astype(float), 9)
    return float(np.sum(np.abs(np.diff(lf)) > 0.5))

def cross_flow(arr, loc, w=10):
    # 2D 방향장에 "수직"인 변위 성분의 부호 있는 누적 (차로 단위)
    # 방향장이 곡선 기하를 인코딩하므로, 차선 추종이면 ~0, 가로지르면 누적 —
    # net_shift의 곡률 교란(직선 축 기준 드리프트)에 면역
    f = dir_fields_[loc]; G = f['G']
    if len(arr) <= w:
        return 0.0
    disp = arr[w:] - arr[:-w]
    mid = (arr[w:] + arr[:-w]) / 2
    ix = np.clip(((mid[:, 0] - f['x0']) / (f['x1'] - f['x0'] + 1e-9) * G).astype(int), 0, G - 1)
    iy = np.clip(((mid[:, 1] - f['y0']) / (f['y1'] - f['y0'] + 1e-9) * G).astype(int), 0, G - 1)
    # 국소 기대 진행방향의 수직 벡터 n = (-uy, ux)
    perp = -disp[:, 0] * f['uy'][iy, ix] + disp[:, 1] * f['ux'][iy, ix]
    conf = f['coherence'][iy, ix] > 0.7
    if not conf.any():
        return 0.0
    return float(abs(perp[conf].sum()) / w / lane_models_[loc]['spacing'])

def osc_energy(dl):
    # 횡방향 2차 차분 표준편차 (차로 단위) — 고주파 사행(지그재그) 에너지
    if len(dl) < 5:
        return 0.0
    return float(np.std(np.diff(dl, 2)))

FEATURE_SETS = {
    'D0 base (A2 rules)':      [],
    'D1 +net_shift (naive)':   ['net_shift'],
    'D2 +cross_flow':          ['cross_flow'],
    'D3 +cross_flow +osc':     ['cross_flow', 'osc'],
}

def rules(arr, loc, extra):
    off, sp, dl, li = lane_feats(arr, loc)
    r = [ww_score(arr, loc), np.abs(off).max(), off.std(), (sp < speed_q_[loc]).mean()]
    if 'net_shift' in extra:
        r.append(net_shift(dl))
    if 'n_cross' in extra:
        r.append(n_cross(li))
    if 'cross_flow' in extra:
        r.append(cross_flow(arr, loc))
    if 'osc' in extra:
        r.append(osc_energy(dl))
    return np.array(r)

def evaluate(extra):
    tr_rules = defaultdict(list)
    for k in train_keys:
        tr_rules[k[0]].append(rules(T[k], k[0], extra))
    stats = {loc: (np.mean(tr_rules[loc], axis=0), np.std(tr_rules[loc], axis=0) + 1e-9)
             for loc in LOCATIONS}

    def score(arr, loc):
        z = (rules(arr, loc, extra) - stats[loc][0]) / stats[loc][1]
        return float(z.max()), z

    thr = {loc: np.percentile([score(T[k], loc)[0] for k in train_keys if k[0] == loc], 95)
           for loc in LOCATIONS}

    y_true, y_score, meta, z_list = [], [], [], []
    for k in test_keys:
        s, z = score(T[k], k[0])
        y_true.append(0); y_score.append(s); meta.append(('normal', k[0])); z_list.append(z)
    for (loc, name), arr in S.items():
        s, z = score(arr, loc)
        y_true.append(1); y_score.append(s); meta.append((name.rsplit('_', 1)[0], loc)); z_list.append(z)
    y_true, y_score = np.array(y_true), np.array(y_score)
    ap = average_precision_score(y_true, y_score)
    pred = np.array([sc > thr[m[1]] for sc, m in zip(y_score, meta)])
    tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0)))
    fn_ = int(np.sum(~pred & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    type_recall = {t: float(np.mean(pred[[i for i, m in enumerate(meta) if m[0] == t]]))
                   for t in ANOM_TYPES}
    return dict(ap=ap, prec=prec, rec=rec, f1=f1, type_recall=type_recall,
                fp=fp, meta=meta, pred=pred, z=z_list, y_true=y_true)

res = {}
for label, extra in FEATURE_SETS.items():
    r = evaluate(extra)
    res[label] = r
    print(f"{label}: PR-AUC {r['ap']:.3f} | P {r['prec']:.2f} / R {r['rec']:.2f} / F1 {r['f1']:.2f} "
          f"(FP {r['fp']}) | " + " ".join(f"{t}:{v:.0%}" for t, v in r['type_recall'].items()))

# ---------- 특징 분포 진단: net_shift vs cross_flow 분리력 비교 ----------
ns_norm, ns_lc, cf_norm, cf_lc = [], [], [], []
for k in test_keys:
    _, _, dl, _ = lane_feats(T[k], k[0])
    ns_norm.append(net_shift(dl))
    cf_norm.append(cross_flow(T[k], k[0]))
for (loc, name), arr in S.items():
    if name.startswith('lane_cross'):
        _, _, dl, _ = lane_feats(arr, loc)
        ns_lc.append(net_shift(dl))
        cf_lc.append(cross_flow(arr, loc))
ns_norm, ns_lc = np.array(ns_norm), np.array(ns_lc)
cf_norm, cf_lc = np.array(cf_norm), np.array(cf_lc)
print(f"\nnet_shift(차로) — 정상 중앙값 {np.median(ns_norm):.2f} / 95% {np.percentile(ns_norm, 95):.2f}"
      f" | 차로횡단 중앙값 {np.median(ns_lc):.2f}  (곡률 드리프트에 오염)")
print(f"cross_flow(차로) — 정상 중앙값 {np.median(cf_norm):.2f} / 95% {np.percentile(cf_norm, 95):.2f}"
      f" | 차로횡단 중앙값 {np.median(cf_lc):.2f} / 5% {np.percentile(cf_lc, 5):.2f}")

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.2
colors = ['gray', 'seagreen', 'royalblue', 'darkorange']
for i, label in enumerate(FEATURE_SETS):
    axs[0].bar(x + (i - 1.5) * w, [res[label]['type_recall'][t] for t in types], w,
               color=colors[i], label=f"{label} (F1={res[label]['f1']:.2f})")
axs[0].set_xticks(x); axs[0].set_xticklabels(types, rotation=12)
axs[0].set_ylim(0, 1.05); axs[0].set_ylabel('Recall (calibrated thr)')
axs[0].set_title('Cumulative lane-cross features (enlarged eval set)')
axs[0].legend(fontsize=8)

bins = np.linspace(0, max(cf_lc.max(), cf_norm.max()) * 1.05, 40)
axs[1].hist(cf_norm, bins=bins, alpha=0.6, color='gray', label='normal (test)', density=True)
axs[1].hist(cf_lc, bins=bins, alpha=0.6, color='crimson', label='lane_cross (synth)', density=True)
axs[1].set_xlabel('cross-flow accumulation (lane units)'); axs[1].legend()
axs[1].set_title('cross_flow: displacement perpendicular to direction field')

labels = list(FEATURE_SETS)
axs[2].plot(labels, [res[l]['f1'] for l in labels], 'o-', color='darkorange', label='F1')
axs[2].plot(labels, [res[l]['ap'] for l in labels], 's-', color='royalblue', label='PR-AUC')
axs[2].plot(labels, [res[l]['type_recall']['lane_cross'] for l in labels], '^-',
            color='crimson', label='lane_cross recall')
axs[2].plot(labels, [res[l]['type_recall']['zigzag'] for l in labels], 'v-',
            color='purple', label='zigzag recall')
axs[2].set_ylim(0, 1); axs[2].grid(alpha=0.3); axs[2].legend(fontsize=8)
plt.setp(axs[2].get_xticklabels(), rotation=10, fontsize=8)
axs[2].set_title('Overall metrics per cumulative feature')
plt.tight_layout()
plt.savefig(f"{OUT}/16_lane_cross_features_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 16_lane_cross_features_eval.png")
print("LCFEAT_OK")
