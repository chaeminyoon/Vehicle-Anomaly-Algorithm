# 개선 실험 ⑦: 하단 중앙점(bottom-center) 입력 + 입력 개선 재검증 (누적 ablation)
# - 같은 YOLO 추적 실행에서 저장한 [박스 중심 / 하단 중앙점] 두 좌표의 짝비교(paired)
#   (호모그래피는 도로 평면 위 점에만 유효 -> 차량 높이만큼 뜬 박스 중심은 계통 오차)
# - A0 중심점(원본) / A1 중심점+스무딩 / A2 하단중앙점+스무딩
#   / A3 +원거리 절단·완전 사영보정 / A4 +국소 분산 정규화
# - (④) 완전 보정 좌표에서 차선 간격 = 3.25m로 고정해 미터/km/h 스케일 부여 보고
# 데이터: 1_trajectory_extraction/trajectory_yolo8_bottomcenter.py 재추출 CSV
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d, gaussian_filter

np.random.seed(42)

OUT = "/Volumes/T7/1. 논문정리/github/analysis_results"
DATA_DIR = f"{OUT}/bottom_center"
LOCATIONS = [11, 14, 17]
SEQ_LEN = 50
N_PER_TYPE = 6
FPS = 20.0
LANE_WIDTH_M = 3.25  # 국내 도로 차로폭 표준 3.0~3.5m의 중앙값

# ---------- 데이터 로드: 좌표 변형(중심/하단) 두 벌을 같은 트랙 키로 구축 ----------
tracks = {'center': {}, 'bottom': {}}
for loc in LOCATIONS:
    df = pd.read_csv(f"{DATA_DIR}/track_history_bc_{loc}.csv")
    for tid, g in df.groupby('TrackID'):
        c = g[['X_center', 'Y_center']].values.astype(float)
        b = g[['X_bottom', 'Y_bottom']].values.astype(float)
        if len(c) >= SEQ_LEN + 10:
            tracks['center'][(loc, tid)] = c
            tracks['bottom'][(loc, tid)] = b

keys = sorted(tracks['center'].keys())
rng = np.random.RandomState(42)
rng.shuffle(keys)
n_test = int(len(keys) * 0.2)
test_keys, train_keys = keys[:n_test], keys[n_test:]

loc_range = {loc: max(a[:, 0].max() for k, a in tracks['center'].items() if k[0] == loc) -
                  min(a[:, 0].min() for k, a in tracks['center'].items() if k[0] == loc)
             for loc in LOCATIONS}

def lateral_unit(arr):
    d = arr[-1] - arr[0]
    n = np.array([-d[1], d[0]])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else np.array([0.0, 1.0])

def make_wrong_way(arr, scale):
    return arr[::-1].copy()

def make_lane_cross(arr, scale):
    t = np.linspace(-6, 6, len(arr))
    offset = 1 / (1 + np.exp(-t)) * scale * 0.08
    return arr + np.outer(offset, lateral_unit(arr))

def make_sudden_stop(arr, scale):
    a = arr.copy()
    m = len(a) // 2
    dur = max(int(len(a) * 0.3), 15)
    a[m:m+dur] = a[m]
    return a

def make_zigzag(arr, scale):
    t = np.linspace(0, 3 * 2 * np.pi, len(arr))
    offset = np.sin(t) * scale * 0.03
    return arr + np.outer(offset, lateral_unit(arr))

ANOM_TYPES = {'wrong_way': make_wrong_way, 'lane_cross': make_lane_cross,
              'sudden_stop': make_sudden_stop, 'zigzag': make_zigzag}

# 합성 이상: 같은 원본 궤적 선택(동일 시드)에 좌표 변형별로 동일 변형 적용 -> 짝비교 유지
test_by_loc = defaultdict(list)
for k in test_keys:
    test_by_loc[k[0]].append(k)
picks = {}
for loc in LOCATIONS:
    pool = test_by_loc[loc]
    for tname in ANOM_TYPES:
        idx = rng.choice(len(pool), size=min(N_PER_TYPE, len(pool)), replace=False)
        picks[(loc, tname)] = [pool[i] for i in idx]
synth = {}
for coord in ['center', 'bottom']:
    synth[coord] = {}
    for (loc, tname), pks in picks.items():
        fn = ANOM_TYPES[tname]
        for i, k in enumerate(pks):
            synth[coord][(loc, f"{tname}_{i}")] = fn(tracks[coord][k], loc_range[loc])
print(f"트랙 {len(keys)}개 (학습 {len(train_keys)} / 평가정상 {len(test_keys)}) / "
      f"합성이상 {len(synth['center'])}개 x 좌표 2벌")

# ---------- 입력 처리 단계 ----------
def smooth_track(a):
    if len(a) < 9:
        return a.copy()
    w = 11 if len(a) >= 11 else (len(a) if len(a) % 2 == 1 else len(a) - 1)
    return savgol_filter(a, w, 2, axis=0)

def estimate_homography(train_arrs):
    normals, cents = [], []
    for a in train_arrs:
        c = a.mean(axis=0)
        A = a - c
        cov = A.T @ A / len(A)
        eigval, eigvec = np.linalg.eigh(cov)
        if eigval[-1] / (eigval.sum() + 1e-9) < 0.995:
            continue
        v = eigvec[:, -1]
        normals.append(np.array([-v[1], v[0]])); cents.append(c)
    if len(normals) < 10:
        return None
    N, C = np.array(normals), np.array(cents)
    A_mat = np.einsum('ij,ik->jk', N, N)
    b_vec = np.einsum('ij,ij->i', N, C) @ N
    vp = np.linalg.solve(A_mat, b_vec)
    P_all = np.vstack(train_arrs)
    ctr = P_all.mean(axis=0)
    if np.linalg.norm(vp - ctr) < 0.2 * np.linalg.norm(P_all - ctr, axis=1).max():
        return None
    vpc = vp - ctr
    return dict(ctr=ctr, ab=-vpc / (vpc @ vpc))  # 완전 보정 (lambda=1)

W_MIN = 0.3  # 원거리 ROI 기준

def rect_transform(a, h):
    p = a - h['ctr']
    w = p @ h['ab'] + 1.0
    keep = w >= W_MIN
    if keep.sum() >= 15:
        p, w = p[keep], w[keep]
    else:
        w = np.clip(w, W_MIN, None)
    return p / w[:, None]

# ---------- 규칙 기반 파이프라인 (실험 ④~⑥과 동일 구조) ----------
def run_pipeline(coord, use_smooth, use_rect, use_localnorm):
    trk, syn_in = tracks[coord], synth[coord]
    pre = smooth_track if use_smooth else (lambda a: a.copy())
    homos = {}
    if use_rect:
        for loc in LOCATIONS:
            homos[loc] = estimate_homography([pre(trk[k]) for k in train_keys if k[0] == loc])

    def tf_(a, loc):
        a = pre(a)
        if use_rect and homos.get(loc):
            a = rect_transform(a, homos[loc])
        return a

    T = {k: tf_(trk[k], k[0]) for k in trk}
    S = {sk: tf_(arr, sk[0]) for sk, arr in syn_in.items()}

    lane_models_, dir_fields_, off_sigma_ = {}, {}, {}
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
        if not centers_list:  # 방향별 궤적이 모두 5개 미만인 소규모 데이터 폴백
            centers_list = [0.0]
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
        dd = np.diff(d); d1, s1 = d[1:], s[1:]
        li = np.argmin(np.abs(d1[:, None] - lm['centers'][None, :]), axis=1)
        off = (d1 - lm['centers'][li]) / lm['spacing']
        ds = np.diff(s)
        return off, np.hypot(ds, dd) / lm['spacing'], s1

    if use_localnorm:
        for loc in LOCATIONS:
            offs, ss = [], []
            for k in train_keys:
                if k[0] != loc:
                    continue
                o, _, s1 = lane_feats(T[k], loc)
                offs.append(o); ss.append(s1)
            offs, ss = np.concatenate(offs), np.concatenate(ss)
            qs = np.quantile(ss, np.linspace(0, 1, 9))
            qs[0] -= 1; qs[-1] += 1
            sig = np.array([max(offs[(ss >= qs[i]) & (ss < qs[i + 1])].std(), 1e-3)
                            for i in range(8)])
            sig = np.maximum(sig, 0.3 * np.median(sig))
            off_sigma_[loc] = (qs, sig)

    def norm_off(off, s1, loc):
        if not use_localnorm:
            return off
        qs, sig = off_sigma_[loc]
        bi = np.clip(np.searchsorted(qs, s1) - 1, 0, 7)
        return off / sig[bi]

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

    def rules(arr, loc):
        off, sp, s1 = lane_feats(arr, loc)
        onm = norm_off(off, s1, loc)
        return np.array([ww_score(arr, loc), np.abs(onm).max(), onm.std(),
                         (sp < speed_q_[loc]).mean()])

    tr_rules = defaultdict(list)
    for k in train_keys:
        tr_rules[k[0]].append(rules(T[k], k[0]))
    stats = {loc: (np.mean(tr_rules[loc], axis=0), np.std(tr_rules[loc], axis=0) + 1e-9)
             for loc in LOCATIONS}

    def score(arr, loc):
        z = (rules(arr, loc) - stats[loc][0]) / stats[loc][1]
        return float(z.max())

    thr = {loc: np.percentile([score(T[k], loc) for k in train_keys if k[0] == loc], 95)
           for loc in LOCATIONS}

    y_true, y_score, meta = [], [], []
    for k in test_keys:
        y_true.append(0); y_score.append(score(T[k], k[0])); meta.append(('normal', k[0]))
    for (loc, name), arr in S.items():
        y_true.append(1); y_score.append(score(arr, loc)); meta.append((name.rsplit('_', 1)[0], loc))
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
                lane_models=lane_models_, T=T, lane_feats=lane_feats)

# ---------- 누적 ablation ----------
CONFIGS = [
    ('A0 center raw',        'center', False, False, False),
    ('A1 center +smooth',    'center', True,  False, False),
    ('A2 bottom +smooth',    'bottom', True,  False, False),
    ('A3 +ROI/full-rectify', 'bottom', True,  True,  False),
    ('A4 +local-var norm',   'bottom', True,  True,  True),
]
res = {}
for label, coord, sm, rc, ln in CONFIGS:
    r = run_pipeline(coord, sm, rc, ln)
    res[label] = r
    print(f"{label}: PR-AUC {r['ap']:.3f} | P {r['prec']:.2f} / R {r['rec']:.2f} / F1 {r['f1']:.2f} | " +
          " ".join(f"{t}:{v:.0%}" for t, v in r['type_recall'].items()))

# ---------- (④) 차로폭 기반 미터 스케일: 완전 보정 좌표에서 차선 간격 = 3.25m ----------
# 주의: 소실점 단독 보정은 차로 "평행화"까지만 보장 (사영 -> 아핀). 아핀 자유도가 남아
# 종방향(진행 방향) 스케일은 미결정이므로, 차로폭 캘리브레이션으로 물리량이 되는 것은
# "횡방향" 오프셋(m)뿐. 속도 km/h 환산은 종방향 기준(차선 점선 주기, 위성사진 실측
# 호모그래피 등)이 추가로 필요 -> 아래 속도값은 참고용 명목치.
print("\n[미터 스케일 보고 — A3(완전 보정) 좌표, 차선 간격 = 3.25m 가정]")
r3 = res['A3 +ROI/full-rectify']
for loc in LOCATIONS:
    lm = r3['lane_models'][loc]
    m_per_unit = LANE_WIDTH_M / lm['spacing']
    offs, sps = [], []
    for k in train_keys:
        if k[0] != loc:
            continue
        o, sp, _ = r3['lane_feats'](r3['T'][k], loc)
        offs.append(o); sps.append(sp)
    offs, sps = np.concatenate(offs), np.concatenate(sps)
    kmh = np.median(sps) * LANE_WIDTH_M * FPS * 3.6
    print(f"지점 {loc}: 차선간격 {lm['spacing']:.1f}unit -> {m_per_unit:.3f} m/unit | "
          f"정상 오프셋 std(횡방향, 유효) {offs.std() * LANE_WIDTH_M:.2f} m | "
          f"중앙값 속도(명목치, 종방향 스케일 미결정) {kmh:.0f} km/h")
print("-> 속도의 물리 환산은 종방향 캘리브레이션(점선 주기/실측 호모그래피) 필요")

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.16
colors = ['gray', 'seagreen', 'darkorange', 'royalblue', 'purple']
for i, (label, *_) in enumerate(CONFIGS):
    axs[0].bar(x + (i - 2) * w, [res[label]['type_recall'][t] for t in types], w,
               color=colors[i], label=label)
axs[0].set_xticks(x); axs[0].set_xticklabels(types, rotation=12)
axs[0].set_ylim(0, 1.05); axs[0].set_ylabel('Recall (calibrated thr)')
axs[0].set_title('Detection rate by anomaly type (cumulative steps, paired tracks)')
axs[0].legend(fontsize=8)

labels = [c[0] for c in CONFIGS]
axs[1].plot(labels, [res[l]['f1'] for l in labels], 'o-', color='darkorange', label='F1')
axs[1].plot(labels, [res[l]['ap'] for l in labels], 's-', color='royalblue', label='PR-AUC')
axs[1].plot(labels, [res[l]['rec'] for l in labels], '^-', color='seagreen', label='Recall')
axs[1].set_ylim(0, 1); axs[1].grid(alpha=0.3); axs[1].legend()
axs[1].set_title('Overall metrics per cumulative step')
plt.setp(axs[1].get_xticklabels(), rotation=10, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/13_bottom_center_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 13_bottom_center_eval.png")
print("BC_OK")
