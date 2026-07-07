# 개선 실험 ⑥: 입력 품질 개선 누적 실험 (스무딩 -> ROI+완전보정 -> 위치별 분산 정규화)
# Step0 기준선(원본) / Step1 +Savitzky-Golay 스무딩 / Step2 +원거리 절단·완전 사영보정 / Step3 +국소 분산 정규화
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# ---------- 데이터/합성셋 준비 (기존 실험과 동일 시드) ----------
exec(open('lane_relative_ae_eval.py').read().split("# ---------- 차로 모델")[0]
     .replace("import tensorflow as tf", "")
     .replace("from tensorflow.keras.models import Sequential", "")
     .replace("from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense", "")
     .replace("tf.random.set_seed(42)", ""))

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
    # w >= W_MIN 구간만 사용(원거리 절단) 후 완전 사영보정; 잔여 프레임 부족 시 클리핑 폴백
    p = a - h['ctr']
    w = p @ h['ab'] + 1.0
    keep = w >= W_MIN
    if keep.sum() >= 15:
        p, w = p[keep], w[keep]
    else:
        w = np.clip(w, W_MIN, None)
    return p / w[:, None]

# ---------- 파이프라인 (설정 인자: 스무딩 / 보정 / 국소 분산 정규화) ----------
def run_pipeline(use_smooth, use_rect, use_localnorm):
    pre = smooth_track if use_smooth else (lambda a: a.copy())
    homos = {}
    if use_rect:
        for loc in LOCATIONS:
            homos[loc] = estimate_homography([pre(tracks[k]) for k in train_keys if k[0] == loc])

    def tf_(a, loc):
        a = pre(a)
        if use_rect and homos.get(loc):
            a = rect_transform(a, homos[loc])
        return a

    T = {k: tf_(tracks[k], k[0]) for k in tracks}
    S = {sk: tf_(arr, sk[0]) for sk, arr in synth.items()}

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

    # 국소 분산: 종방향 s 분위 구간별 학습 오프셋 표준편차
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
            sig = np.maximum(sig, 0.3 * np.median(sig))  # 과소 분산 하한
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
    return dict(ap=ap, prec=prec, rec=rec, f1=f1, type_recall=type_recall)

# 플롯 라벨은 폰트 문제를 피해 영문 사용 (Step1~3은 누적 적용)
CONFIGS = [
    ('Step0 baseline (raw)', False, False, False),
    ('Step1 +smoothing', True, False, False),
    ('Step2 +ROI/full-rectify', True, True, False),
    ('Step3 +local-var norm', True, True, True),
]
res = {}
for label, sm, rc, ln in CONFIGS:
    r = run_pipeline(sm, rc, ln)
    res[label] = r
    print(f"{label}: PR-AUC {r['ap']:.3f} | P {r['prec']:.2f} / R {r['rec']:.2f} / F1 {r['f1']:.2f} | " +
          " ".join(f"{t}:{v:.0%}" for t, v in r['type_recall'].items()))

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.2
colors = ['gray', 'seagreen', 'royalblue', 'darkorange']
for i, (label, *_ ) in enumerate(CONFIGS):
    axs[0].bar(x + (i - 1.5) * w, [res[label]['type_recall'][t] for t in types], w,
               color=colors[i], label=label)
axs[0].set_xticks(x); axs[0].set_xticklabels(types, rotation=12)
axs[0].set_ylim(0, 1.05); axs[0].set_ylabel('Recall (calibrated thr)')
axs[0].set_title('Detection rate by anomaly type (cumulative steps)')
axs[0].legend(fontsize=8)

labels = [c[0] for c in CONFIGS]
axs[1].plot(labels, [res[l]['f1'] for l in labels], 'o-', color='darkorange', label='F1')
axs[1].plot(labels, [res[l]['ap'] for l in labels], 's-', color='royalblue', label='PR-AUC')
axs[1].plot(labels, [res[l]['rec'] for l in labels], '^-', color='seagreen', label='Recall')
axs[1].set_ylim(0, 1); axs[1].grid(alpha=0.3); axs[1].legend()
axs[1].set_title('Overall metrics per cumulative step')
plt.setp(axs[1].get_xticklabels(), rotation=10, fontsize=8)
plt.tight_layout()
plt.savefig("/Volumes/T7/1. 논문정리/github/analysis_results/12_input_quality_eval.png",
            dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 12_input_quality_eval.png")
print("INPUTQ_OK")
