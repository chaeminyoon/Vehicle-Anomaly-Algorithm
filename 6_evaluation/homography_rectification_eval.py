# 개선 실험 ⑤: 소실점 기반 자동 원근 보정(호모그래피) -> 차로횡단/지그재그 탐지 개선
# - 학습 궤적의 직선 성분들로 소실점(VP)을 최소제곱 추정
# - VP를 무한원점으로 보내는 사영변환 H로 좌표 보정 (차로 평행화)
# - 보정 전/후 좌표계에서 동일한 규칙 기반 파이프라인을 재구축해 비교
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# ---------- 데이터/합성셋 준비 (실험 ③④와 동일 시드) ----------
exec(open('lane_relative_ae_eval.py').read().split("# ---------- 차로 모델")[0]
     .replace("import tensorflow as tf", "")
     .replace("from tensorflow.keras.models import Sequential", "")
     .replace("from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense", "")
     .replace("tf.random.set_seed(42)", ""))

# ---------- 소실점 추정 + 사영변환 ----------
homographies = {}
for loc in LOCATIONS:
    tkeys = [k for k in train_keys if k[0] == loc]
    # 각 궤적을 직선으로 근사 (충분히 직선인 것만: 주성분 설명력 > 0.995)
    normals, offsets, cents = [], [], []
    for k in tkeys:
        a = tracks[k]
        c = a.mean(axis=0)
        A = a - c
        cov = A.T @ A / len(A)
        eigval, eigvec = np.linalg.eigh(cov)
        if eigval[-1] / (eigval.sum() + 1e-9) < 0.995:
            continue
        v = eigvec[:, -1]
        n = np.array([-v[1], v[0]])
        normals.append(n); cents.append(c)
    if len(normals) < 10:
        homographies[loc] = None
        continue
    N = np.array(normals); C = np.array(cents)
    # VP: sum ((x-c_i)·n_i)^2 최소화
    A_mat = np.einsum('ij,ik->jk', N, N)
    b_vec = np.einsum('ij,ij->i', N, C) @ N
    vp = np.linalg.solve(A_mat, b_vec)
    # 잔차: 각 직선까지 VP의 수직거리
    res = np.abs(np.einsum('ij,ij->i', N, vp[None, :] - C))
    P_all = np.vstack([tracks[k] for k in tkeys])
    ctr = P_all.mean(axis=0)
    span = np.linalg.norm(P_all - ctr, axis=1).max()
    vp_dist = np.linalg.norm(vp - ctr)
    print(f"지점 {loc}: VP=({vp[0]:.0f},{vp[1]:.0f}), 데이터중심 거리 {vp_dist:.0f}px "
          f"(span {span:.0f}px), 직선궤적 {len(normals)}개, 잔차중앙값 {np.median(res):.0f}px")
    if vp_dist < span * 0.2:  # VP가 데이터 한복판이면 퇴화 -> 보정 생략
        homographies[loc] = None
        continue
    vpc = vp - ctr
    ab_full = -vpc / (vpc @ vpc)  # a*vx+b*vy+1=0 -> VP를 무한원점으로 (완전 보정)
    # 부분 보정: 데이터가 소실점(w=0)에 근접하면 발산하므로,
    # 학습 데이터의 최소 w가 0.3이 되도록 사영 강도 lambda를 축소
    w_all = (P_all - ctr) @ ab_full + 1.0
    w_min = w_all.min()
    lam = 1.0 if w_min >= 0.3 else (1.0 - 0.3) / (1.0 - w_min)
    homographies[loc] = dict(ctr=ctr, ab=ab_full * lam)
    print(f"        w_min={w_min:.2f} -> lambda={lam:.2f} ({'완전' if lam == 1 else '부분'} 보정)")

def rectify(arr, loc):
    h = homographies[loc]
    if h is None:
        return arr.copy()
    p = arr - h['ctr']
    w = p @ h['ab'] + 1.0
    w = np.clip(w, 0.08, None)  # 소실점 근접 발산 방지
    return p / w[:, None]

# ---------- 규칙 기반 파이프라인 (좌표계를 인자로 받아 재구축) ----------
def build_and_eval(space):
    tf_ = (lambda a, l: rectify(a, l)) if space == 'rect' else (lambda a, l: a.copy())
    T = {k: tf_(tracks[k], k[0]) for k in tracks}
    S = {sk: tf_(arr, sk[0]) for sk, arr in synth.items()}

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
            smooth = gaussian_filter1d(hist.astype(float), 2)
            bin_w = edges[1] - edges[0]
            min_sep = 0.02 * (d_dir.max() - d_dir.min())
            peaks, _ = find_peaks(smooth, distance=max(int(min_sep / bin_w), 1),
                                  height=smooth.max() * 0.05)
            centers_list.extend((edges[peaks] + edges[peaks + 1]) / 2)
            dir_list.extend([sgn] * len(peaks))
        order = np.argsort(centers_list)
        centers = np.array(centers_list)[order]
        lane_dir = np.array(dir_list)[order]
        spacing = np.median(np.diff(centers)) if len(centers) > 1 else \
            np.concatenate([(T[k] - center) @ v for k in tkeys]).std()
        lane_models_[loc] = dict(center=center, u=u, v=v, centers=centers,
                                 spacing=spacing, lane_dir=lane_dir)

        # 2D 방향장
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
        d = (arr - lm['center']) @ lm['v']
        dd = np.diff(d); d1 = d[1:]
        li = np.argmin(np.abs(d1[:, None] - lm['centers'][None, :]), axis=1)
        off = (d1 - lm['centers'][li]) / lm['spacing']
        ds = np.diff((arr - lm['center']) @ lm['u'])
        speed = np.hypot(ds, dd)
        return off, speed / lm['spacing']

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
        # 이동량 기준: 지점별 스케일에 맞춰 속도 하한 적용
        conf = (f['coherence'][iy, ix] > 0.7) & (nrm > np.percentile(nrm, 20))
        return -align[conf].mean() if conf.any() else 0.0

    speed_q_ = {}
    for loc in LOCATIONS:
        sp = np.concatenate([lane_feats(T[k], loc)[1] for k in train_keys if k[0] == loc])
        speed_q_[loc] = np.percentile(sp, 5)

    def rules(arr, loc):
        off, sp = lane_feats(arr, loc)
        return np.array([ww_score(arr, loc), np.abs(off).max(), off.std(),
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
    pred = np.array([s > thr[m[1]] for s, m in zip(y_score, meta)])
    tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0)))
    fn_ = int(np.sum(~pred & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    type_recall = {t: float(np.mean(pred[[i for i, m in enumerate(meta) if m[0] == t]]))
                   for t in ANOM_TYPES}
    return dict(ap=ap, prec=prec, rec=rec, f1=f1, type_recall=type_recall, T=T)

res = {}
for space, label in [('image', '원본 이미지 좌표'), ('rect', '소실점 보정 좌표')]:
    r = build_and_eval(space)
    res[space] = r
    print(f"\n[{label}] PR-AUC {r['ap']:.3f} | P {r['prec']:.2f} / R {r['rec']:.2f} / F1 {r['f1']:.2f}")
    print("  유형별:", {t: f"{v:.0%}" for t, v in r['type_recall'].items()})

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
loc = 11
for k in [kk for kk in train_keys if kk[0] == loc][:80]:
    axs[0].plot(tracks[k][:, 0], tracks[k][:, 1], color='gray', alpha=0.3, linewidth=0.6)
axs[0].invert_yaxis(); axs[0].set_title(f'Location {loc}: original image coords')
for k in [kk for kk in train_keys if kk[0] == loc][:80]:
    r_ = rectify(tracks[k], loc)
    axs[1].plot(r_[:, 0], r_[:, 1], color='steelblue', alpha=0.3, linewidth=0.6)
axs[1].invert_yaxis(); axs[1].set_title('Vanishing-point rectified (lanes parallelized)')

types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.35
axs[2].bar(x - w/2, [res['image']['type_recall'][t] for t in types], w,
           color='darkorange', label=f"Image coords (F1={res['image']['f1']:.2f})")
axs[2].bar(x + w/2, [res['rect']['type_recall'][t] for t in types], w,
           color='purple', label=f"Rectified (F1={res['rect']['f1']:.2f})")
axs[2].set_xticks(x); axs[2].set_xticklabels(types, rotation=15)
axs[2].set_ylim(0, 1.05); axs[2].set_ylabel('Recall'); axs[2].legend()
axs[2].set_title('Detection by type: image vs rectified')
plt.tight_layout()
plt.savefig("/Volumes/T7/1. 논문정리/github/analysis_results/11_homography_eval.png",
            dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 11_homography_eval.png")
print("HOMO_OK")
