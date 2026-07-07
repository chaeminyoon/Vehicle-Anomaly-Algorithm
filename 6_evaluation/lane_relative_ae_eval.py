# 개선 실험 ④: 차로 기준 상대 특징 (lane-relative features)
# - 학습 궤적에서 도로 주축(PCA)·차선 중심(횡방향 밀도 피크)·차로별 통행방향 추정
# - 특징: [차선 대비 횡방향 오프셋, 통행방향 정렬도, 횡방향 속도, 속력]
# - 개선 실험 ③과 동일한 합성 평가셋(동일 시드)으로 기존 최고 구성과 비교
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

np.random.seed(42); tf.random.set_seed(42)

OUT = "/Volumes/T7/1. 논문정리/github/analysis_results"
DATA = "/Volumes/T7/1. 논문정리/trajectory_project/Code/combined_trajectory_data.csv"
LOCATIONS = [11, 14, 17]
SEQ_LEN, STEP = 50, 10
N_PER_TYPE = 6

# ---------- 데이터 준비 (실험 ③과 동일한 시드/절차 -> 동일한 분리·합성셋) ----------
raw = pd.read_csv(DATA)
raw = raw[raw['Location'].isin(LOCATIONS)][['Location', 'TrackID', 'Time', 'X_center', 'Y_center']]
raw = raw.sort_values(['Location', 'Time'])

tracks = {}
for (loc, tid), g in raw.groupby(['Location', 'TrackID']):
    arr = g[['X_center', 'Y_center']].values.astype(float)
    if len(arr) >= SEQ_LEN + 10:
        tracks[(loc, tid)] = arr

keys = sorted(tracks.keys())
rng = np.random.RandomState(42)
rng.shuffle(keys)
n_test = int(len(keys) * 0.2)
test_keys, train_keys = keys[:n_test], keys[n_test:]

loc_range = {loc: raw[raw['Location'] == loc]['X_center'].max() - raw[raw['Location'] == loc]['X_center'].min()
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

synth = {}
test_by_loc = defaultdict(list)
for k in test_keys:
    test_by_loc[k[0]].append(k)
for loc in LOCATIONS:
    pool = test_by_loc[loc]
    for tname, fn in ANOM_TYPES.items():
        picks = rng.choice(len(pool), size=min(N_PER_TYPE, len(pool)), replace=False)
        for i, pi in enumerate(picks):
            synth[(loc, f"{tname}_{i}")] = fn(tracks[pool[pi]], loc_range[loc])
print(f"학습 {len(train_keys)} / 평가정상 {len(test_keys)} / 합성이상 {len(synth)}")

# ---------- 차로 모델: 도로 주축 + 차선 중심 + 차로별 통행방향 (학습 데이터로만 추정) ----------
lane_models = {}
for loc in LOCATIONS:
    tkeys = [k for k in train_keys if k[0] == loc]
    P = np.vstack([tracks[k] for k in tkeys])
    center = P.mean(axis=0)
    cov = np.cov((P - center).T)
    eigval, eigvec = np.linalg.eigh(cov)
    u = eigvec[:, -1]              # 도로 주축 (분산 최대 방향)
    v = np.array([-u[1], u[0]])    # 횡방향

    # 통행 방향별로 궤적을 분리한 뒤, 방향별 횡방향 밀도 히스토그램에서 차선 검출
    # (양방향 궤적이 한 차선에 섞여 배정되면 정렬도 특징이 무력화되므로 방향 분리가 필수)
    track_sign = {k: np.sign(((tracks[k] - center) @ u)[-1] - ((tracks[k] - center) @ u)[0])
                  for k in tkeys}
    centers_list, dir_list = [], []
    for sgn in [1.0, -1.0]:
        dkeys = [k for k in tkeys if track_sign[k] == sgn]
        if len(dkeys) < 5:
            continue
        d_dir = np.concatenate([(tracks[k] - center) @ v for k in dkeys])
        hist, edges = np.histogram(d_dir, bins=100)
        smooth = gaussian_filter1d(hist.astype(float), 2)
        min_sep_px = 15  # 차선 최소 간격(px)
        bin_w = edges[1] - edges[0]
        peaks, _ = find_peaks(smooth, distance=max(int(min_sep_px / bin_w), 1),
                              height=smooth.max() * 0.05)
        centers_list.extend((edges[peaks] + edges[peaks + 1]) / 2)
        dir_list.extend([sgn] * len(peaks))
    order = np.argsort(centers_list)
    centers = np.array(centers_list)[order]
    lane_dir = np.array(dir_list)[order]
    spacing = np.median(np.diff(centers)) if len(centers) > 1 else \
        np.concatenate([(tracks[k] - center) @ v for k in tkeys]).std()

    # 통행방향 프로파일: 횡방향 위치 d -> 기대 진행방향(-1~+1), 커널 평균으로 추정
    # (차선 배정 오류에 강건 -- 역주행 정렬도 계산에 사용)
    d_pts, s_pts = [], []
    for k in tkeys:
        dd_ = (tracks[k] - center) @ v
        d_pts.append(dd_)
        s_pts.append(np.full(len(dd_), track_sign[k]))
    d_pts, s_pts = np.concatenate(d_pts), np.concatenate(s_pts)
    prof_bins = np.linspace(d_pts.min(), d_pts.max(), 100)
    cnt, _ = np.histogram(d_pts, bins=prof_bins)
    sgn_sum, _ = np.histogram(d_pts, bins=prof_bins, weights=s_pts)
    cnt_s = gaussian_filter1d(cnt.astype(float), 2)
    sgn_s = gaussian_filter1d(sgn_sum, 2)
    dir_profile = np.where(cnt_s > 0, sgn_s / (cnt_s + 1e-9), 0.0)
    prof_x = (prof_bins[:-1] + prof_bins[1:]) / 2

    lane_models[loc] = dict(center=center, u=u, v=v, centers=centers,
                            spacing=spacing, lane_dir=lane_dir,
                            prof_x=prof_x, dir_profile=dir_profile)
    n_up = int(np.sum(lane_dir > 0)); n_dn = int(np.sum(lane_dir < 0))
    print(f"지점 {loc}: 차선 {len(centers)}개 (순방향 {n_up} / 역방향 {n_dn}), 간격 {spacing:.1f}px")

def lane_rel_features(arr, lm):
    # (N,2) -> (N-1,4): [횡방향 오프셋/간격, 통행방향 정렬도, 횡방향속도/간격, 속력/간격]
    s = (arr - lm['center']) @ lm['u']
    d = (arr - lm['center']) @ lm['v']
    ds, dd = np.diff(s), np.diff(d)
    d1 = d[1:]
    li = np.argmin(np.abs(d1[:, None] - lm['centers'][None, :]), axis=1)
    off = (d1 - lm['centers'][li]) / lm['spacing']
    speed = np.hypot(ds, dd)
    # 정렬도: 해당 횡방향 위치의 기대 통행방향(연속 프로파일) 대비 실제 진행방향
    exp_dir = np.interp(d1, lm['prof_x'], lm['dir_profile'])
    align = ds * exp_dir / (speed + 1e-6)              # +순방향, -역주행, ~0 정지/횡이동
    return np.column_stack([off, align, dd / lm['spacing'], speed / lm['spacing']])

def add_velocity(arr):
    return np.hstack([arr[1:], np.diff(arr, axis=0)])

def to_sequences(arr):
    return np.array([arr[i:i+SEQ_LEN] for i in range(0, len(arr) - SEQ_LEN + 1, STEP)])

def build_model(n_feat):
    m = Sequential([
        LSTM(64, activation='tanh', recurrent_activation='sigmoid',
             input_shape=(SEQ_LEN, n_feat), return_sequences=False),
        RepeatVector(SEQ_LEN),
        LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
        TimeDistributed(Dense(n_feat)),
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

# ---------- 두 구성 비교 ----------
MODES = {
    'perloc_vel': dict(label='Position+velocity (per-loc norm)'),
    'lane_rel':   dict(label='Lane-relative features'),
}

def featurize(arr, mode, loc):
    if mode == 'lane_rel':
        return lane_rel_features(arr, lane_models[loc])
    return add_velocity(arr)

def fit_scalers(mode):
    scalers = {}
    for loc in LOCATIONS:
        s = MinMaxScaler() if mode == 'perloc_vel' else StandardScaler()
        s.fit(np.vstack([featurize(tracks[k], mode, loc) for k in train_keys if k[0] == loc]))
        scalers[loc] = s
    return scalers

def track_seq_errors(model, arr, scaler, mode, loc):
    seqs = to_sequences(scaler.transform(featurize(arr, mode, loc)))
    recon = model.predict(seqs, verbose=0)
    return np.mean(np.square(seqs - recon), axis=(1, 2))

results = {}
for mode in MODES:
    print(f"\n===== {MODES[mode]['label']} =====")
    scalers = fit_scalers(mode)
    X_train = np.vstack([to_sequences(scalers[k[0]].transform(featurize(tracks[k], mode, k[0])))
                         for k in train_keys])
    print(f"학습 시퀀스: {X_train.shape}")
    tf.keras.utils.set_random_seed(42)
    model = build_model(X_train.shape[2])
    model.fit(X_train, X_train, epochs=10, batch_size=64, validation_split=0.1, shuffle=True, verbose=0)

    train_errs = {k: track_seq_errors(model, tracks[k], scalers[k[0]], mode, k[0]) for k in train_keys}
    thr = {}
    for agg, fn_agg in [('mean', np.mean), ('max', np.max)]:
        by_loc = defaultdict(list)
        for k, e in train_errs.items():
            by_loc[k[0]].append(fn_agg(e))
        thr[agg] = {loc: np.percentile(by_loc[loc], 95) for loc in LOCATIONS}

    y_true, meta, err_list = [], [], []
    for k in test_keys:
        y_true.append(0); meta.append(('normal', k[0]))
        err_list.append(track_seq_errors(model, tracks[k], scalers[k[0]], mode, k[0]))
    for (loc, name), arr in synth.items():
        y_true.append(1); meta.append((name.rsplit('_', 1)[0], loc))
        err_list.append(track_seq_errors(model, arr, scalers[loc], mode, loc))
    y_true = np.array(y_true)

    results[mode] = {}
    for agg, fn_agg in [('mean', np.mean), ('max', np.max)]:
        y_score = np.array([fn_agg(e) for e in err_list])
        ap = average_precision_score(y_true, y_score)
        pred = np.array([s > thr[agg][m[1]] for s, m in zip(y_score, meta)])
        tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0)))
        fn_ = int(np.sum(~pred & (y_true == 1)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn_) if tp + fn_ else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        type_recall = {t: float(np.mean(pred[[i for i, m in enumerate(meta) if m[0] == t]]))
                       for t in ANOM_TYPES}
        print(f"[{agg:>4} 집계] PR-AUC {ap:.3f} | P {prec:.2f} / R {rec:.2f} / F1 {f1:.2f} | " +
              " ".join(f"{t}:{r:.0%}" for t, r in type_recall.items()))
        results[mode][agg] = dict(ap=ap, prec=prec, rec=rec, f1=f1, type_recall=type_recall,
                                  y_true=y_true, y_score=y_score)

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
for mode, color in [('perloc_vel', 'royalblue'), ('lane_rel', 'darkorange')]:
    for agg, ls in [('max', '-'), ('mean', ':')]:
        r = results[mode][agg]
        p, rc, _ = precision_recall_curve(r['y_true'], r['y_score'])
        axs[0].plot(rc, p, color=color, linestyle=ls,
                    label=f"{MODES[mode]['label']} [{agg}] AP={r['ap']:.3f}")
axs[0].set_xlabel('Recall'); axs[0].set_ylabel('Precision')
axs[0].set_title('PR curves: lane-relative vs position+velocity')
axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.35
axs[1].bar(x - w/2, [results['perloc_vel']['max']['type_recall'][t] for t in types], w,
           color='royalblue', label='Position+velocity [max]')
axs[1].bar(x + w/2, [results['lane_rel']['max']['type_recall'][t] for t in types], w,
           color='darkorange', label='Lane-relative [max]')
axs[1].set_xticks(x); axs[1].set_xticklabels(types, rotation=15)
axs[1].set_ylim(0, 1.05); axs[1].set_ylabel('Recall (calibrated thr)')
axs[1].set_title('Detection rate by anomaly type'); axs[1].legend()

# 차로 모델 시각화 (지점 11): 차선 중심선 + 방향
loc = 11
lm = lane_models[loc]
for k in [kk for kk in train_keys if kk[0] == loc][:80]:
    axs[2].plot(tracks[k][:, 0], tracks[k][:, 1], color='gray', alpha=0.2, linewidth=0.5)
L = 260
for c, dirn in zip(lm['centers'], lm['lane_dir']):
    p0 = lm['center'] + c * lm['v'] - L * lm['u']
    p1 = lm['center'] + c * lm['v'] + L * lm['u']
    axs[2].plot([p0[0], p1[0]], [p0[1], p1[1]], color='crimson', linewidth=1.2, alpha=0.8)
    mid = lm['center'] + c * lm['v']
    ar = 60 * dirn * lm['u']
    axs[2].annotate('', xy=(mid[0] + ar[0], mid[1] + ar[1]), xytext=(mid[0], mid[1]),
                    arrowprops=dict(arrowstyle='->', color='navy', lw=1.6))
axs[2].invert_yaxis()
axs[2].set_title(f'Estimated lane centers & directions (Location {loc})')
plt.tight_layout()
plt.savefig(f"{OUT}/10_lane_relative_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 10_lane_relative_eval.png")
print("LANEREL_OK")
