# 개선 실험 ③: 합성 이상 주입 평가셋으로 정량 평가 (PR-AUC, 유형별 탐지율)
# - 정상 궤적 80/20 학습/평가 분리, 평가 궤적 변형으로 4유형 합성 이상 생성
# - 임계값은 학습(정상) 데이터의 지점별 95% 분위수로 보정 -> 평가셋에 적용
# - 전역 정규화 모델 vs 지점별 정규화 모델 비교
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

np.random.seed(42); tf.random.set_seed(42)

OUT = "/Volumes/T7/1. 논문정리/github/analysis_results"
DATA = "/Volumes/T7/1. 논문정리/trajectory_project/Code/combined_trajectory_data.csv"
LOCATIONS = [11, 14, 17]
SEQ_LEN, STEP = 50, 10
N_PER_TYPE = 6  # 지점당 유형별 합성 이상 수

raw = pd.read_csv(DATA)
raw = raw[raw['Location'].isin(LOCATIONS)][['Location', 'TrackID', 'Time', 'X_center', 'Y_center']]
raw = raw.sort_values(['Location', 'Time'])

# ---- 궤적 단위로 추출 (원좌표, 길이 >= SEQ_LEN+10) ----
tracks = {}  # (loc, tid) -> (N,2) array
for (loc, tid), g in raw.groupby(['Location', 'TrackID']):
    arr = g[['X_center', 'Y_center']].values.astype(float)
    if len(arr) >= SEQ_LEN + 10:
        tracks[(loc, tid)] = arr
print(f"사용 가능 궤적(길이>=60): {len(tracks)}개")

# ---- 80/20 분리 ----
keys = sorted(tracks.keys())
rng = np.random.RandomState(42)
rng.shuffle(keys)
n_test = int(len(keys) * 0.2)
test_keys, train_keys = keys[:n_test], keys[n_test:]
print(f"학습(정상): {len(train_keys)}개 / 평가(정상): {len(test_keys)}개")

# ---- 합성 이상 생성 (지점별 좌표 범위에 비례한 변형 크기) ----
loc_range = {loc: raw[raw['Location'] == loc]['X_center'].max() - raw[raw['Location'] == loc]['X_center'].min()
             for loc in LOCATIONS}

def lateral_unit(arr):
    d = arr[-1] - arr[0]
    n = np.array([-d[1], d[0]])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else np.array([0.0, 1.0])

def make_wrong_way(arr, scale):      # 역주행: 시간 역순
    return arr[::-1].copy()

def make_lane_cross(arr, scale):     # 차로 횡단: 시그모이드 횡방향 오프셋 (차로 2~3개 폭)
    t = np.linspace(-6, 6, len(arr))
    offset = 1 / (1 + np.exp(-t)) * scale * 0.08
    return arr + np.outer(offset, lateral_unit(arr))

def make_sudden_stop(arr, scale):    # 급정거: 중간 30% 구간 위치 정지
    a = arr.copy()
    m = len(a) // 2
    dur = max(int(len(a) * 0.3), 15)
    a[m:m+dur] = a[m]
    return a

def make_zigzag(arr, scale):         # 지그재그: 사인파 횡방향 사행 (3주기)
    t = np.linspace(0, 3 * 2 * np.pi, len(arr))
    offset = np.sin(t) * scale * 0.03
    return arr + np.outer(offset, lateral_unit(arr))

ANOM_TYPES = {'wrong_way': make_wrong_way, 'lane_cross': make_lane_cross,
              'sudden_stop': make_sudden_stop, 'zigzag': make_zigzag}

synth = {}  # (loc, 'type_i') -> arr
test_by_loc = defaultdict(list)
for k in test_keys:
    test_by_loc[k[0]].append(k)
for loc in LOCATIONS:
    pool = test_by_loc[loc]
    for tname, fn in ANOM_TYPES.items():
        picks = rng.choice(len(pool), size=min(N_PER_TYPE, len(pool)), replace=False)
        for i, pi in enumerate(picks):
            synth[(loc, f"{tname}_{i}")] = fn(tracks[pool[pi]], loc_range[loc])
print(f"합성 이상궤적: {len(synth)}개 ({len(ANOM_TYPES)}유형 x {len(LOCATIONS)}지점 x {N_PER_TYPE})")

# ---- 정규화 스케일러 (학습 데이터 기준으로만 fit) ----
def featurize(arr, mode):
    return add_velocity(arr) if mode.endswith('_vel') else arr

def fit_scalers(mode):
    if mode == 'global':
        s = MinMaxScaler()
        s.fit(np.vstack([featurize(tracks[k], mode) for k in train_keys]))
        return {loc: s for loc in LOCATIONS}
    scalers = {}
    for loc in LOCATIONS:
        s = MinMaxScaler()
        s.fit(np.vstack([featurize(tracks[k], mode) for k in train_keys if k[0] == loc]))
        scalers[loc] = s
    return scalers

def to_sequences(arr):
    return np.array([arr[i:i+SEQ_LEN] for i in range(0, len(arr) - SEQ_LEN + 1, STEP)])

def add_velocity(arr):
    # (N,2) 위치 -> (N-1,4) [X, Y, dX, dY]
    return np.hstack([arr[1:], np.diff(arr, axis=0)])

def build_model(n_feat=2):
    m = Sequential([
        LSTM(64, activation='tanh', recurrent_activation='sigmoid',
             input_shape=(SEQ_LEN, n_feat), return_sequences=False),
        RepeatVector(SEQ_LEN),
        LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
        TimeDistributed(Dense(n_feat)),
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

def track_seq_errors(model, arr, scaler, mode):
    seqs = to_sequences(scaler.transform(featurize(arr, mode)))
    recon = model.predict(seqs, verbose=0)
    return np.mean(np.square(seqs - recon), axis=(1, 2))

results = {}
for mode in ['global', 'perloc', 'perloc_vel']:
    labels_kr = {'global': '전역 정규화', 'perloc': '지점별 정규화', 'perloc_vel': '지점별 정규화 + 속도 특징'}
    print(f"\n===== 모델: {labels_kr[mode]} =====")
    scalers = fit_scalers(mode)
    X_train = np.vstack([to_sequences(scalers[k[0]].transform(featurize(tracks[k], mode))) for k in train_keys])
    print(f"학습 시퀀스: {X_train.shape}")
    tf.keras.utils.set_random_seed(42)
    model = build_model(n_feat=X_train.shape[2])
    model.fit(X_train, X_train, epochs=10, batch_size=64, validation_split=0.1, shuffle=True, verbose=0)

    # 임계값 보정: 학습(정상) 궤적의 지점별 오차 95% 분위수 (mean/max 집계 각각)
    train_errs = {k: track_seq_errors(model, tracks[k], scalers[k[0]], mode) for k in train_keys}
    thr = {}
    for agg, fn_agg in [('mean', np.mean), ('max', np.max)]:
        by_loc = defaultdict(list)
        for k, e in train_errs.items():
            by_loc[k[0]].append(fn_agg(e))
        thr[agg] = {loc: np.percentile(by_loc[loc], 95) for loc in LOCATIONS}

    # 평가셋 스코어링
    y_true, meta, err_list = [], [], []
    for k in test_keys:
        y_true.append(0); meta.append(('normal', k[0]))
        err_list.append(track_seq_errors(model, tracks[k], scalers[k[0]], mode))
    for (loc, name), arr in synth.items():
        y_true.append(1); meta.append((name.rsplit('_', 1)[0], loc))
        err_list.append(track_seq_errors(model, arr, scalers[loc], mode))
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
        type_recall = {}
        for tname in ANOM_TYPES:
            idx = [i for i, m in enumerate(meta) if m[0] == tname]
            type_recall[tname] = float(np.mean(pred[idx]))
        print(f"[{agg:>4} 집계] PR-AUC {ap:.3f} | P {prec:.2f} / R {rec:.2f} / F1 {f1:.2f} | " +
              " ".join(f"{t}:{r:.0%}" for t, r in type_recall.items()))
        results[mode][agg] = dict(ap=ap, prec=prec, rec=rec, f1=f1, type_recall=type_recall,
                                  y_true=y_true, y_score=y_score)

# ---- 시각화 ----
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
for mode, label, color in [('global', 'Global norm (position only)', 'tomato'),
                           ('perloc', 'Per-loc norm (position only)', 'seagreen'),
                           ('perloc_vel', 'Per-loc norm + velocity', 'royalblue')]:
    r = results[mode]['max']
    p, rc, _ = precision_recall_curve(r['y_true'], r['y_score'])
    axs[0].plot(rc, p, color=color, label=f"{label} (PR-AUC={r['ap']:.3f})")
    rm = results[mode]['mean']
    pm, rcm, _ = precision_recall_curve(rm['y_true'], rm['y_score'])
    axs[0].plot(rcm, pm, color=color, linestyle=':', alpha=0.5,
                label=f"  (mean agg: {rm['ap']:.3f})")
axs[0].set_xlabel('Recall'); axs[0].set_ylabel('Precision')
axs[0].set_title('PR curve (solid: max agg, dotted: mean agg)')
axs[0].legend(); axs[0].grid(alpha=0.3)

types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.27
axs[1].bar(x - w, [results['global']['max']['type_recall'][t] for t in types], w,
           color='tomato', label='Global norm')
axs[1].bar(x, [results['perloc']['max']['type_recall'][t] for t in types], w,
           color='seagreen', label='Per-loc norm')
axs[1].bar(x + w, [results['perloc_vel']['max']['type_recall'][t] for t in types], w,
           color='royalblue', label='Per-loc + velocity')
axs[1].set_xticks(x); axs[1].set_xticklabels(types, rotation=15)
axs[1].set_ylim(0, 1.05); axs[1].set_ylabel('Recall (calibrated thr, max agg)')
axs[1].set_title('Detection rate by anomaly type'); axs[1].legend()

# 합성 이상 예시 (지점 11)
loc = 11
shown = set()
for (l, name), arr in synth.items():
    t = name.rsplit('_', 1)[0]
    if l == loc and t not in shown:
        axs[2].plot(arr[:, 0], arr[:, 1], linewidth=2, label=t, zorder=3)
        shown.add(t)
for k in test_by_loc[loc][:60]:
    axs[2].plot(tracks[k][:, 0], tracks[k][:, 1], color='gray', alpha=0.2, linewidth=0.6)
axs[2].invert_yaxis(); axs[2].legend(fontsize=9)
axs[2].set_title(f'Synthetic anomaly examples (Location {loc})')
plt.tight_layout()
plt.savefig(f"{OUT}/9_synthetic_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 9_synthetic_eval.png")
print("SYNTH_OK")
