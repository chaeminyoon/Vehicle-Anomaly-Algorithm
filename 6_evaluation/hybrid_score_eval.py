# 개선 실험 ⑪: 하이브리드 스코어 — 규칙 점수(A2+D3) + LSTM-AE 재구성 오차 결합
# 실험 ④의 교훈("하이브리드(규칙+학습) 구조가 실용적 방향")을 실제로 검증:
#   - AE가 규칙이 놓치는 이상을 잡아주는가? (상보성 분석)
#   - 결합 방식(max / mean)에 따라 순개선이 있는가?
# 두 점수 모두 지점별 학습 정상 통계로 z-정규화 후 결합, 임계값은 학습 95% 보정.
# 평가: 확대 평가셋 (지점·유형당 24개, 시드 43 — 실험 ⑨⑩과 동일)
import numpy as np  # noqa

# ---------- 데이터/도로모델/규칙 파이프라인 준비 (실험 ⑩ 스크립트 재사용) ----------
# lane_cross_features_eval.py 를 evaluate() 정의까지 exec -> T, S(확대셋), 규칙/특징 함수 확보
exec(open('lane_cross_features_eval.py').read().split("FEATURE_SETS = {")[0])

RULE_EXTRA = ['cross_flow', 'osc']  # D3 채택 구성

def rules(arr, loc):
    off, sp, dl, li = lane_feats(arr, loc)
    return np.array([ww_score(arr, loc), np.abs(off).max(), off.std(),
                     (sp < speed_q_[loc]).mean(), cross_flow(arr, loc), osc_energy(dl)])

# ---------- 규칙 점수 (지점별 z-max) ----------
tr_rules = defaultdict(list)
for k in train_keys:
    tr_rules[k[0]].append(rules(T[k], k[0]))
rstats = {loc: (np.mean(tr_rules[loc], axis=0), np.std(tr_rules[loc], axis=0) + 1e-9)
          for loc in LOCATIONS}

def rule_score(arr, loc):
    z = (rules(arr, loc) - rstats[loc][0]) / rstats[loc][1]
    return float(z.max())

# ---------- LSTM-AE 재구성 오차 (실험 ③ 최고 구성: 지점별 정규화 + 위치+속도, max 집계) ----------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN, STEP = 50, 10
tf.keras.utils.set_random_seed(42)

def add_velocity(arr):
    return np.hstack([arr[1:], np.diff(arr, axis=0)])

def to_sequences(arr):
    return np.array([arr[i:i + SEQ_LEN] for i in range(0, len(arr) - SEQ_LEN + 1, STEP)])

scalers = {}
for loc in LOCATIONS:
    sc = MinMaxScaler()
    sc.fit(np.vstack([add_velocity(T[k]) for k in train_keys if k[0] == loc]))
    scalers[loc] = sc

X_train = np.vstack([to_sequences(scalers[k[0]].transform(add_velocity(T[k]))) for k in train_keys])
print(f"AE 학습 시퀀스: {X_train.shape}")
model = Sequential([
    LSTM(64, activation='tanh', recurrent_activation='sigmoid',
         input_shape=(SEQ_LEN, 4), return_sequences=False),
    RepeatVector(SEQ_LEN),
    LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
    TimeDistributed(Dense(4)),
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=10, batch_size=64, validation_split=0.1, shuffle=True, verbose=0)

def ae_error(arr, loc):
    seqs = to_sequences(scalers[loc].transform(add_velocity(arr)))
    if len(seqs) == 0:
        return 0.0
    recon = model.predict(seqs, verbose=0)
    return float(np.max(np.mean(np.square(seqs - recon), axis=(1, 2))))  # max 집계

# ---------- 두 점수의 지점별 z-정규화 (학습 정상 기준) ----------
print("AE 오차 산출 중 (학습/평가/합성)...")
rule_train = {loc: [rule_score(T[k], loc) for k in train_keys if k[0] == loc] for loc in LOCATIONS}
ae_train = {loc: [ae_error(T[k], loc) for k in train_keys if k[0] == loc] for loc in LOCATIONS}
znorm = {loc: dict(rm=np.mean(rule_train[loc]), rs=np.std(rule_train[loc]) + 1e-9,
                   am=np.mean(ae_train[loc]), as_=np.std(ae_train[loc]) + 1e-9)
         for loc in LOCATIONS}

def both_z(arr, loc):
    n = znorm[loc]
    return ((rule_score(arr, loc) - n['rm']) / n['rs'],
            (ae_error(arr, loc) - n['am']) / n['as_'])

items = []  # (y, type, loc, rule_z, ae_z)
for k in test_keys:
    rz, az = both_z(T[k], k[0])
    items.append((0, 'normal', k[0], rz, az))
for (loc, name), arr in S.items():
    rz, az = both_z(arr, loc)
    items.append((1, name.rsplit('_', 1)[0], loc, rz, az))
train_z = {loc: [((r - znorm[loc]['rm']) / znorm[loc]['rs'], (a - znorm[loc]['am']) / znorm[loc]['as_'])
                 for r, a in zip(rule_train[loc], ae_train[loc])] for loc in LOCATIONS}

# ---------- 결합 방식 비교 ----------
FUSIONS = {
    'rule only (D3)':  lambda r, a: r,
    'AE only':         lambda r, a: a,
    'hybrid max':      lambda r, a: max(r, a),
    'hybrid mean':     lambda r, a: 0.5 * (r + a),
}
res = {}
y_true = np.array([it[0] for it in items])
for label, fuse in FUSIONS.items():
    thr = {loc: np.percentile([fuse(r, a) for r, a in train_z[loc]], 95) for loc in LOCATIONS}
    y_score = np.array([fuse(it[3], it[4]) for it in items])
    pred = np.array([fuse(it[3], it[4]) > thr[it[2]] for it in items])
    ap = average_precision_score(y_true, y_score)
    tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0)))
    fn_ = int(np.sum(~pred & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    type_recall = {t: float(np.mean(pred[[i for i, it in enumerate(items) if it[1] == t]]))
                   for t in ANOM_TYPES}
    res[label] = dict(ap=ap, prec=prec, rec=rec, f1=f1, fp=fp, type_recall=type_recall, pred=pred)
    print(f"{label}: PR-AUC {ap:.3f} | P {prec:.2f} / R {rec:.2f} / F1 {f1:.2f} (FP {fp}) | " +
          " ".join(f"{t}:{v:.0%}" for t, v in type_recall.items()))

# ---------- 상보성 분석: 규칙이 놓친 이상 중 AE가 잡는 것 ----------
rp, ap_ = res['rule only (D3)']['pred'], res['AE only']['pred']
anom = y_true == 1
missed_by_rule = anom & ~rp
print(f"\n규칙이 놓친 이상 {int(missed_by_rule.sum())}개 중 AE가 잡음: {int((missed_by_rule & ap_).sum())}개")
by_type = {}
for t in ANOM_TYPES:
    idx = np.array([it[1] == t for it in items])
    m = missed_by_rule & idx
    by_type[t] = (int(m.sum()), int((m & ap_).sum()))
print("유형별 (규칙 미탐 / 그중 AE 탐지):", by_type)
ae_fp_only = int(((y_true == 0) & ap_ & ~rp).sum())
print(f"AE만 발생시키는 추가 오탐(정상): {ae_fp_only}개")

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.2
colors = ['darkorange', 'royalblue', 'seagreen', 'purple']
for i, label in enumerate(FUSIONS):
    axs[0].bar(x + (i - 1.5) * w, [res[label]['type_recall'][t] for t in types], w,
               color=colors[i], label=f"{label} (F1={res[label]['f1']:.2f})")
axs[0].set_xticks(x); axs[0].set_xticklabels(types, rotation=12)
axs[0].set_ylim(0, 1.05); axs[0].set_ylabel('Recall (calibrated thr)')
axs[0].set_title('Rule vs AE vs hybrid (enlarged eval set)')
axs[0].legend(fontsize=8)

rz_n = [it[3] for it in items if it[0] == 0]; az_n = [it[4] for it in items if it[0] == 0]
rz_a = [it[3] for it in items if it[0] == 1]; az_a = [it[4] for it in items if it[0] == 1]
axs[1].scatter(rz_n, az_n, s=12, color='gray', alpha=0.5, label='normal')
axs[1].scatter(rz_a, az_a, s=12, color='crimson', alpha=0.6, label='anomaly')
axs[1].set_xlabel('rule z-score'); axs[1].set_ylabel('AE z-score')
axs[1].set_title('Score space: are rule and AE complementary?')
axs[1].legend(); axs[1].grid(alpha=0.3)

labels = list(FUSIONS)
axs[2].plot(labels, [res[l]['f1'] for l in labels], 'o-', color='darkorange', label='F1')
axs[2].plot(labels, [res[l]['ap'] for l in labels], 's-', color='royalblue', label='PR-AUC')
axs[2].plot(labels, [res[l]['rec'] for l in labels], '^-', color='seagreen', label='Recall')
axs[2].set_ylim(0, 1); axs[2].grid(alpha=0.3); axs[2].legend()
plt.setp(axs[2].get_xticklabels(), rotation=10, fontsize=8)
axs[2].set_title('Overall metrics per fusion')
plt.tight_layout()
plt.savefig(f"{OUT}/17_hybrid_score_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 17_hybrid_score_eval.png")
print("HYBRID_OK")
