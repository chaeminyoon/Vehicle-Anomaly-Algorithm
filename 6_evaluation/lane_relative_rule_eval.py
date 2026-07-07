# 개선 실험 ④-b: 차로 상대 특징 + 통계 규칙 점수 vs LSTM-AE 재구성 오차
# 궤적 단위 규칙 특징: [역주행 점수, 최대 차선이탈, 오프셋 변동성, 정지 비율]
# 점수 = 지점별 z-score의 최댓값, 임계값 = 학습 정상 95% 분위수
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

exec(open('lane_relative_ae_eval.py').read().split("# ---------- 두 구성 비교")[0]
     .replace("import tensorflow as tf", "")
     .replace("from tensorflow.keras.models import Sequential", "")
     .replace("from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense", "")
     .replace("tf.random.set_seed(42)", ""))

# 지점별 학습 프레임 속도 5% 분위수 (정지 판정 기준)
speed_q = {}
for loc in LOCATIONS:
    sp = np.concatenate([lane_rel_features(tracks[k], lane_models[loc])[:, 3]
                         for k in train_keys if k[0] == loc])
    speed_q[loc] = np.percentile(sp, 5)

# 2차원 방향장(direction field): 화면 격자별 기대 진행방향 (학습 궤적으로 추정)
# 1차원 횡방향 투영은 원근으로 상/하행 영역이 겹쳐 방향 오배정이 많음 -> 2D 격자로 해소
from scipy.ndimage import gaussian_filter
GRID = 48
dir_fields = {}
for loc in LOCATIONS:
    tkeys = [k for k in train_keys if k[0] == loc]
    P = np.vstack([tracks[k] for k in tkeys])
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    vx = np.zeros((GRID, GRID)); vy = np.zeros((GRID, GRID)); cnt = np.zeros((GRID, GRID))
    w = 10
    for k in tkeys:
        a = tracks[k]
        if len(a) <= w:
            continue
        disp = a[w:] - a[:-w]
        norm = np.linalg.norm(disp, axis=1) + 1e-9
        unit = disp / norm[:, None]
        mid = (a[w:] + a[:-w]) / 2
        ix = np.clip(((mid[:, 0] - x0) / (x1 - x0 + 1e-9) * GRID).astype(int), 0, GRID - 1)
        iy = np.clip(((mid[:, 1] - y0) / (y1 - y0 + 1e-9) * GRID).astype(int), 0, GRID - 1)
        np.add.at(vx, (iy, ix), unit[:, 0]); np.add.at(vy, (iy, ix), unit[:, 1])
        np.add.at(cnt, (iy, ix), 1)
    vx, vy, cnt = gaussian_filter(vx, 1.5), gaussian_filter(vy, 1.5), gaussian_filter(cnt, 1.5)
    mag = np.hypot(vx, vy)
    coherence = mag / (cnt + 1e-9)        # 1이면 셀 내 방향 완전 일치
    ux = np.where(mag > 0, vx / (mag + 1e-9), 0.0)
    uy = np.where(mag > 0, vy / (mag + 1e-9), 0.0)
    dir_fields[loc] = dict(x0=x0, x1=x1, y0=y0, y1=y1, ux=ux, uy=uy,
                           coherence=coherence, cnt=cnt)

def wrong_way_score(arr, loc, w=10):
    # 2D 방향장 대비 10프레임 변위의 정렬도 반전 평균
    f = dir_fields[loc]
    if len(arr) <= w:
        return 0.0
    disp = arr[w:] - arr[:-w]
    norm = np.linalg.norm(disp, axis=1) + 1e-9
    mid = (arr[w:] + arr[:-w]) / 2
    ix = np.clip(((mid[:, 0] - f['x0']) / (f['x1'] - f['x0'] + 1e-9) * GRID).astype(int), 0, GRID - 1)
    iy = np.clip(((mid[:, 1] - f['y0']) / (f['y1'] - f['y0'] + 1e-9) * GRID).astype(int), 0, GRID - 1)
    align = (disp[:, 0] * f['ux'][iy, ix] + disp[:, 1] * f['uy'][iy, ix]) / norm
    conf = (f['coherence'][iy, ix] > 0.7) & (norm > 1.0)  # 방향 명확 + 실제 이동 구간만
    return -align[conf].mean() if conf.any() else 0.0

def rule_features(arr, loc):
    f = lane_rel_features(arr, lane_models[loc])
    return np.array([
        wrong_way_score(arr, loc),          # 역주행 점수 (10프레임 변위 정렬도 반전)
        np.abs(f[:, 0]).max(),              # 최대 차선 이탈
        f[:, 0].std(),                      # 오프셋 변동성 (사행)
        (f[:, 3] < speed_q[loc]).mean(),    # 정지 비율
    ])

RULE_NAMES = ['wrong_way_score', 'max_offset', 'offset_std', 'stop_ratio']

# 학습 정상으로 지점별 z-score 기준 산출
train_rules = defaultdict(list)
for k in train_keys:
    train_rules[k[0]].append(rule_features(tracks[k], k[0]))
stats = {loc: (np.mean(train_rules[loc], axis=0), np.std(train_rules[loc], axis=0) + 1e-9)
         for loc in LOCATIONS}

def rule_score(arr, loc):
    z = (rule_features(arr, loc) - stats[loc][0]) / stats[loc][1]
    return float(z.max()), z

# 임계값 보정: 학습 정상 점수의 지점별 95% 분위수
thr = {}
for loc in LOCATIONS:
    scores = [rule_score(tracks[k], loc)[0] for k in train_keys if k[0] == loc]
    thr[loc] = np.percentile(scores, 95)
print("보정 임계값:", {l: round(t, 2) for l, t in thr.items()})

# 평가
y_true, y_score, meta, z_all = [], [], [], []
for k in test_keys:
    s, z = rule_score(tracks[k], k[0])
    y_true.append(0); y_score.append(s); meta.append(('normal', k[0])); z_all.append(z)
for (loc, name), arr in synth.items():
    s, z = rule_score(arr, loc)
    y_true.append(1); y_score.append(s); meta.append((name.rsplit('_', 1)[0], loc)); z_all.append(z)
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
print(f"\n규칙 기반: PR-AUC {ap:.3f} | P {prec:.2f} / R {rec:.2f} / F1 {f1:.2f}")
print("유형별 탐지율:", {t: f"{r:.0%}" for t, r in type_recall.items()})

# 어떤 규칙이 각 유형을 잡는지
print("\n유형별 최대 z-score 특징 (탐지된 것 기준):")
for t in ANOM_TYPES:
    idx = [i for i, m in enumerate(meta) if m[0] == t and pred[i]]
    if idx:
        which = [RULE_NAMES[int(np.argmax(z_all[i]))] for i in idx]
        print(f"  {t}: {dict((w, which.count(w)) for w in set(which))}")

# 시각화
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
p, rc, _ = precision_recall_curve(y_true, y_score)
axs[0].plot(rc, p, color='darkorange', linewidth=2,
            label=f'Rule-based lane-relative (PR-AUC={ap:.3f})')
# 비교선: 실험 ③의 최고 구성 결과 (perloc_vel max)
axs[0].axhline(0.34, color='gray', linestyle=':', alpha=0.5)
axs[0].plot([], [], color='royalblue', label='LSTM-AE best (PR-AUC=0.428, exp.3)')
axs[0].set_xlabel('Recall'); axs[0].set_ylabel('Precision')
axs[0].set_title('Rule-based scoring on lane-relative features')
axs[0].legend(); axs[0].grid(alpha=0.3)

types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.35
ae_best = {'wrong_way': 0.22, 'lane_cross': 0.17, 'sudden_stop': 0.28, 'zigzag': 0.0}
axs[1].bar(x - w/2, [ae_best[t] for t in types], w, color='royalblue',
           label='LSTM-AE (pos+vel, max agg)')
axs[1].bar(x + w/2, [type_recall[t] for t in types], w, color='darkorange',
           label='Rule-based (lane-relative)')
axs[1].set_xticks(x); axs[1].set_xticklabels(types, rotation=15)
axs[1].set_ylim(0, 1.05); axs[1].set_ylabel('Recall (calibrated thr)')
axs[1].set_title('Detection rate: reconstruction vs rule-based'); axs[1].legend()
plt.tight_layout()
plt.savefig("/Volumes/T7/1. 논문정리/github/analysis_results/10_rule_based_eval.png",
            dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 10_rule_based_eval.png")
print("RULE_OK")
