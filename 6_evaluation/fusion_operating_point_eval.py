# 개선 실험 ⑬-b: 융합 위상(score vs decision) × 운영점(오경보 예산) 분석
# prediction_score_eval.py 가 덤프한 점수(pred_eval_scores.npz)에서 재학습 없이 수행.
#
# 질문 1 — 결정 융합(OR)이 점수 융합(mean)보다 나은가?
#   OR(r,a,p) q95는 F1 0.88로 ⑪(0.85)을 넘는 듯 보이지만 오탐이 11→22.
#   같은 오탐에서 비교하면(hybrid mean q87, FP 22) mean이 F1 0.90으로 역전 —
#   OR의 이득은 구조가 아니라 "오경보 예산을 더 쓴 것". 모든 매칭 지점에서
#   score-mean이 decision-OR를 지배한다.
# 질문 2 — 그렇다면 ⑪ 점수의 실제 한계는 어디인가?
#   q 스윕으로 운영점 곡선을 그리면 q85에서 F1 0.92 (R 0.93, FP 27) —
#   점수가 아니라 운영점(명목 오경보율 5%)이 Recall을 묶고 있었다.
#   단, q를 평가셋 F1로 고르는 것은 누수이므로 "채택 구성 변경"이 아니라
#   운영점-비용 곡선으로 기록한다 (⑫ POT의 q=오경보율 해석과 연결).
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "/Volumes/T7/1. 논문정리/github/analysis_results"
D = np.load(f"{OUT}/pred_eval_scores.npz", allow_pickle=True)
y, types, locs = D['y'], D['types'], D['locs'].astype(int)
rz, az, pz = D['rz'], D['az'], D['pz']
LOCATIONS = [11, 14, 17]
train = {loc: D[f"train_{loc}"] for loc in LOCATIONS}
ANOM_TYPES = ['wrong_way', 'lane_cross', 'sudden_stop', 'zigzag']
N_NORMAL = int((y == 0).sum())

def metrics(pred):
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tr = {t: float(np.mean(pred[types == t])) for t in ANOM_TYPES}
    return dict(f1=f1, p=prec, r=rec, fp=fp, tr=tr)

def flag(fuse, q):
    thr = {loc: np.percentile([fuse(*row) for row in train[loc]], q) for loc in LOCATIONS}
    s = np.array([fuse(r, a, p) for r, a, p in zip(rz, az, pz)])
    return np.array([si > thr[l] for si, l in zip(s, locs)])

hmean = lambda r, a, p: (r + a) / 2
fr = lambda r, a, p: r; fa = lambda r, a, p: a; fp_ = lambda r, a, p: p

# ---------- (1) hybrid mean 운영점 곡선 (q 스윕) ----------
qs = np.arange(80, 99.5, 0.5)
sweep = [metrics(flag(hmean, q)) for q in qs]

# ---------- (2) OR 결정 융합 점들 ----------
or_points = {}
for q in [95, 96, 97, 98, 99]:
    or_points[q] = metrics(flag(fr, q) | flag(fa, q) | flag(fp_, q))
for q, m in or_points.items():
    print(f"OR(r,a,p) q{q}: F1 {m['f1']:.2f} FP {m['fp']}")

# ---------- (3) 오탐 매칭 표 ----------
print(f"\n{'FP':>3} | {'mean q':>6} {'mean F1':>7} | {'OR q':>4} {'OR F1':>6}")
match_rows = []
for orq, orm in or_points.items():
    close = min(range(len(qs)), key=lambda i: abs(sweep[i]['fp'] - orm['fp']))
    match_rows.append((orm['fp'], qs[close], sweep[close]['f1'], orq, orm['f1']))
    print(f"{orm['fp']:>3} | {qs[close]:>6.1f} {sweep[close]['f1']:>7.2f} | {orq:>4} {orm['f1']:>6.2f}")

# ---------- (4) 운영점별 유형 recall (오경보 예산이 사는 것) ----------
OP_QS = [95, 90, 85]
op_metrics = {q: metrics(flag(hmean, q)) for q in OP_QS}
for q in OP_QS:
    m = op_metrics[q]
    print(f"hybrid mean q{q}: F1 {m['f1']:.2f} P {m['p']:.2f} R {m['r']:.2f} FP {m['fp']} | " +
          " ".join(f"{t}:{v:.0%}" for t, v in m['tr'].items()))

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))

fps = [m['fp'] for m in sweep]
axs[0].plot(fps, [m['f1'] for m in sweep], '-', color='darkorange', lw=2, label='score fusion: hybrid mean (q sweep)')
axs[0].plot(fps, [m['r'] for m in sweep], '--', color='darkorange', lw=1.2, alpha=0.6, label='  recall')
axs[0].scatter([m['fp'] for m in or_points.values()], [m['f1'] for m in or_points.values()],
               marker='s', s=55, color='royalblue', zorder=5, label='decision fusion: OR(r,a,p), q 95-99')
b11 = next(m for q, m in zip(qs, sweep) if q == 95.0)
axs[0].scatter([b11['fp']], [b11['f1']], marker='*', s=260, color='crimson', zorder=6,
               label=f"adopted exp.11 (q95): F1 {b11['f1']:.2f}")
axs[0].annotate(f"q85: F1 {op_metrics[85]['f1']:.2f}", xy=(op_metrics[85]['fp'], op_metrics[85]['f1']),
                xytext=(op_metrics[85]['fp'] - 6, op_metrics[85]['f1'] + 0.035), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=0.8))
axs[0].set_xlabel(f'false positives (of {N_NORMAL} normal tracks)')
axs[0].set_ylabel('F1 / Recall'); axs[0].set_ylim(0.6, 1.0)
axs[0].set_title('Operating curve: score-mean dominates decision-OR\nat every matched false-alarm budget')
axs[0].legend(fontsize=8, loc='lower right'); axs[0].grid(alpha=0.3)

xm = np.arange(len(match_rows)); w = 0.38
axs[1].bar(xm - w/2, [r[2] for r in match_rows], w, color='darkorange', label='hybrid mean @ same FP')
axs[1].bar(xm + w/2, [r[4] for r in match_rows], w, color='royalblue', label='OR(r,a,p)')
axs[1].set_xticks(xm)
axs[1].set_xticklabels([f"FP≈{r[0]}" for r in match_rows])
axs[1].set_ylim(0.6, 1.0); axs[1].set_ylabel('F1')
axs[1].set_title('Matched-FP comparison: the OR "win" was\njust extra false-alarm budget')
axs[1].legend(fontsize=9); axs[1].grid(alpha=0.3, axis='y')

xt = np.arange(len(ANOM_TYPES)); w2 = 0.26
colors = ['crimson', 'darkorange', 'seagreen']
for i, q in enumerate(OP_QS):
    m = op_metrics[q]
    axs[2].bar(xt + (i - 1) * w2, [m['tr'][t] for t in ANOM_TYPES], w2, color=colors[i],
               label=f"q{q}: F1 {m['f1']:.2f} (FP {m['fp']}, FPR {m['fp']/N_NORMAL:.0%})")
axs[2].set_xticks(xt); axs[2].set_xticklabels(ANOM_TYPES, rotation=12)
axs[2].set_ylim(0, 1.05); axs[2].set_ylabel('Recall')
axs[2].set_title('What extra false-alarm budget buys, per type\n(hybrid mean, operating point q)')
axs[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/20_fusion_operating_point.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 20_fusion_operating_point.png")
print("FUSION_OP_OK")
