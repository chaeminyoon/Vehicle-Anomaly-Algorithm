# 개선 실험 ⑫: EVT(POT) 기반 동적 임계값 — "상위 X%" 임의 기준 제거
# 문제: 지점별 95% 분위수 임계값(①②⑪)은 정의상 학습 정상의 5%를 항상 초과로 만들고,
#       "5%"라는 수치 자체에 통계적 근거가 없음. 꼬리 표본 몇 개에 그대로 좌우됨.
# 방법: Peaks-Over-Threshold — 학습 정상 하이브리드 점수의 앵커(u=90% 분위수) 초과분에
#       일반화 파레토 분포(GPD)를 MLE 적합, 목표 초과확률 q의 임계값을 해석적으로 산출
#       (Siffer et al., KDD 2017, "Anomaly Detection in Streams with Extreme Value Theory"):
#           z_q = u + (σ/γ)((q·n/Nu)^(-γ) - 1)     (γ→0 극한: z_q = u - σ·ln(q·n/Nu))
#       q는 "정상 궤적이 임계값을 넘을 확률"이라는 운영적 의미를 가짐 (허용 오경보율).
# 평가: 채택 구성(⑪ 하이브리드 mean) 점수 고정, 지점별 95% 분위수 vs 지점별 POT 비교.
#       + 앵커 민감도: 분위수 방식은 기준(95%)을 바꾸면 임계값이 그대로 흔들리지만
#         POT z_q는 앵커 u(85/90/95%) 선택에 둔감한지 확인.
# 평가셋: 확대 평가셋 (지점·유형당 24개, 시드 43 — 실험 ⑨⑩⑪과 동일)
import numpy as np  # noqa

# ---------- 점수 파이프라인 준비 (실험 ⑪ 스크립트 재사용: 결합 방식 비교 직전까지) ----------
exec(open('hybrid_score_eval.py').read().split("# ---------- 결합 방식 비교")[0])
from scipy.stats import genpareto

fuse = lambda r, a: 0.5 * (r + a)  # 채택 구성: 하이브리드 mean
train_scores = {loc: np.array([fuse(r, a) for r, a in train_z[loc]]) for loc in LOCATIONS}
y_true = np.array([it[0] for it in items])
y_score = np.array([fuse(it[3], it[4]) for it in items])
item_loc = np.array([it[2] for it in items])
item_type = np.array([it[1] for it in items])

# ---------- POT 적합 및 동적 임계값 ----------
def pot_fit(scores, anchor_q=0.90):
    """앵커 u 초과분에 GPD 적합 -> (u, gamma, sigma, n, Nu)"""
    u = np.quantile(scores, anchor_q)
    exc = scores[scores > u] - u
    gamma, _, sigma = genpareto.fit(exc, floc=0)
    return dict(u=u, gamma=gamma, sigma=sigma, n=len(scores), Nu=len(exc), exc=exc)

def pot_threshold(fit, q):
    """초과확률 q에 대한 임계값 z_q"""
    u, g, s, n, Nu = fit['u'], fit['gamma'], fit['sigma'], fit['n'], fit['Nu']
    r = q * n / Nu
    if abs(g) < 1e-6:
        return u - s * np.log(r)
    return u + (s / g) * (r ** (-g) - 1)

pot = {loc: pot_fit(train_scores[loc]) for loc in LOCATIONS}
print("지점별 POT 적합 (앵커 u = 학습 정상 90% 분위수):")
for loc in LOCATIONS:
    f = pot[loc]
    print(f"  지점 {loc}: n={f['n']}, Nu={f['Nu']}, u={f['u']:.3f}, "
          f"gamma={f['gamma']:.3f}, sigma={f['sigma']:.3f}")

# ---------- 평가: 95% 분위수 vs POT(q 그리드) ----------
def evaluate(thr):
    pred = y_score > np.array([thr[l] for l in item_loc])
    tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0)))
    fn_ = int(np.sum(~pred & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    type_recall = {t: float(np.mean(pred[item_type == t])) for t in ANOM_TYPES}
    return dict(prec=prec, rec=rec, f1=f1, fp=fp, type_recall=type_recall, pred=pred)

results = {}
thr_q95 = {loc: np.percentile(train_scores[loc], 95) for loc in LOCATIONS}
results['baseline q95'] = (thr_q95, evaluate(thr_q95))

Q_GRID = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
for q in Q_GRID:
    thr = {loc: pot_threshold(pot[loc], q) for loc in LOCATIONS}
    results[f'POT q={q}'] = (thr, evaluate(thr))

print(f"\n{'구성':<16} {'P':>5} {'R':>5} {'F1':>5} {'FP':>3}  임계값(11/14/17)   " +
      " ".join(f"{t[:6]:>7}" for t in ANOM_TYPES))
for label, (thr, r) in results.items():
    print(f"{label:<16} {r['prec']:.2f} {r['rec']:>5.2f} {r['f1']:>5.2f} {r['fp']:>3}  " +
          "/".join(f"{thr[l]:.2f}" for l in LOCATIONS) + "   " +
          " ".join(f"{v:>6.0%}" for v in r['type_recall'].values()))

# ---------- 앵커 민감도: 분위수 방식 vs POT ----------
# 분위수 방식에서 "기준 X%"를 바꾸는 것 = 임계값이 그대로 이동 (기준 선택이 결과를 지배)
# POT는 앵커 u를 85/90/95%로 바꿔도 같은 q의 z_q가 안정적인지 확인
print("\n앵커 민감도 (지점별, q=0.01 고정):")
print(f"{'지점':>4} {'분위수 85/90/95%':>24} {'POT z_q (u=85/90/95%)':>26}")
sens = {}
for loc in LOCATIONS:
    qs = [np.quantile(train_scores[loc], a) for a in (0.85, 0.90, 0.95)]
    zs = [pot_threshold(pot_fit(train_scores[loc], a), 0.01) for a in (0.85, 0.90, 0.95)]
    sens[loc] = (qs, zs)
    print(f"{loc:>4} " + "/".join(f"{v:7.3f}" for v in qs) + "  " +
          "/".join(f"{v:7.3f}" for v in zs) +
          f"   (분위수 변동폭 {max(qs)-min(qs):.3f} vs POT {max(zs)-min(zs):.3f})")

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))

# (1) 꼬리 적합 진단: 경험적 초과 생존확률 vs GPD 모델 (로그 스케일)
colors = {11: 'darkorange', 14: 'royalblue', 17: 'seagreen'}
for loc in LOCATIONS:
    f = pot[loc]
    exc = np.sort(f['exc'])
    emp_sf = 1.0 - np.arange(1, len(exc) + 1) / (len(exc) + 1)
    axs[0].semilogy(f['u'] + exc, emp_sf, 'o', ms=4, alpha=0.6, color=colors[loc],
                    label=f"loc {loc} empirical (Nu={f['Nu']})")
    xs = np.linspace(0, exc.max() * 1.3, 200)
    axs[0].semilogy(f['u'] + xs, genpareto.sf(xs, f['gamma'], 0, f['sigma']),
                    '-', color=colors[loc], alpha=0.8,
                    label=f"loc {loc} GPD (γ={f['gamma']:.2f})")
axs[0].set_xlabel('hybrid mean score'); axs[0].set_ylabel('P(X > x | X > u)')
axs[0].set_title('POT tail fit per location (train normal)')
axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)

# (2) q에 따른 성능 곡선 + 분위수 베이스라인
b = results['baseline q95'][1]
f1s = [results[f'POT q={q}'][1]['f1'] for q in Q_GRID]
ps = [results[f'POT q={q}'][1]['prec'] for q in Q_GRID]
rs = [results[f'POT q={q}'][1]['rec'] for q in Q_GRID]
axs[1].semilogx(Q_GRID, f1s, 'o-', color='darkorange', label='POT F1')
axs[1].semilogx(Q_GRID, ps, 's--', color='royalblue', label='POT Precision')
axs[1].semilogx(Q_GRID, rs, '^--', color='seagreen', label='POT Recall')
axs[1].axhline(b['f1'], color='darkorange', ls=':', alpha=0.7, label=f"q95 F1={b['f1']:.2f}")
axs[1].axhline(b['prec'], color='royalblue', ls=':', alpha=0.5)
axs[1].axhline(b['rec'], color='seagreen', ls=':', alpha=0.5)
axs[1].invert_xaxis()
axs[1].set_xlabel('target exceedance probability q (log)'); axs[1].set_ylim(0, 1.05)
axs[1].set_title('POT threshold: metrics vs q (dotted = 95% quantile baseline)')
axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3, which='both')

# (3) 유형별 recall: 베이스라인 vs 최고 F1의 POT
best_q = Q_GRID[int(np.argmax(f1s))]
bb = results[f'POT q={best_q}'][1]
types = list(ANOM_TYPES); x = np.arange(len(types)); w = 0.35
axs[2].bar(x - w / 2, [b['type_recall'][t] for t in types], w,
           color='gray', label=f"q95 baseline (F1={b['f1']:.2f}, FP={b['fp']})")
axs[2].bar(x + w / 2, [bb['type_recall'][t] for t in types], w,
           color='darkorange', label=f"POT q={best_q} (F1={bb['f1']:.2f}, FP={bb['fp']})")
axs[2].set_xticks(x); axs[2].set_xticklabels(types, rotation=12)
axs[2].set_ylim(0, 1.05); axs[2].set_ylabel('Recall')
axs[2].set_title('Per-type recall: quantile vs POT threshold')
axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUT}/18_evt_threshold_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n최고 F1 POT 구성: q={best_q} (F1 {max(f1s):.2f} vs 베이스라인 {b['f1']:.2f})")
print("저장: 18_evt_threshold_eval.png")
print("EVT12_OK")
