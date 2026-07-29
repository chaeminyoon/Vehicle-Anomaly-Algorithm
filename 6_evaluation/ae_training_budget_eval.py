# 개선 실험 ⑮: AE 학습 예산 — 10 에폭(③~⑪ 관례)은 수렴 전이었나
# 동기: ROADMAP 4-1이 "충분한 에폭 재학습"을 미결 항목으로 명시. ⑪까지의 AE는
#   검증용 10 에폭 관례를 유지해 왔는데, val_loss가 계속 내려가는 상태라면
#   하이브리드의 AE 축이 제 성능을 못 내고 있었을 가능성이 있다.
# 방법: ⑪ 구성 고정(하이브리드 mean, q95), AE만 10 -> 30 -> 60 에폭 연장 학습.
#   임계값 이동 효과와 분리하기 위해 (a) 고정 q95 성적, (b) PR-AUC(임계값 무관),
#   (c) 운영점 곡선(오탐 매칭 — ⑭의 교훈)을 모두 기록.
# 평가: 확대 평가셋 (지점·유형당 24개, 시드 43 — 실험 ⑨~⑭와 동일)
import numpy as np  # noqa

# ---------- 데이터/규칙/AE(10에폭) 준비 (실험 ⑪ 스크립트 재사용) ----------
exec(open('hybrid_score_eval.py').read().split("# ---------- 두 점수의 지점별 z-정규화")[0])
# 여기까지: 10에폭 학습된 model, rule_score, ae_error, T/S, train/test keys

STAGES = [10, 30, 60]

def collect_stage(tag):
    """현 시점 AE로 학습/평가/합성 전체의 (rule, ae) 원점수 수집."""
    rule_train = {loc: [rule_score(T[k], loc) for k in train_keys if k[0] == loc] for loc in LOCATIONS}
    ae_train = {loc: [ae_error(T[k], loc) for k in train_keys if k[0] == loc] for loc in LOCATIONS}
    rows = []  # (y, type, loc, rule_raw, ae_raw)
    for k in test_keys:
        rows.append((0, 'normal', k[0], rule_score(T[k], k[0]), ae_error(T[k], k[0])))
    for (loc, name), arr in S.items():
        rows.append((1, name.rsplit('_', 1)[0], loc, rule_score(arr, loc), ae_error(arr, loc)))
    print(f"[{tag}ep] 점수 수집 완료", flush=True)
    return rule_train, ae_train, rows

stage_data = {}
val_losses = {}
stage_data[10] = collect_stage(10)
for prev, nxt in zip(STAGES[:-1], STAGES[1:]):
    h = model.fit(X_train, X_train, epochs=nxt - prev, batch_size=64,
                  validation_split=0.1, shuffle=True, verbose=0)
    val_losses[nxt] = float(h.history['val_loss'][-1])
    print(f"AE {nxt} 에폭 연장 학습 (val_loss {val_losses[nxt]:.5f})", flush=True)
    stage_data[nxt] = collect_stage(nxt)

# ---------- 지점별 z-정규화 + 하이브리드 mean (⑪ 프로토콜) ----------
def hybrid_items(rule_train, ae_train, rows):
    zn = {loc: dict(rm=np.mean(rule_train[loc]), rs=np.std(rule_train[loc]) + 1e-9,
                    am=np.mean(ae_train[loc]), as_=np.std(ae_train[loc]) + 1e-9) for loc in LOCATIONS}
    tr_h = {loc: [0.5 * (((r - zn[loc]['rm']) / zn[loc]['rs']) + ((a - zn[loc]['am']) / zn[loc]['as_']))
                  for r, a in zip(rule_train[loc], ae_train[loc])] for loc in LOCATIONS}
    hs = np.array([0.5 * (((r - zn[loc]['rm']) / zn[loc]['rs']) + ((a - zn[loc]['am']) / zn[loc]['as_']))
                   for (_, _, loc, r, a) in rows])
    return tr_h, hs

y = np.array([r[0] for r in stage_data[10][2]])
types = np.array([r[1] for r in stage_data[10][2]])
locs_arr = np.array([r[2] for r in stage_data[10][2]])
N_NORMAL = int((y == 0).sum())

def metrics(pred):
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tr = {t: float(np.mean(pred[types == t])) for t in ANOM_TYPES}
    return dict(f1=f1, p=prec, r=rec, fp=fp, tr=tr)

results = {}
for ep in STAGES:
    rule_train, ae_train, rows = stage_data[ep]
    tr_h, hs = hybrid_items(rule_train, ae_train, rows)
    ap = average_precision_score(y, hs)
    # 고정 q95
    thr95 = {loc: np.percentile(tr_h[loc], 95) for loc in LOCATIONS}
    m95 = metrics(np.array([s > thr95[l] for s, l in zip(hs, locs_arr)]))
    # 운영점 곡선 (q 스윕)
    sweep = []
    for q in np.arange(80, 99.5, 0.5):
        thr = {loc: np.percentile(tr_h[loc], q) for loc in LOCATIONS}
        sweep.append((q, metrics(np.array([s > thr[l] for s, l in zip(hs, locs_arr)]))))
    results[ep] = dict(ap=ap, m95=m95, sweep=sweep)
    print(f"AE {ep}ep: PR-AUC {ap:.3f} | q95: F1 {m95['f1']:.2f} P {m95['p']:.2f} "
          f"R {m95['r']:.2f} FP {m95['fp']} | " +
          " ".join(f"{t}:{v:.0%}" for t, v in m95['tr'].items()), flush=True)

# 오탐 매칭: 10ep의 FP(기준)와 같은 오탐에서 각 에폭의 F1
base_fp = results[10]['m95']['fp']
print(f"\n오탐 매칭 (FP={base_fp} 기준):")
for ep in STAGES:
    close = min(results[ep]['sweep'], key=lambda qm: abs(qm[1]['fp'] - base_fp))
    print(f"  {ep}ep @ FP {close[1]['fp']} (q{close[0]}): F1 {close[1]['f1']:.2f} R {close[1]['r']:.2f} | " +
          " ".join(f"{t}:{v:.0%}" for t, v in close[1]['tr'].items()))

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
colors = {10: 'gray', 30: 'royalblue', 60: 'darkorange'}

for ep in STAGES:
    sw = results[ep]['sweep']
    axs[0].plot([m['fp'] for _, m in sw], [m['f1'] for _, m in sw], '-', lw=2,
                color=colors[ep], label=f"{ep} epochs (PR-AUC {results[ep]['ap']:.3f})")
    m95 = results[ep]['m95']
    axs[0].scatter([m95['fp']], [m95['f1']], marker='*', s=180, color=colors[ep], zorder=5)
axs[0].set_xlabel(f'false positives (of {N_NORMAL} normal tracks)')
axs[0].set_ylabel('F1'); axs[0].set_ylim(0.6, 1.0)
axs[0].set_title('Operating curves by AE training budget\n(stars = fixed q95 operating point)')
axs[0].legend(fontsize=9); axs[0].grid(alpha=0.3)

x = np.arange(len(ANOM_TYPES)); w = 0.26
for i, ep in enumerate(STAGES):
    m95 = results[ep]['m95']
    axs[1].bar(x + (i - 1) * w, [m95['tr'][t] for t in ANOM_TYPES], w, color=colors[ep],
               label=f"{ep}ep: F1 {m95['f1']:.2f} (FP {m95['fp']})")
axs[1].set_xticks(x); axs[1].set_xticklabels(ANOM_TYPES, rotation=12)
axs[1].set_ylim(0, 1.05); axs[1].set_ylabel('Recall @ q95')
axs[1].set_title('Per-type recall at the default operating point (q95)')
axs[1].legend(fontsize=8)

axs[2].plot(STAGES, [results[ep]['ap'] for ep in STAGES], 'o-', color='royalblue', label='PR-AUC (threshold-free)')
axs[2].plot(STAGES, [results[ep]['m95']['f1'] for ep in STAGES], 's-', color='darkorange', label='F1 @ q95')
axs[2].plot(STAGES, [results[ep]['m95']['r'] for ep in STAGES], '^-', color='seagreen', label='Recall @ q95')
axs[2].set_xticks(STAGES); axs[2].set_xlabel('AE training epochs')
axs[2].set_ylim(0.6, 1.0); axs[2].grid(alpha=0.3); axs[2].legend(fontsize=9)
axs[2].set_title('Training budget vs detection quality')
plt.tight_layout()
plt.savefig(f"{OUT}/21_ae_training_budget.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 21_ae_training_budget.png")
print("BUDGET_OK")
