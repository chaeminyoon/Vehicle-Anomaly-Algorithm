# 개선 실험 ⑬: 예측 기반 이상 점수 — AE 과잉일반화 회피 시도
# 동기 (ROADMAP 4-3): 재구성(AE)은 이상 궤적까지 "잘 복원"해버리는 과잉일반화가
#   알려진 약점. 예측(seq2seq)은 과거 40스텝에서 미래 10스텝을 맞혀야 하므로
#   행동 변화(급정거·사행·횡단 개시)가 예측 오차로 즉시 드러난다는 가설을 검증.
# 검증 질문:
#   1) 예측 오차 단독이 AE 재구성 오차보다 나은가? (유형별로 어디가 다른가)
#   2) 하이브리드에서 AE를 예측기로 교체/병행하면 ⑪(F1 0.85)을 넘는가?
#   3) 규칙이 놓치는 이상을 예측기가 AE보다 더 많이 회수하는가? (상보성)
# 구성: A2+D3 고정, 점수원만 변경. 모델 규모(LSTM 64)·에폭(10)·시드(42)는 ⑪의
#   AE와 동일하게 맞춰 "재구성 vs 예측" 외 변인을 제거.
# 평가: 확대 평가셋 (지점·유형당 24개, 시드 43 — 실험 ⑨⑩⑪⑫와 동일)

# ---------- 데이터/규칙/AE 파이프라인 준비 (실험 ⑪ 스크립트 재사용) ----------
# hybrid_score_eval.py 를 결합 비교 직전까지 exec ->
#   T/S(궤적), items(rule_z, ae_z), train_z, znorm, scalers, to_sequences 확보
exec(open('hybrid_score_eval.py').read().split("# ---------- 결합 방식 비교")[0])

# ---------- 예측기: 과거 W_IN -> 미래 H 스텝 (seq2seq) ----------
W_IN, H = 40, 10  # SEQ_LEN(50) 윈도를 40/10으로 분할 — 윈도 생성 자체는 ⑪과 동일
tf.keras.utils.set_random_seed(42)

pred_model = Sequential([
    LSTM(64, activation='tanh', recurrent_activation='sigmoid',
         input_shape=(W_IN, 4), return_sequences=False),
    RepeatVector(H),
    LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
    TimeDistributed(Dense(4)),
])
pred_model.compile(optimizer='adam', loss='mse')
pred_model.fit(X_train[:, :W_IN], X_train[:, W_IN:], epochs=10, batch_size=64,
               validation_split=0.1, shuffle=True, verbose=0)
print(f"예측기 학습 완료: in {W_IN} -> out {H} (시퀀스 {X_train.shape[0]}개)")

def pred_error(arr, loc):
    seqs = to_sequences(scalers[loc].transform(add_velocity(arr)))
    if len(seqs) == 0:
        return 0.0
    fut = pred_model.predict(seqs[:, :W_IN], verbose=0)
    return float(np.max(np.mean(np.square(seqs[:, W_IN:] - fut), axis=(1, 2))))  # max 집계 (⑪과 동일)

# ---------- 예측 오차의 지점별 z-정규화 (학습 정상 기준 — ⑪ 프로토콜 동일) ----------
print("예측 오차 산출 중 (학습/평가/합성)...")
pred_train = {loc: [pred_error(T[k], loc) for k in train_keys if k[0] == loc] for loc in LOCATIONS}
pnorm = {loc: (np.mean(pred_train[loc]), np.std(pred_train[loc]) + 1e-9) for loc in LOCATIONS}

def pred_z(arr, loc):
    return (pred_error(arr, loc) - pnorm[loc][0]) / pnorm[loc][1]

# items: (y, type, loc, rule_z, ae_z) -> pred_z를 같은 순서로 추가
pz_items = []
for k in test_keys:
    pz_items.append(pred_z(T[k], k[0]))
for (loc, name), arr in S.items():
    pz_items.append(pred_z(arr, loc))
items = [it + (pz,) for it, pz in zip(items, pz_items)]  # (y, type, loc, rz, az, pz)
train_z3 = {loc: [(r, a, (p - pnorm[loc][0]) / pnorm[loc][1])
                  for (r, a), p in zip(train_z[loc], pred_train[loc])] for loc in LOCATIONS}

# 점수 덤프 — 융합 방식 오프라인 탐색용 (재학습 없이 결합만 재실험)
np.savez(f"{OUT}/pred_eval_scores.npz",
         y=np.array([it[0] for it in items]),
         types=np.array([it[1] for it in items]),
         locs=np.array([it[2] for it in items]),
         rz=np.array([it[3] for it in items]),
         az=np.array([it[4] for it in items]),
         pz=np.array([it[5] for it in items]),
         **{f"train_{loc}": np.array(train_z3[loc]) for loc in LOCATIONS})
print(f"점수 덤프 저장: pred_eval_scores.npz ({len(items)} items)")

# ---------- 결합 방식 비교 (⑪ 기준선 포함, 동일 보정 프로토콜) ----------
FUSIONS3 = {
    'AE only':               lambda r, a, p: a,
    'pred only':             lambda r, a, p: p,
    'hybrid mean (exp.11)':  lambda r, a, p: 0.5 * (r + a),
    'rule+pred mean':        lambda r, a, p: 0.5 * (r + p),
    'rule+AE+pred mean':     lambda r, a, p: (r + a + p) / 3.0,
    'rule+max(AE,pred)':     lambda r, a, p: 0.5 * (r + max(a, p)),
}
res3 = {}
y_true = np.array([it[0] for it in items])
for label, fuse in FUSIONS3.items():
    thr = {loc: np.percentile([fuse(r, a, p) for r, a, p in train_z3[loc]], 95) for loc in LOCATIONS}
    y_score = np.array([fuse(it[3], it[4], it[5]) for it in items])
    pred = np.array([fuse(it[3], it[4], it[5]) > thr[it[2]] for it in items])
    ap = average_precision_score(y_true, y_score)
    tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0)))
    fn_ = int(np.sum(~pred & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    type_recall = {t: float(np.mean(pred[[i for i, it in enumerate(items) if it[1] == t]]))
                   for t in ANOM_TYPES}
    res3[label] = dict(ap=ap, prec=prec, rec=rec, f1=f1, fp=fp, type_recall=type_recall, pred=pred)
    print(f"{label}: PR-AUC {ap:.3f} | P {prec:.2f} / R {rec:.2f} / F1 {f1:.2f} (FP {fp}) | " +
          " ".join(f"{t}:{v:.0%}" for t, v in type_recall.items()))

# ---------- 상보성: 규칙 미탐을 AE vs 예측기가 각각 얼마나 회수하는가 ----------
rule_pred_only = {loc: np.percentile([r for r, a, p in train_z3[loc]], 95) for loc in LOCATIONS}
rp = np.array([it[3] > rule_pred_only[it[2]] for it in items])
ap_pred = res3['AE only']['pred']
pp_pred = res3['pred only']['pred']
anom = y_true == 1
missed = anom & ~rp
print(f"\n규칙 미탐 이상 {int(missed.sum())}개 회수: AE {int((missed & ap_pred).sum())}개 / "
      f"예측기 {int((missed & pp_pred).sum())}개 / 합집합 {int((missed & (ap_pred | pp_pred)).sum())}개")
by_type = {}
for t in ANOM_TYPES:
    idx = np.array([it[1] == t for it in items])
    m = missed & idx
    by_type[t] = (int(m.sum()), int((m & ap_pred).sum()), int((m & pp_pred).sum()))
print("유형별 (규칙 미탐 / AE 회수 / 예측기 회수):", by_type)
pred_fp_only = int(((y_true == 0) & pp_pred & ~rp).sum())
print(f"예측기만의 추가 오탐(정상): {pred_fp_only}개")

# ---------- 시각화 ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.18
SHOW = ['AE only', 'pred only', 'hybrid mean (exp.11)', 'rule+AE+pred mean']
colors = ['royalblue', 'teal', 'darkorange', 'crimson']
for i, label in enumerate(SHOW):
    axs[0].bar(x + (i - 1.5) * w, [res3[label]['type_recall'][t] for t in types], w,
               color=colors[i], label=f"{label} (F1={res3[label]['f1']:.2f})")
axs[0].set_xticks(x); axs[0].set_xticklabels(types, rotation=12)
axs[0].set_ylim(0, 1.05); axs[0].set_ylabel('Recall (calibrated thr)')
axs[0].set_title('Reconstruction (AE) vs prediction error, and fusions')
axs[0].legend(fontsize=8)

az_n = [it[4] for it in items if it[0] == 0]; pz_n = [it[5] for it in items if it[0] == 0]
az_a = [it[4] for it in items if it[0] == 1]; pz_a = [it[5] for it in items if it[0] == 1]
axs[1].scatter(az_n, pz_n, s=12, color='gray', alpha=0.5, label='normal')
axs[1].scatter(az_a, pz_a, s=12, color='crimson', alpha=0.6, label='anomaly')
r_all = np.corrcoef(np.array([it[4] for it in items]), np.array([it[5] for it in items]))[0, 1]
axs[1].annotate(f"Pearson r = {r_all:.2f} -> prediction sees largely the SAME anomalies as AE",
                xy=(0.03, 0.95), xycoords='axes fraction', fontsize=9, va='top')
axs[1].set_xlabel('AE z-score (reconstruction)'); axs[1].set_ylabel('prediction z-score')
axs[1].set_title('Do prediction and reconstruction see different anomalies?')
axs[1].legend(); axs[1].grid(alpha=0.3)

labels = list(FUSIONS3)
axs[2].plot(labels, [res3[l]['f1'] for l in labels], 'o-', color='darkorange', label='F1')
axs[2].plot(labels, [res3[l]['ap'] for l in labels], 's-', color='royalblue', label='PR-AUC')
axs[2].plot(labels, [res3[l]['rec'] for l in labels], '^-', color='seagreen', label='Recall')
axs[2].set_ylim(0, 1.02); axs[2].grid(alpha=0.3); axs[2].legend()
plt.setp(axs[2].get_xticklabels(), rotation=14, fontsize=7.5)
axs[2].set_title('Overall metrics per fusion')
plt.tight_layout()
plt.savefig(f"{OUT}/19_prediction_score_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 19_prediction_score_eval.png")
print("PRED_OK")
