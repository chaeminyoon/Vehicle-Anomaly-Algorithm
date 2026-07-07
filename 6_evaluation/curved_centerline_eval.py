# 개선 실험 ⑨: 곡선 중심선(호길이) 좌표계 — 실험 ⑧이 노출한 1D 직선 차선 모델 한계 해소
# - 학습 궤적 밀도에서 도로 중심선을 강건 추정(주축 s 구간별 횡좌표 중앙값 -> 3차 다항)
# - 궤적을 (호길이 s_arc, 법선 오프셋 d_n)로 "직선화"한 뒤 기존 파이프라인 그대로 적용
# - 2x2 ablation: 좌표 평면(이미지 / 실측 미터 ⑧) x 차선 모델(직선 / 곡선 중심선)
#   (지점 17은 위성사진이 없어 두 평면 모두 이미지 좌표)
import numpy as np  # noqa: F811 (exec 준비부와 중복 무해)

# ---------- 데이터/H/신뢰구역/파이프라인 준비 (실험 ⑧ 스크립트 재사용) ----------
exec(open('measured_homography_eval.py').read().split("\nCONFIGS = [")[0])

# ---------- 평가셋 확대: 유형·지점당 6 -> 24 (탐지율 양자 17%p -> 4%p) ----------
# 실험 ③~⑧의 소규모 셋은 지점·유형당 6개라 ±1개 차이로 수치가 크게 요동
# -> 곡선 모델의 효과 판정을 위해 별도 시드(43)로 확대 재생성
N_PER_TYPE = 24
rng2 = np.random.RandomState(43)
picks = {}
for loc in LOCATIONS:
    pool = test_by_loc[loc]
    for tname in ANOM_TYPES:
        idx = rng2.choice(len(pool), size=min(N_PER_TYPE, len(pool)), replace=False)
        picks[(loc, tname)] = [pool[i] for i in idx]
synth = {}
for coord in ['center', 'bottom']:
    synth[coord] = {}
    for (loc, tname), pks in picks.items():
        fn = ANOM_TYPES[tname]
        for i, k in enumerate(pks):
            synth[coord][(loc, f"{tname}_{i}")] = fn(tracks[coord][k], loc_range[loc])
print(f"확대 평가셋: 합성이상 {len(synth['bottom'])}개 (지점·유형당 최대 {N_PER_TYPE})")

# ---------- 곡선 중심선 모델 ----------
CURV_MIN = 1.01   # 곡선/직선 길이비가 이보다 커야 곡선 모델 채택 (미미한 곡률은 잡음만 추가)
CURV_MAX = 1.5    # 이보다 크면 fit 붕괴로 간주 -> 직선 폴백

def build_centerlines(base_tf):
    """base_tf로 사상된 평면에서 지점별 중심선(3차 다항 d=f(s)) + 호길이 테이블 추정
    - 방향 필터: 주축 정렬(|cos|>0.6) 궤적만 사용 -> 교차로 횡단 궤적의 오염 방지
    - 곡률 유의성: 길이비가 [CURV_MIN, CURV_MAX] 밖이면 직선 폴백(coef=0)"""
    models = {}
    for loc in LOCATIONS:
        arrs = []
        for k in train_keys:
            if k[0] != loc:
                continue
            a = base_tf(smooth_track(tracks['bottom'][k]), loc)
            if a is not None:
                arrs.append(a)
        P = np.vstack(arrs)
        c = P.mean(axis=0)
        cov = np.cov((P - c).T)
        _, V = np.linalg.eigh(cov)
        u = V[:, -1]; v = np.array([-u[1], u[0]])
        # 주축 정렬 궤적만으로 fit (본선 흐름; 역방향 |cos|도 허용)
        aligned = []
        for a in arrs:
            disp = a[-1] - a[0]
            n = np.linalg.norm(disp)
            if n > 1e-6 and abs(disp @ u) / n > 0.6:
                aligned.append(a)
        # 궤적별 자기 오프셋(중앙값 d)을 제거한 "곡률 성분"만 합산 -> 차로별 관측 구간
        # 차이(밀도 이동)가 중심선을 끌고 가는 것을 방지 (궤적은 차선에 평행하다는 성질 이용)
        s_list, d_list = [], []
        for a in (aligned if aligned else arrs):
            sa = (a - c) @ u; da = (a - c) @ v
            d_list.append(da - np.median(da))
            s_list.append(sa)
        s = np.concatenate(s_list); d = np.concatenate(d_list)
        qs = np.linspace(s.min(), s.max(), 41)
        mids, meds = [], []
        for i in range(40):
            m = (s >= qs[i]) & (s < qs[i + 1])
            if m.sum() >= 30:
                mids.append((qs[i] + qs[i + 1]) / 2)
                meds.append(np.median(d[m]))
        coef = np.polyfit(mids, meds, 3) if len(mids) >= 8 else np.zeros(4)
        sg = np.linspace(s.min(), s.max(), 500)
        arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(sg), np.diff(np.polyval(coef, sg))))])
        ratio = arc[-1] / (sg[-1] - sg[0])
        curved = CURV_MIN < ratio < CURV_MAX
        if not curved:  # 직선 폴백
            coef = np.zeros(4)
            arc = sg - sg[0]
        curv_span = np.polyval(coef, sg).max() - np.polyval(coef, sg).min()
        models[loc] = dict(c=c, u=u, v=v, coef=coef, smin=s.min(), smax=s.max(),
                           sg=sg, arc=arc, curved=curved)
        print(f"  지점 {loc}: 길이비 {ratio:.4f} -> {'곡선' if curved else '직선 폴백'} "
              f"(횡변위 폭 {curv_span:.1f}, 정렬궤적 {len(aligned)}/{len(arrs)})")
    return models

def straighten(a, loc, cm):
    if a is None:
        return None
    m = cm[loc]
    s = (a - m['c']) @ m['u']; d = (a - m['c']) @ m['v']
    sc = np.clip(s, m['smin'], m['smax'])  # fit 구간 밖 외삽 방지
    dc = np.polyval(m['coef'], sc)
    dp = np.polyval(np.polyder(m['coef']), sc)
    d_n = (d - dc) / np.sqrt(1 + dp ** 2)   # 법선 오프셋 (완경사 근사)
    s_arc = np.interp(sc, m['sg'], m['arc']) + (s - sc)  # 구간 밖은 선형 연장
    return np.column_stack([s_arc, d_n])

identity_tf = lambda a, loc: a

print("\n[중심선 추정 — 이미지 평면]")
cl_img = build_centerlines(identity_tf)
print("[중심선 추정 — 실측 미터 평면]")
cl_met = build_centerlines(to_metric)

# ---------- 2x2 ablation ----------
CONFIGS = [
    ('C0 image/straight (A2)', identity_tf),
    ('C1 image/curved',        lambda a, loc: straighten(a, loc, cl_img)),
    ('C2 metric/straight (B1)', to_metric),
    ('C3 metric/curved',       lambda a, loc: straighten(to_metric(a, loc), loc, cl_met)),
]
res = {}
for label, tfx in CONFIGS:
    r = run_pipeline('bottom', True, tfx)
    res[label] = r
    print(f"\n{label}: PR-AUC {r['ap']:.3f} | P {r['prec']:.2f} / R {r['rec']:.2f} / F1 {r['f1']:.2f} | " +
          " ".join(f"{t}:{v:.0%}" for t, v in r['type_recall'].items()))
    for loc in LOCATIONS:
        print(f"  지점 {loc}: " + " ".join(f"{t}:{v:.0%}" for t, v in r['loc_type'][loc].items()))

# ---------- 물리량 (C3, 실측 보정 지점) ----------
print("\n[물리량 — 미터 평면 + 곡선 중심선]")
r3 = res['C3 metric/curved']
for loc in homographies_gt:
    lm = r3['lane_models'][loc]
    offs = []
    for k in train_keys:
        if k[0] != loc or k not in r3['T']:
            continue
        offs.append(r3['lane_feats'](r3['T'][k], loc)[0])
    offs = np.concatenate(offs)
    print(f"지점 {loc}: 차선간격 {lm['spacing']:.2f}m (기대 3.0~3.5m) | "
          f"차선 {len(lm['centers'])}개 | 오프셋 std {offs.std() * lm['spacing']:.2f}m")

# ---------- 시각화 ----------
fig = plt.figure(figsize=(20, 5.5))
# (1) 지점 11 미터 평면: 궤적 + 추정 중심선
ax = fig.add_subplot(1, 3, 1)
m = cl_met[11]
for k in [kk for kk in train_keys if kk[0] == 11][:80]:
    a = to_metric(smooth_track(tracks['bottom'][k]), 11)
    if a is not None:
        ax.plot(a[:, 0], a[:, 1], color='gray', alpha=0.3, linewidth=0.6)
sg = m['sg']
cline = m['c'][None, :] + np.outer(sg, m['u']) + np.outer(np.polyval(m['coef'], sg), m['v'])
ax.plot(cline[:, 0], cline[:, 1], color='crimson', linewidth=2, label='fitted centerline')
ax.set_aspect('equal'); ax.legend(); ax.set_title('Loc 11 metric plane: curved centerline fit')

# (2) 지점 11 직선화 좌표: (s_arc, d_n)
ax = fig.add_subplot(1, 3, 2)
for k in [kk for kk in train_keys if kk[0] == 11][:80]:
    a = straighten(to_metric(smooth_track(tracks['bottom'][k]), 11), 11, cl_met)
    if a is not None:
        ax.plot(a[:, 0], a[:, 1], color='steelblue', alpha=0.3, linewidth=0.6)
ax.set_xlabel('arc length s (m)'); ax.set_ylabel('normal offset d (m)')
ax.set_title('Loc 11 straightened: lanes become horizontal bands')

# (3) 유형별 탐지율 (4구성)
ax = fig.add_subplot(1, 3, 3)
types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.2
colors = ['gray', 'seagreen', 'royalblue', 'darkorange']
for i, (label, _) in enumerate(CONFIGS):
    ax.bar(x + (i - 1.5) * w, [res[label]['type_recall'][t] for t in types], w,
           color=colors[i], label=f"{label} (F1={res[label]['f1']:.2f})")
ax.set_xticks(x); ax.set_xticklabels(types, rotation=12)
ax.set_ylim(0, 1.05); ax.set_ylabel('Recall (calibrated thr)')
ax.set_title('2x2: plane x lane model'); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/15_curved_centerline_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 15_curved_centerline_eval.png")
print("CURVE_OK")
