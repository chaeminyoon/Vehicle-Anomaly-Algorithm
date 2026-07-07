# 개선 실험 ⑧: 위성사진 대응점 기반 실측 호모그래피 -> 미터 좌표계 평가
# - 대응점: make_correspondence_tool.py 로 생성한 annotate_{loc}.html에서 수동 지정,
#   JSON을 analysis_results/homography_gt/points_{loc}.json 에 저장
# - H: RANSAC(DLT), 스케일: 위성사진 스케일 측정선(실거리 m) 평균 -> 완전 미터 좌표
# - 비교: B0 = 실험 ⑦ 채택 구성(하단중앙점+스무딩, 이미지 좌표)
#         B1 = 같은 입력을 실측 호모그래피로 미터 평면 변환 (지점 17은 위성사진이
#              없어 두 구성 모두 이미지 좌표 -> 지점 11/14의 짝비교가 판정 대상)
import json
import os

import cv2

# ---------- 데이터/합성셋/전처리 준비 (실험 ⑦과 동일) ----------
exec(open('bottom_center_eval.py').read().split("# ---------- 규칙 기반 파이프라인")[0])

GT_DIR = f"{OUT}/homography_gt"

homographies_gt = {}
for loc in LOCATIONS:
    pf = f"{GT_DIR}/points_{loc}.json"
    if not os.path.exists(pf):
        continue
    d = json.load(open(pf))
    if len(d['pairs']) < 4:
        print(f"지점 {loc}: 대응점 {len(d['pairs'])}개 (<4) -> 생략")
        continue
    src = np.float32([p['cctv'] for p in d['pairs']])
    dst = np.float32([p['sat'] for p in d['pairs']])
    # 대응점이 적을 때 RANSAC은 근거리 군집만 남겨 퇴화하기 쉬움 -> 8개 미만은 전체 최소제곱
    if len(src) >= 8:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
        inl = int(mask.sum())
    else:
        H, _ = cv2.findHomography(src, dst, 0)
        inl = len(src)
    # H는 스케일(부호 포함) 임의 -> 데이터 영역에서 w>0이 되도록 부호 정규화
    w_c = np.array([*src.mean(axis=0), 1.0]) @ H.T
    if w_c[2] < 0:
        H = -H
    proj = cv2.perspectiveTransform(src[None].astype(np.float64), H)[0]
    res = np.linalg.norm(proj - dst, axis=1)
    print(f"지점 {loc} 점별 잔차(px): {np.round(res, 1)}")
    if not d['scale_lines']:
        print(f"지점 {loc}: 스케일 측정선 없음 -> 생략 (스케일 모드로 실거리 지정 필요)")
        continue
    mpp = np.mean([sl['meters'] / (np.linalg.norm(np.array(sl['p1']) - np.array(sl['p2'])) + 1e-9)
                   for sl in d['scale_lines']])
    homographies_gt[loc] = dict(H=H, m_per_px=mpp)
    print(f"지점 {loc}: 대응점 {len(src)}개 (inlier {inl}), 잔차 중앙값 {np.median(res):.1f}px "
          f"(={np.median(res) * mpp:.2f}m), 스케일 {mpp:.3f} m/px x{len(d['scale_lines'])}")

if not homographies_gt:
    print("대응점 JSON이 없습니다. annotate_{11,14}.html에서 지정 후 "
          f"{GT_DIR}/points_{{loc}}.json 으로 저장하세요.")
    raise SystemExit(0)

# 신뢰 구역: 횡방향(이상 신호 방향) 잡음 = 지터(px) x 횡방향 확대율(m/px) <= NOISE_M
# - 종방향 확대율은 원근상 근거리에서도 크므로(특히 640px 지점 11) 기준에서 제외
# - 지터는 지점별로 실측 (원 궤적 vs Savitzky-Golay 잔차 RMS)
NOISE_M = 0.25

JITTER = {}
for loc in LOCATIONS:
    r_ = [np.sqrt(np.mean((tracks['bottom'][k] - smooth_track(tracks['bottom'][k])) ** 2))
          for k in keys if k[0] == loc]
    JITTER[loc] = float(np.median(r_))
print("지점별 지터 RMS(px):", {l: round(j, 2) for l, j in JITTER.items()})

def to_metric(a, loc):
    h = homographies_gt.get(loc)
    if h is None:
        return a  # 위성사진 없는 지점은 이미지 좌표 유지 (지점별 파이프라인이라 무방)
    H = h['H']
    hom = np.hstack([a, np.ones((len(a), 1))]) @ H.T
    w = hom[:, 2]
    # 지평선(w->0) 근접·후방(w<0) 점 제외 + 횡방향 확대율 상한으로 원거리 잡음 차단
    valid = w > 1e-6
    mag_lat = np.full(len(a), np.inf)
    A2 = H[:2, :2]; ab = H[2, :2]
    pts = hom[valid, :2] / w[valid, None]
    # 야코비안 최소특이값 = 횡방향(도로 가로) 방향의 m/px 확대율
    sv_min = np.array([np.linalg.svd(A2 - np.outer(p, ab), compute_uv=False)[-1] for p in pts])
    mag_lat[valid] = sv_min / w[valid] * h['m_per_px']
    keep = valid.copy()
    keep[valid] &= (mag_lat[valid] * JITTER[loc] <= NOISE_M)
    if keep.sum() >= 15:  # 유효(근·중거리) 부분궤적만 사용
        hom_k = hom[keep]
        return hom_k[:, :2] / hom_k[:, 2:3] * h['m_per_px']
    return None  # 궤적 전체가 보정 신뢰 구역 밖 -> 미터 평면에서는 분석 불가

# ---------- 규칙 기반 파이프라인 (실험 ⑦과 동일, 좌표 변환만 인자화) ----------
def run_pipeline(coord, use_smooth, tf_extra):
    trk, syn_in = tracks[coord], synth[coord]
    pre = smooth_track if use_smooth else (lambda a: a.copy())

    def tf_(a, loc):
        return tf_extra(pre(a), loc)

    # None = 보정 신뢰 구역(확대율 상한) 밖 궤적 -> 학습에서 제외, 평가에서는 '판정 불가(정상 처리)'
    T = {k: v for k, v in ((k, tf_(trk[k], k[0])) for k in trk) if v is not None}
    S = {sk: tf_(arr, sk[0]) for sk, arr in syn_in.items()}
    n_drop = len(trk) - len(T)
    if n_drop:
        by = defaultdict(int)
        for k in trk:
            if k not in T:
                by[k[0]] += 1
        print(f"  신뢰 구역 밖 제외: {n_drop}개 궤적 {dict(by)}")

    lane_models_, dir_fields_ = {}, {}
    for loc in LOCATIONS:
        tkeys = [k for k in train_keys if k[0] == loc and k in T]
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
            smoo = gaussian_filter1d(hist.astype(float), 2)
            bin_w = edges[1] - edges[0]
            min_sep = 0.02 * (d_dir.max() - d_dir.min())
            peaks, _ = find_peaks(smoo, distance=max(int(min_sep / bin_w), 1), height=smoo.max() * 0.05)
            centers_list.extend((edges[peaks] + edges[peaks + 1]) / 2)
            dir_list.extend([sgn] * len(peaks))
        if not centers_list:
            centers_list = [0.0]
        order = np.argsort(centers_list)
        centers = np.array(centers_list)[order]
        spacing = np.median(np.diff(centers)) if len(centers) > 1 else \
            np.concatenate([(T[k] - center) @ v for k in tkeys]).std()
        lane_models_[loc] = dict(center=center, u=u, v=v, centers=centers, spacing=spacing)

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
        s = (arr - lm['center']) @ lm['u']
        d = (arr - lm['center']) @ lm['v']
        dd = np.diff(d); d1 = d[1:]
        li = np.argmin(np.abs(d1[:, None] - lm['centers'][None, :]), axis=1)
        off = (d1 - lm['centers'][li]) / lm['spacing']
        ds = np.diff(s)
        return off, np.hypot(ds, dd) / lm['spacing']

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
        conf = (f['coherence'][iy, ix] > 0.7) & (nrm > np.percentile(nrm, 20))
        return -align[conf].mean() if conf.any() else 0.0

    speed_q_ = {loc: np.percentile(np.concatenate(
        [lane_feats(T[k], loc)[1] for k in train_keys if k[0] == loc and k in T]), 5) for loc in LOCATIONS}

    def rules(arr, loc):
        off, sp = lane_feats(arr, loc)
        return np.array([ww_score(arr, loc), np.abs(off).max(), off.std(),
                         (sp < speed_q_[loc]).mean()])

    tr_rules = defaultdict(list)
    for k in train_keys:
        if k in T:
            tr_rules[k[0]].append(rules(T[k], k[0]))
    stats = {loc: (np.mean(tr_rules[loc], axis=0), np.std(tr_rules[loc], axis=0) + 1e-9)
             for loc in LOCATIONS}

    def score(arr, loc):
        z = (rules(arr, loc) - stats[loc][0]) / stats[loc][1]
        return float(z.max())

    thr = {loc: np.percentile([score(T[k], loc) for k in train_keys if k[0] == loc and k in T], 95)
           for loc in LOCATIONS}

    y_true, y_score, meta = [], [], []
    NA = -999.0  # 신뢰 구역 밖 -> 판정 불가(정상 취급, B1의 커버리지 한계를 정직하게 반영)
    for k in test_keys:
        y_true.append(0); y_score.append(score(T[k], k[0]) if k in T else NA)
        meta.append(('normal', k[0]))
    for (loc, name), arr in S.items():
        y_true.append(1); y_score.append(score(arr, loc) if arr is not None else NA)
        meta.append((name.rsplit('_', 1)[0], loc))
    y_true, y_score = np.array(y_true), np.array(y_score)
    ap = average_precision_score(y_true, y_score)
    pred = np.array([sc > thr[m[1]] for sc, m in zip(y_score, meta)])
    tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0)))
    fn_ = int(np.sum(~pred & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    type_recall = {t: float(np.mean(pred[[i for i, m in enumerate(meta) if m[0] == t]]))
                   for t in ANOM_TYPES}
    # 지점별 유형 탐지율 (실측 보정 지점의 짝비교용)
    loc_type = {loc: {t: float(np.mean(pred[[i for i, m in enumerate(meta)
                                             if m[0] == t and m[1] == loc]]))
                      for t in ANOM_TYPES} for loc in LOCATIONS}
    return dict(ap=ap, prec=prec, rec=rec, f1=f1, type_recall=type_recall,
                loc_type=loc_type, T=T, lane_models=lane_models_, lane_feats=lane_feats)

CONFIGS = [
    ('B0 image coords (A2)', lambda a, loc: a),
    ('B1 satellite metric',  to_metric),
]
res = {}
for label, tfx in CONFIGS:
    r = run_pipeline('bottom', True, tfx)
    res[label] = r
    print(f"\n{label}: PR-AUC {r['ap']:.3f} | P {r['prec']:.2f} / R {r['rec']:.2f} / F1 {r['f1']:.2f} | " +
          " ".join(f"{t}:{v:.0%}" for t, v in r['type_recall'].items()))
    for loc in LOCATIONS:
        tag = " (실측보정)" if loc in homographies_gt and label.startswith('B1') else ""
        print(f"  지점 {loc}{tag}: " + " ".join(f"{t}:{v:.0%}" for t, v in r['loc_type'][loc].items()))

# ---------- 물리량 검증 (B1, 실측 보정 지점) ----------
print("\n[물리량 검증 — 미터 평면]")
r1 = res['B1 satellite metric']
for loc in homographies_gt:
    lm = r1['lane_models'][loc]
    sp_lane = lm['spacing']  # m (미터 좌표계에서 추정된 차선 간격)
    offs, sps = [], []
    for k in train_keys:
        if k[0] != loc or k not in r1['T']:
            continue
        o, sp = r1['lane_feats'](r1['T'][k], loc)
        offs.append(o); sps.append(sp)
    offs, sps = np.concatenate(offs), np.concatenate(sps)
    mv = sps[sps > np.percentile(sps, 50)]  # 정체 프레임 제외: 상위 절반(주행 중)
    kmh_med = np.median(sps) * sp_lane * FPS * 3.6
    kmh_p85 = np.percentile(sps, 85) * sp_lane * FPS * 3.6
    print(f"지점 {loc}: 차선간격 추정 {sp_lane:.2f}m (기대 3.0~3.5m) | "
          f"오프셋 std {offs.std() * sp_lane:.2f}m | "
          f"속도 중앙값 {kmh_med:.0f} / 85% {kmh_p85:.0f} km/h")

# ---------- 시각화: 위성사진 위 변환 궤적 + 유형별 탐지율 ----------
n_gt = len(homographies_gt)
fig, axs = plt.subplots(1, n_gt + 1, figsize=(7 * (n_gt + 1), 6))
for ax, loc in zip(axs[:n_gt], homographies_gt):
    sat_path = {11: '11지점.png', 14: '14.png'}.get(loc)
    sat = cv2.cvtColor(cv2.imread(
        f"/Volumes/T7/1. 논문정리/trajectory_project_ver1/위성사진(지점)/{sat_path}"), cv2.COLOR_BGR2RGB)
    ax.imshow(sat)
    mpp = homographies_gt[loc]['m_per_px']
    for k in [kk for kk in train_keys if kk[0] == loc][:80]:
        a = to_metric(smooth_track(tracks['bottom'][k]), loc)
        if a is None:
            continue
        a = a / mpp  # 위성 px로 되돌려 오버레이
        ax.plot(a[:, 0], a[:, 1], color='red', alpha=0.35, linewidth=0.8)
    ax.set_title(f'Location {loc}: trajectories on satellite (measured H)')
    ax.axis('off')

types = list(ANOM_TYPES)
x = np.arange(len(types)); w = 0.35
ax = axs[-1]
ax.bar(x - w/2, [res['B0 image coords (A2)']['type_recall'][t] for t in types], w,
       color='darkorange', label=f"Image coords (F1={res['B0 image coords (A2)']['f1']:.2f})")
ax.bar(x + w/2, [res['B1 satellite metric']['type_recall'][t] for t in types], w,
       color='forestgreen', label=f"Satellite metric (F1={res['B1 satellite metric']['f1']:.2f})")
ax.set_xticks(x); ax.set_xticklabels(types, rotation=15)
ax.set_ylim(0, 1.05); ax.set_ylabel('Recall (calibrated thr)'); ax.legend()
ax.set_title('Detection: image vs measured-homography metric')
plt.tight_layout()
plt.savefig(f"{OUT}/14_measured_homography_eval.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n저장: 14_measured_homography_eval.png")
print("MEASH_OK")
