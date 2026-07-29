# 방법론 진화 요약 그림 — 확대 평가셋 F1 기준의 실험 ③~⑬ 궤적
# 수치 출처: docs/ANALYSIS.md 각 실험 절 (여기서는 기록된 결과만 사용, 재계산 없음)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "/Volumes/T7/1. 논문정리/github/analysis_results"

# 채택 경로 (누적 구성, 확대 평가셋 F1)
steps = [
    ("#3\nLSTM-AE\nbaseline", 0.25),
    ("#4\nlane-relative\nrules + dir. field", 0.67),
    ("#6\n+ smoothing", 0.69),
    ("#7\n+ bottom-center\n(A2)", 0.70),
    ("#10\n+ cross_flow\n/ osc", 0.81),
    ("#11\n+ AE hybrid\nmean", 0.85),
]
labels = [s[0] for s in steps]
f1s = [s[1] for s in steps]

fig, ax = plt.subplots(figsize=(12.5, 5.2))
x = np.arange(len(steps))
ax.plot(x, f1s, 'o-', color='#1E3A5F', lw=2.2, markersize=8, zorder=5)
for i, (xi, yi) in enumerate(zip(x, f1s)):
    off = (-16, 8) if i == len(x) - 1 else (0, 10)
    ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=off,
                ha='center', fontsize=10, color='#1E3A5F', fontweight='bold')

# 운영점 범위 (⑬-b): q95(채택 기본) -> q85, 같은 점수의 오경보 예산 트레이드오프
ax.fill_between([x[-1] + 0.1, x[-1] + 0.75], 0.85, 0.92, color='seagreen', alpha=0.18, zorder=1)
ax.annotate("operating-point range (#14)\nq95 -> q85: F1 0.85 -> 0.92\n(recall 0.76 -> 0.93, FPR 7 -> 18%)",
            xy=(x[-1] + 0.42, 0.815), ha='center', va='top', fontsize=8.5, color='seagreen')

# 기각·조건부 실험 주석 (경로 밖, 평탄 구간 아래 여백)
side_notes = [
    ("#5 auto-perspective: rejected", 0.57, 'crimson'),
    ("#8-9 measured homography / curved centerline:\nrejected for detection (kept for physical units)", 0.51, 'crimson'),
    ("#12 POT dynamic threshold: F1 0.84, FP 11->7\n- conditional (ops alternative)", 0.44, '#B8860B'),
    ("#13 prediction score: rejected (r=0.86 with AE\n- sees the same anomalies; blind to wrong-way)", 0.37, 'crimson'),
    ("#14 fusion topology: OR's \"F1 0.88\" = false-alarm\nbudget illusion; score-mean dominates at matched FP", 0.30, '#2F7D5C'),
    ("#15 AE training budget 10->60 epochs: \"F1 0.89\" =\nsame illusion; operating curves overlap - rejected", 0.23, 'crimson'),
]
for txt, yy, c in side_notes:
    ax.annotate(txt, xy=(0.28, yy), xycoords=('axes fraction', 'data'),
                fontsize=8, color=c, va='center')

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_xlim(-0.5, len(steps) - 0.1 + 0.8)
ax.set_ylim(0.13, 1.0)
ax.set_ylabel('F1 (enlarged synthetic benchmark, 24/type/site)')
ax.set_title('Method evolution: thesis LSTM-AE baseline -> lane-relative rule scoring + hybrid\n'
             '(adopted path; rejected and conditional branches noted below)')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUT}/00_method_evolution.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: 00_method_evolution.png")
print("EVOLUTION_OK")
