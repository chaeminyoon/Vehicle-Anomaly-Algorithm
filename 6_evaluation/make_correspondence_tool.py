# 개선 실험 ⑧용: 실측 호모그래피 대응점 지정 도구(HTML) 생성기
# - CCTV 배경 프레임(중앙값, 차량 제거)과 위성사진을 나란히 놓고 클릭으로 대응점 쌍 지정
# - 위성사진 위 두 점 + 실거리(m) 입력으로 스케일 측정선 지정 (여러 개면 평균)
# - 결과는 JSON 다운로드: {pairs: [{cctv:[x,y], sat:[x,y]}], scale_lines: [{p1,p2,meters}]}
# 사용: python make_correspondence_tool.py -> analysis_results/homography_gt/annotate_{loc}.html
import base64
import json
import os

GT_DIR = "/Volumes/T7/1. 논문정리/github/analysis_results/homography_gt"
SAT_DIR = "/Volumes/T7/1. 논문정리/trajectory_project_ver1/위성사진(지점)"
SAT_FILES = {11: "11지점.png", 14: "14.png"}

TEMPLATE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>대응점 지정 — 지점 __LOC__</title>
<style>
body { font-family: sans-serif; margin: 10px; background: #1e1e1e; color: #ddd; }
#panes { display: flex; gap: 8px; }
.pane { flex: 1; min-width: 0; }
.pane h3 { margin: 4px 0; font-size: 14px; }
.wrap { position: relative; border: 1px solid #555; overflow: hidden; height: 72vh; background: #000; }
canvas { position: absolute; left: 0; top: 0; transform-origin: 0 0; cursor: crosshair; }
#bar { margin: 8px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
button { padding: 6px 12px; background: #333; color: #eee; border: 1px solid #666; border-radius: 4px; cursor: pointer; }
button:hover { background: #444; }
button.active { background: #0a5; border-color: #0c6; }
#list { font-size: 12px; max-height: 14vh; overflow-y: auto; margin-top: 6px; }
#list span { display: inline-block; margin: 2px 6px 2px 0; padding: 2px 6px; background: #2a2a2a; border-radius: 3px; }
#list span b { color: #6cf; cursor: pointer; }
#status { color: #fc6; }
</style></head><body>
<div id="bar">
  <b>지점 __LOC__</b>
  <button id="mode-pair" class="active">대응점 모드</button>
  <button id="mode-scale">스케일 모드 (위성에 2점+실거리)</button>
  <button id="undo">되돌리기 (u)</button>
  <button id="save">JSON 다운로드</button>
  <span id="status">CCTV에서 1번 점을 클릭하세요</span>
  <span style="color:#888">휠=확대 / 드래그=이동 / 클릭=점 지정. 4쌍 이상, 화면 전체에 퍼지게 8쌍 권장.</span>
</div>
<div id="panes">
  <div class="pane"><h3>CCTV 배경 (원본 해상도 __CW__x__CH__)</h3>
    <div class="wrap" id="wrap0"><canvas id="c0"></canvas></div></div>
  <div class="pane"><h3>위성사진</h3>
    <div class="wrap" id="wrap1"><canvas id="c1"></canvas></div></div>
</div>
<div id="list"></div>
<script>
const LOC = __LOC__;
const imgs = [new Image(), new Image()];
imgs[0].src = "data:image/png;base64,__CCTV_B64__";
imgs[1].src = "data:image/png;base64,__SAT_B64__";
let pairs = [];          // {cctv:[x,y], sat:[x,y]}
let scaleLines = [];     // {p1:[x,y], p2:[x,y], meters}
let pending = null;      // 대응점: CCTV 점 대기 중이면 [x,y]
let pendingScale = null; // 스케일: 첫 점
let mode = 'pair';
const KEY = 'corr_loc___LOC__';
try { const s = JSON.parse(localStorage.getItem(KEY)); if (s) { pairs = s.pairs || []; scaleLines = s.scale_lines || []; } } catch (e) {}

const views = [ {z: 1, ox: 0, oy: 0}, {z: 1, ox: 0, oy: 0} ];
const canvases = [document.getElementById('c0'), document.getElementById('c1')];
const wraps = [document.getElementById('wrap0'), document.getElementById('wrap1')];

function draw(i) {
  const cv = canvases[i], im = imgs[i];
  if (!im.complete || !im.naturalWidth) return;
  cv.width = im.naturalWidth; cv.height = im.naturalHeight;
  const ctx = cv.getContext('2d');
  ctx.drawImage(im, 0, 0);
  const v = views[i];
  cv.style.transform = `translate(${v.ox}px,${v.oy}px) scale(${v.z})`;
  const r = 6 / v.z;
  ctx.lineWidth = 2 / v.z; ctx.font = `${16 / v.z}px sans-serif`;
  pairs.forEach((p, k) => {
    const pt = i === 0 ? p.cctv : p.sat;
    ctx.strokeStyle = '#0f0'; ctx.fillStyle = '#0f0';
    ctx.beginPath(); ctx.arc(pt[0], pt[1], r, 0, 7); ctx.stroke();
    ctx.fillText(k + 1, pt[0] + r, pt[1] - r);
  });
  if (i === 0 && pending) {
    ctx.strokeStyle = '#ff0';
    ctx.beginPath(); ctx.arc(pending[0], pending[1], r, 0, 7); ctx.stroke();
  }
  if (i === 1) {
    ctx.strokeStyle = '#f6f'; ctx.fillStyle = '#f6f';
    scaleLines.forEach(sl => {
      ctx.beginPath(); ctx.moveTo(sl.p1[0], sl.p1[1]); ctx.lineTo(sl.p2[0], sl.p2[1]); ctx.stroke();
      ctx.fillText(sl.meters + 'm', (sl.p1[0] + sl.p2[0]) / 2, (sl.p1[1] + sl.p2[1]) / 2);
    });
    if (pendingScale) { ctx.beginPath(); ctx.arc(pendingScale[0], pendingScale[1], r, 0, 7); ctx.stroke(); }
  }
}
function drawAll() { draw(0); draw(1); renderList(); persist(); }
function persist() { localStorage.setItem(KEY, JSON.stringify({pairs, scale_lines: scaleLines})); }
function renderList() {
  const el = document.getElementById('list');
  el.innerHTML = pairs.map((p, k) =>
    `<span>#${k + 1} cctv(${p.cctv[0].toFixed(0)},${p.cctv[1].toFixed(0)}) sat(${p.sat[0].toFixed(0)},${p.sat[1].toFixed(0)}) <b data-k="${k}">x</b></span>`).join('') +
    scaleLines.map((s, k) => `<span style="color:#f9f">척도 ${s.meters}m <b data-s="${k}">x</b></span>`).join('');
  el.querySelectorAll('b[data-k]').forEach(b => b.onclick = () => { pairs.splice(+b.dataset.k, 1); drawAll(); });
  el.querySelectorAll('b[data-s]').forEach(b => b.onclick = () => { scaleLines.splice(+b.dataset.s, 1); drawAll(); });
  const st = document.getElementById('status');
  if (mode === 'pair') st.textContent = pending ? '위성사진에서 같은 지물을 클릭하세요' : `CCTV에서 ${pairs.length + 1}번 점을 클릭하세요 (현재 ${pairs.length}쌍)`;
  else st.textContent = pendingScale ? '위성에서 두 번째 점 클릭 후 실거리(m) 입력' : '위성에서 스케일 선의 첫 점을 클릭하세요';
}
imgs.forEach((im, i) => im.onload = () => draw(i));

canvases.forEach((cv, i) => {
  const wrap = wraps[i], v = views[i];
  let dragging = false, moved = false, sx, sy;
  wrap.addEventListener('mousedown', e => { dragging = true; moved = false; sx = e.clientX; sy = e.clientY; });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    v.ox += dx; v.oy += dy; sx = e.clientX; sy = e.clientY;
    cv.style.transform = `translate(${v.ox}px,${v.oy}px) scale(${v.z})`;
  });
  window.addEventListener('mouseup', () => dragging = false);
  wrap.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const f = e.deltaY < 0 ? 1.25 : 0.8;
    v.ox = mx - f * (mx - v.ox); v.oy = my - f * (my - v.oy); v.z *= f;
    cv.style.transform = `translate(${v.ox}px,${v.oy}px) scale(${v.z})`;
  }, {passive: false});
  wrap.addEventListener('click', e => {
    if (moved) return;
    const rect = wrap.getBoundingClientRect();
    const x = (e.clientX - rect.left - v.ox) / v.z, y = (e.clientY - rect.top - v.oy) / v.z;
    if (x < 0 || y < 0 || x > cv.width || y > cv.height) return;
    if (mode === 'pair') {
      if (i === 0 && !pending) pending = [x, y];
      else if (i === 1 && pending) { pairs.push({cctv: pending, sat: [x, y]}); pending = null; }
    } else if (i === 1) {
      if (!pendingScale) pendingScale = [x, y];
      else {
        const m = parseFloat(prompt('이 선의 실거리(미터)를 입력:', '3.25'));
        if (m > 0) scaleLines.push({p1: pendingScale, p2: [x, y], meters: m});
        pendingScale = null;
      }
    }
    drawAll();
  });
});
document.getElementById('mode-pair').onclick = e => { mode = 'pair'; pendingScale = null; e.target.classList.add('active'); document.getElementById('mode-scale').classList.remove('active'); drawAll(); };
document.getElementById('mode-scale').onclick = e => { mode = 'scale'; pending = null; e.target.classList.add('active'); document.getElementById('mode-pair').classList.remove('active'); drawAll(); };
document.getElementById('undo').onclick = () => { if (mode === 'pair') { if (pending) pending = null; else pairs.pop(); } else { if (pendingScale) pendingScale = null; else scaleLines.pop(); } drawAll(); };
window.addEventListener('keydown', e => { if (e.key === 'u') document.getElementById('undo').onclick(); });
document.getElementById('save').onclick = () => {
  const data = {location: LOC, cctv_size: [imgs[0].naturalWidth, imgs[0].naturalHeight],
                sat_size: [imgs[1].naturalWidth, imgs[1].naturalHeight], pairs, scale_lines: scaleLines};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], {type: 'application/json'}));
  a.download = `points___LOC__.json`; a.click();
};
setInterval(renderList, 800);
</script></body></html>
"""

def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

if __name__ == '__main__':
    import cv2
    for loc, sat_name in SAT_FILES.items():
        cctv_png = f"{GT_DIR}/cctv_bg_{loc}.png"
        im = cv2.imread(cctv_png)
        h, w = im.shape[:2]
        html = (TEMPLATE
                .replace('__CCTV_B64__', b64(cctv_png))
                .replace('__SAT_B64__', b64(os.path.join(SAT_DIR, sat_name)))
                .replace('__LOC__', str(loc))
                .replace('__CW__', str(w)).replace('__CH__', str(h)))
        out = f"{GT_DIR}/annotate_{loc}.html"
        with open(out, 'w') as f:
            f.write(html)
        print(f"생성: {out} ({len(html) // 1024} KB)")
    print("TOOL_OK")
