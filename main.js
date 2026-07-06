/* ============================================================
 * 揉纸 · Paper Crumple
 * MediaPipe HandLandmarker 手势 → 实时控制 video.currentTime
 *
 * 数据流:
 *   摄像头帧 → detectForVideo() → 21 个手部关键点
 *     → handCloseProgress (0 = 手完全张开 … 1 = 握拳/捏合)
 *     → EMA 平滑
 *     → targetTime = progress * duration
 *     → currentTime = lerp(currentTime, targetTime, 0.18)   (每帧)
 *
 * 纸团视频只被 seek,从不调用 paperVideo.play()。
 * ============================================================ */

"use strict";

/* ---------------- 可调参数 ---------------- */
const TUNING = {
  // 伸展度:四指尖到手腕的平均距离 / 手掌尺寸。≥OPEN 记为完全张开,≤CLOSED 记为握拳
  EXT_OPEN: 1.60,
  EXT_CLOSED: 0.95,
  // 聚拢度:四指尖到拇指尖的平均距离 / 手掌尺寸(捏合手势主要由它驱动)
  GATHER_OPEN: 1.10,
  GATHER_CLOSED: 0.38,

  PROGRESS_EMA: 0.35,   // handCloseProgress 一阶平滑系数(每个检测帧)
  TIME_LERP: 0.18,      // currentTime → targetTime 插值系数(题目公式,60fps 基准)
  HAND_LOST_MS: 350,    // 连续无手超过该时长 → Waiting,画面定格
  TREND_EPS: 0.010,     // 区分“揉皱中 / 展开中”的进度变化阈值
  END_EPS: 0.02,        // 距视频末尾的保护间隙(秒),保证末帧稳定显示
  MIN_SEEK_STEP: 0.003, // 小于该差值不再写 currentTime,避免无意义 seek
};

/* MediaPipe 资源:优先本地 vendor(可离线),失败回退 CDN */
const ASSETS = {
  local: {
    module: "./vendor/tasks-vision/vision_bundle.mjs",
    wasmRoot: "./vendor/tasks-vision/wasm",
    model: "./vendor/hand_landmarker.task",
  },
  cdn: {
    module: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs",
    wasmRoot: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
    model: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  },
};

/* 手部关键点索引(MediaPipe Hands 拓扑) */
const WRIST = 0, THUMB_TIP = 4, INDEX_MCP = 5, MIDDLE_MCP = 9, PINKY_MCP = 17;
const FINGER_TIPS = [8, 12, 16, 20]; // 食指 / 中指 / 无名指 / 小指指尖
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/* ---------------- DOM ---------------- */
const paperVideo = document.getElementById("paper-video");
const camVideo = document.getElementById("cam-video");
const camCanvas = document.getElementById("cam-canvas");
const camCtx = camCanvas.getContext("2d");
const camLabel = document.getElementById("cam-label");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const statusSub = document.getElementById("status-sub");
const progressFill = document.getElementById("progress-fill");
const progressKnob = document.getElementById("progress-knob");
const progressNum = document.getElementById("progress-num");
const overlay = document.getElementById("overlay");
const overlayIcon = document.getElementById("overlay-icon");
const overlayTitle = document.getElementById("overlay-title");
const overlayMsg = document.getElementById("overlay-msg");
const overlayTip = document.getElementById("overlay-tip");
const retryBtn = document.getElementById("retry-btn");

/* ---------------- 运行状态 ---------------- */
let handLandmarker = null;
let cameraStream = null;
let videoDuration = 0;        // paperVideo.duration(loadedmetadata 后有效)
let rawProgress = 0;          // 当前检测帧算出的原始 handCloseProgress
let smoothProgress = 0;       // EMA 平滑后的 handCloseProgress
let slowProgress = 0;         // 更迟钝的 EMA,用于判断趋势(揉/展)
let lastHandSeenAt = -1e9;    // 最近一次检测到手的时间戳
let manualUntil = -1e9;       // 键盘调试控制的有效期
let lastDetectTs = 0;         // 保证 detectForVideo 时间戳单调递增
let detectFps = 0;
let lastDetectAt = 0;
let lastTrendLabel = null;    // "folding" | "crumpling",趋势不明显时沿用
let detectFailures = 0;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const clamp01 = (v) => clamp(v, 0, 1);
const lerp = (a, b, t) => a + (b - a) * t;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/* ============================================================
 * handCloseProgress:0 手完全张开 → 1 握拳/捏合
 *
 * 关键点先换算到像素空间(消除画面宽高比畸变),再用手掌尺寸
 * 归一化(手腕→中指根的掌长 与 食指根→小指根的掌宽 的加权平均),
 * 使结果与手离摄像头的远近无关。
 *
 *  - 伸展度 ext:四指尖到手腕的平均归一化距离,张开大 / 握拳小
 *  - 聚拢度 gather:四指尖到拇指尖的平均归一化距离,捏合时趋近 0
 *  两个信号各自反向映射到 [0,1] 后取较强者:握拳靠 ext,捏合靠
 *  gather,五指聚拢两者都高 —— 三种“揉纸”手势都能到 1。
 * ============================================================ */
function computeCloseProgress(lm, w, h) {
  const P = (i) => ({ x: lm[i].x * w, y: lm[i].y * h });
  const wrist = P(WRIST);
  const thumbTip = P(THUMB_TIP);

  const palmLen = dist(wrist, P(MIDDLE_MCP));          // 掌长
  const palmWidth = dist(P(INDEX_MCP), P(PINKY_MCP));  // 掌宽
  const palmSize = (palmLen + 1.3 * palmWidth) / 2;    // 综合尺寸,抗透视缩短
  if (palmSize < 1e-3) return null;

  let extSum = 0, gatherSum = 0;
  for (const tip of FINGER_TIPS) {
    const p = P(tip);
    extSum += dist(p, wrist);
    gatherSum += dist(p, thumbTip);
  }
  const ext = extSum / FINGER_TIPS.length / palmSize;
  const gather = gatherSum / FINGER_TIPS.length / palmSize;

  // 反向映射:距离越小 → 越接近 1(揉拢)
  const curlByExt = clamp01((TUNING.EXT_OPEN - ext) / (TUNING.EXT_OPEN - TUNING.EXT_CLOSED));
  const curlByGather = clamp01((TUNING.GATHER_OPEN - gather) / (TUNING.GATHER_OPEN - TUNING.GATHER_CLOSED));
  const close = Math.max(curlByExt, curlByGather);

  // 两端各留 4% 死区,让“完全张开=0 / 完全握拢=1”更容易达到
  return clamp01((close - 0.04) / 0.92);
}

/* ---------------- 手势检测循环(跟随摄像头帧率) ---------------- */
function scheduleDetect() {
  if (camVideo.requestVideoFrameCallback) {
    camVideo.requestVideoFrameCallback(() => detectFrame());
  } else {
    requestAnimationFrame(detectFrame);
  }
}

function detectFrame() {
  if (!handLandmarker || camVideo.readyState < 2) { scheduleDetect(); return; }

  const now = performance.now();
  lastDetectTs = Math.max(lastDetectTs + 1, now); // detectForVideo 要求时间戳严格递增

  let result = null;
  try {
    result = handLandmarker.detectForVideo(camVideo, lastDetectTs);
    detectFailures = 0;
  } catch (err) {
    if (++detectFailures > 30) {
      showOverlay("error", "手势识别中断", String(err && err.message || err));
      return; // 停止调度
    }
  }

  if (lastDetectAt) detectFps = lerp(detectFps, 1000 / (now - lastDetectAt), 0.15);
  lastDetectAt = now;

  const lm = result && result.landmarks && result.landmarks[0];
  if (lm) {
    const p = computeCloseProgress(lm, camVideo.videoWidth, camVideo.videoHeight);
    if (p !== null) {
      rawProgress = p;
      // handCloseProgress 平滑(要求 6)
      smoothProgress = lerp(smoothProgress, rawProgress, TUNING.PROGRESS_EMA);
      lastHandSeenAt = now;
    }
    drawSkeleton(lm);
    camLabel.textContent = `TRACKING · ${Math.round(detectFps)} FPS`;
  } else {
    camCtx.clearRect(0, 0, camCanvas.width, camCanvas.height);
    camLabel.textContent = "NO HAND";
  }

  scheduleDetect();
}

/* 摄像头小窗里的手部骨架,颜色随揉拢程度从绿变橙 */
function drawSkeleton(lm) {
  const w = camCanvas.width, h = camCanvas.height;
  camCtx.clearRect(0, 0, w, h);
  const hue = 135 - 110 * smoothProgress;

  camCtx.lineWidth = Math.max(1.5, w / 260);
  camCtx.strokeStyle = `hsla(${hue}, 75%, 62%, 0.9)`;
  camCtx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    camCtx.moveTo(lm[a].x * w, lm[a].y * h);
    camCtx.lineTo(lm[b].x * w, lm[b].y * h);
  }
  camCtx.stroke();

  for (let i = 0; i < lm.length; i++) {
    const isTip = i === THUMB_TIP || FINGER_TIPS.includes(i);
    camCtx.fillStyle = isTip ? `hsla(${hue}, 85%, 70%, 1)` : "rgba(255,255,255,0.85)";
    camCtx.beginPath();
    camCtx.arc(lm[i].x * w, lm[i].y * h, isTip ? w / 90 : w / 150, 0, Math.PI * 2);
    camCtx.fill();
  }
}

/* ---------------- 渲染循环:currentTime 平滑逼近 targetTime ---------------- */
let lastFrameAt = performance.now();

function renderLoop(now) {
  requestAnimationFrame(renderLoop);
  const dt = Math.min(100, now - lastFrameAt);
  lastFrameAt = now;

  const handActive = now - lastHandSeenAt < TUNING.HAND_LOST_MS;
  const manualActive = now < manualUntil;
  const controlActive = handActive || manualActive;

  if (videoDuration > 0 && controlActive) {
    // 要求 5 的公式:targetTime = progress * duration;currentTime 向其 lerp。
    // 系数按 dt 做帧率无关校正,60fps 时恰为 0.18,高刷屏收敛速度一致。
    //
    // 注意:这里故意不判断 video.seeking 再决定要不要写 currentTime。
    // 部分移动端浏览器(如 iOS 上的 Chrome/Safari,底层都是 WebKit)完成一次 seek
    // 的耗时明显长于桌面端;如果等 seeking 变回 false 才允许下一次赋值,一旦某次
    // seek 异常缓慢或卡住,seeking 会一直停留在 true,后续所有手势都无法再驱动画
    // 面,直接卡死在黑屏/当前帧。每帧直接重新赋值 currentTime 是浏览器原生支持的
    // 用法(新的赋值会自动取代尚未完成的旧 seek),各类拖动进度条都是这样实现的。
    const maxTime = videoDuration - TUNING.END_EPS;
    const targetTime = clamp(smoothProgress * videoDuration, 0, maxTime);
    const k = 1 - Math.pow(1 - TUNING.TIME_LERP, dt / (1000 / 60));
    const next = clamp(lerp(paperVideo.currentTime, targetTime, k), 0, maxTime);
    if (Math.abs(next - paperVideo.currentTime) > TUNING.MIN_SEEK_STEP) {
      paperVideo.currentTime = next; // 从不 play(),只 seek
    }
  }
  // 无手且无键盘控制:不写 currentTime,画面定格在当前帧(要求 8)

  updateHud(controlActive, handActive);
}

/* ---------------- 状态提示(要求 9) ---------------- */
const STATE_TEXT = {
  waiting:   { en: "Waiting for Gesture",      zh: "等待手势",        cls: "waiting" },
  folding:   { en: "Hand Open / Folding Back", zh: "手张开 · 展开中", cls: "folding" },
  crumpling: { en: "Hand Closed / Crumpling",  zh: "手握拢 · 揉皱中", cls: "crumpling" },
};

function updateHud(controlActive, handActive) {
  // 趋势判断:平滑进度与慢速进度的差,决定“揉皱中 / 展开中”
  slowProgress = lerp(slowProgress, smoothProgress, 0.06);
  const trend = smoothProgress - slowProgress;

  let state;
  if (!controlActive) {
    state = "waiting";
  } else {
    if (trend > TUNING.TREND_EPS) lastTrendLabel = "crumpling";
    else if (trend < -TUNING.TREND_EPS) lastTrendLabel = "folding";
    if (!lastTrendLabel) lastTrendLabel = smoothProgress >= 0.5 ? "crumpling" : "folding";
    state = lastTrendLabel;
  }

  const s = STATE_TEXT[state];
  if (statusText.textContent !== s.en) {
    statusText.textContent = s.en;
    statusSub.textContent = s.zh;
    statusPill.className = `status-pill ${s.cls}`;
  }

  // 进度条反映画面实际所处的时间轴位置
  const shown = videoDuration > 0
    ? clamp01(paperVideo.currentTime / (videoDuration - TUNING.END_EPS))
    : 0;
  const pct = `${(shown * 100).toFixed(0)}%`;
  progressFill.style.width = pct;
  progressKnob.style.left = pct;
  progressNum.textContent = pct;

  if (!handActive && !cameraStream) camLabel.textContent = "CAMERA OFF";
}

/* ---------------- 提示层 ---------------- */
function showOverlay(kind, title, msg, { retry = false, tip = false, icon = "✋" } = {}) {
  overlay.classList.remove("hidden");
  overlayIcon.textContent = icon;
  overlayTitle.textContent = title;
  overlayMsg.textContent = msg;
  retryBtn.hidden = !retry;
  overlayTip.hidden = !tip;
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

/* ---------------- 初始化:纸团视频 ---------------- */
function initPaperVideo() {
  return new Promise((resolve, reject) => {
    const ready = () => {
      videoDuration = paperVideo.duration || 0;
      // 轻微 seek 一次,强制解码出第一帧(不播放)
      try { paperVideo.currentTime = 0.001; } catch (_) { /* noop */ }
      resolve();
    };
    if (paperVideo.readyState >= 1) return ready();
    paperVideo.addEventListener("loadedmetadata", ready, { once: true });
    paperVideo.addEventListener("error", () => reject(new Error("视频加载失败,请确认 揉纸_scrub.mp4 / 揉纸.mp4 在同一目录")), { once: true });
  });
}

/* ---------------- 初始化:MediaPipe ---------------- */
async function initVision() {
  let lastErr = null;
  for (const src of [ASSETS.local, ASSETS.cdn]) {
    try {
      const vision = await import(src.module);
      const fileset = await vision.FilesetResolver.forVisionTasks(src.wasmRoot);
      for (const delegate of ["GPU", "CPU"]) {
        try {
          handLandmarker = await vision.HandLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: src.model, delegate },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          return;
        } catch (err) { lastErr = err; }
      }
    } catch (err) { lastErr = err; }
  }
  throw lastErr || new Error("HandLandmarker 初始化失败");
}

/* ---------------- 初始化:摄像头 ---------------- */
async function initCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const err = new Error("insecure");
    err.name = "InsecureContext";
    throw err;
  }
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  camVideo.srcObject = cameraStream;
  await new Promise((resolve) => {
    camVideo.addEventListener("loadedmetadata", resolve, { once: true });
  });
  camCanvas.width = camVideo.videoWidth || 640;
  camCanvas.height = camVideo.videoHeight || 480;
  // 注意:这里播放的是“摄像头预览”,与被手势 seek 的纸团视频无关
  await camVideo.play();
}

function cameraErrorInfo(err) {
  const name = err && err.name || "";
  if (name === "InsecureContext") {
    return {
      title: "无法调用摄像头",
      msg: "当前页面不是安全上下文。\n请通过本地服务器访问,例如在项目目录运行:\npython3 -m http.server 4173\n然后打开 http://localhost:4173",
    };
  }
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return {
      title: "摄像头权限未开启",
      msg: "浏览器拒绝了摄像头访问。\n请点击地址栏的 🎥/🔒 图标,将摄像头设为“允许”;\nmacOS 用户还需在 系统设置 → 隐私与安全性 → 摄像头 中勾选当前浏览器,然后点击下方按钮重试。",
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return { title: "未找到摄像头", msg: "没有检测到可用的摄像头设备。\n接入摄像头后点击重试。" };
  }
  if (name === "NotReadableError") {
    return { title: "摄像头被占用", msg: "摄像头正被其他应用使用。\n关闭占用它的应用(视频会议、相机等)后点击重试。" };
  }
  return { title: "摄像头启动失败", msg: String(err && err.message || err) };
}

/* ---------------- 键盘调试(无摄像头时预览效果) ---------------- */
function setManualProgress(p) {
  rawProgress = clamp01(p);
  smoothProgress = lerp(smoothProgress, rawProgress, 0.6);
  manualUntil = performance.now() + 1200;
}

window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") setManualProgress(rawProgress + 0.045);
  else if (e.key === "ArrowLeft") setManualProgress(rawProgress - 0.045);
});

/* 供控制台/自动化验证使用 */
window.paperDemo = {
  setProgress: (p) => { setManualProgress(p); smoothProgress = clamp01(p); },
  get progress() { return smoothProgress; },
  get video() { return paperVideo; },
};

/* ---------------- 启动 ---------------- */
async function main() {
  requestAnimationFrame(renderLoop);
  showOverlay("loading", "正在加载手势模型…", "Loading MediaPipe HandLandmarker");

  try {
    await initPaperVideo();
  } catch (err) {
    showOverlay("error", "视频加载失败", String(err && err.message || err), { icon: "🎞️" });
    return;
  }

  const [visionRes, camRes] = await Promise.allSettled([initVision(), initCamera()]);

  if (visionRes.status === "rejected") {
    showOverlay("error", "手势模型加载失败",
      `${String(visionRes.reason && visionRes.reason.message || visionRes.reason)}\n请确认 vendor 目录完整,或保持网络连接后刷新重试。`,
      { icon: "⚠️" });
    return;
  }

  if (camRes.status === "rejected") {
    const info = cameraErrorInfo(camRes.reason);
    showOverlay("camera", info.title, info.msg, { retry: true, tip: true, icon: "🎥" });
    return; // 等待用户点击重试
  }

  hideOverlay();
  scheduleDetect();
}

retryBtn.addEventListener("click", async () => {
  retryBtn.disabled = true;
  try {
    await initCamera();
    hideOverlay();
    scheduleDetect();
  } catch (err) {
    const info = cameraErrorInfo(err);
    showOverlay("camera", info.title, info.msg, { retry: true, tip: true, icon: "🎥" });
  } finally {
    retryBtn.disabled = false;
  }
});

window.addEventListener("pagehide", () => {
  if (cameraStream) cameraStream.getTracks().forEach((t) => t.stop());
});

main();
