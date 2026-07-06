# 揉纸 · Paper Crumple —— 手势控制视频时间轴

用一只手隔空控制视频进度:**张开手掌**,折纸头像保持完整;**慢慢握拢 / 捏合**,头像被实时揉成纸团;再张开,纸团反向展开。视频从不播放,`video.currentTime` 完全由手势驱动 —— 就像用手在拖动时间轴。

🔗 **在线体验:** _部署后回填_

> 需要授权摄像头权限,建议使用 Chrome / Edge 桌面浏览器打开。

## 效果说明

- 单手在摄像头前伸出即可,无需按键或点击
- 手掌张开程度实时映射为 `handCloseProgress`(0 → 1),再映射为视频进度
- 松开手 / 移出画面时,画面停在当前帧,不会重置或继续播放
- 页面显示三种状态提示:`Hand Open / Folding Back`、`Hand Closed / Crumpling`、`Waiting for Gesture`
- 摄像头权限被拒绝时会给出针对性的开启指引(浏览器设置 / 系统设置)

## 本地运行

摄像头调用需要安全上下文(`localhost` 或 `https`),请用本地服务器打开,不要直接双击 `index.html`:

```bash
python3 -m http.server 4173
# 浏览器打开 http://localhost:4173 ,允许摄像头权限
```

没有摄像头时,可用键盘 `←` `→` 方向键预览揉纸效果。

## 原理

1. **手部识别** — MediaPipe Tasks Vision `HandLandmarker`(VIDEO 模式,单手 21 关键点),模型与 WASM 运行时已下载到 `vendor/`,离线可用;缺失时自动回退官方 CDN。
2. **handCloseProgress(0 张开 → 1 揉拢)** — 关键点换算到像素空间后:
   - 伸展度 = 四指尖到手腕的平均距离 ÷ 手掌尺寸(掌长 + 掌宽加权),握拳时变小;
   - 聚拢度 = 四指尖到拇指尖的平均距离 ÷ 手掌尺寸,捏合时趋近 0;
   - 两个信号各自反向映射到 `[0,1]` 后取较强者,再做 EMA 平滑 —— 握拳、捏合、五指聚拢三种手势都能揉到底。
3. **时间轴控制** — 每帧执行:
   ```js
   const targetTime = handCloseProgress * video.duration;
   video.currentTime = lerp(video.currentTime, targetTime, 0.18); // 已做帧率无关校正,并夹在 [0, duration]
   ```
   视频始终 `muted playsinline preload="auto"`,无 `autoplay` / `loop`,全程不调用 `play()`,只做 `seek`。
4. **丢手定格** — 检测不到手超过 350ms 时停止写 `currentTime`,画面停在当前帧,不重置、不播放。
5. **状态提示** — 三态文案见上;摄像头权限被拒时展示对应的开启指引。

## 技术栈

- 原生 `HTML / CSS / JavaScript`,无构建步骤、无框架依赖
- [`@mediapipe/tasks-vision`](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) `HandLandmarker`,模型与 WASM 已 vendor 化到本地
- `Canvas 2D` 绘制摄像头小窗中的实时手部骨架

## 项目结构

| 路径 | 说明 |
| --- | --- |
| `index.html` / `style.css` / `main.js` | 页面结构、样式与手势 → 时间轴的核心交互逻辑 |
| `揉纸.mp4` | 原始视频(仅 1 个关键帧,不适合逐帧 seek,保留作后备源) |
| `揉纸_scrub.mp4` | 由原片重编码的全关键帧版本(121/121 I-frame),页面实际使用,保证拖动顺滑 |
| `vendor/` | MediaPipe `HandLandmarker` 模型 + WASM 运行时(本地化,离线可用) |
| `原来.png` / `揉皱.png` | 首帧 / 尾帧参考图(`原来.png` 同时用作视频 `poster`) |

## 调参

`main.js` 顶部的 `TUNING` 对象集中了所有可调数值:手势阈值(`EXT_*` / `GATHER_*`)、平滑系数(`PROGRESS_EMA` / `TIME_LERP`)、丢手定格时长(`HAND_LOST_MS`)等。

## 部署

纯静态站点,推送到任意静态托管(Vercel / Netlify / GitHub Pages 等)均可直接运行;线上环境天然是 `https`,满足摄像头调用所需的安全上下文,无需额外配置。

## 致谢

手势识别基于 Google [MediaPipe](https://github.com/google-ai-edge/mediapipe)(Apache License 2.0),相关模型与 WASM 运行时版权归 Google 所有。
