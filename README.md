# 实验作业2：手部运动光流估计（Farneback）

本项目给出一个可运行、可调参的 Python 脚本，使用 OpenCV 的 Farneback 方法对手部运动视频估计光流并生成可视化视频。任务背景、参数影响、使用示例和常见问题都在本文档中说明。

## 项目结构
```
.
├─ input/                      # 放输入视频（例如 hand.avi）
├─ output/                     # 输出光流可视化视频
├─ run_farneback_flow.py       # 主脚本：计算并可视化光流
├─ requirements.txt            # Python 依赖
└─ README.md
```

## 实验背景
- **任务**：从测试视频估计手部运动的 optical flow，代表性方法可选 HS / Farneback / LK，本实现选用 Farneback。
- **调参与分析**：需要通过调整参数观察效果变化并分析原因，核心参数见下文。
- **数据来源**：输入视频来自课程网盘，下载后放入 `input/hand.avi`，或在运行时用 `--video` 指定路径。

## 安装与环境准备
可使用 venv 或 Conda，以下示例以 Windows PowerShell 为例：
```powershell
# 克隆/进入仓库后，在 PowerShell 执行
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 运行示例
确保视频已放入 `input/hand.avi`，或用 `--video` 指向其他路径。
```powershell
# 默认运行（同时输出 HSV 和箭头可视化）
python run_farneback_flow.py --video input/hand.avi --out_dir output

# 仅生成 HSV 版本，可指定输出目录
python run_farneback_flow.py --video input/hand.avi --viz hsv --out_dir output

# 调整光流参数，让结果更平滑（更大窗口、更高迭代）
python run_farneback_flow.py --video input/hand.avi --winsize 25 --iterations 5 --poly_n 7 --poly_sigma 1.5

# 让结果更敏感/细节更多（更小窗口、更多金字塔层）
python run_farneback_flow.py --video input/hand.avi --winsize 9 --levels 4 --iterations 4

# 放慢输出播放速度（半速），便于观察
python run_farneback_flow.py --video input/hand.avi --playback_rate 0.5

# 加速：缩放最长边至 640，再计算
python run_farneback_flow.py --video input/hand.avi --resize 640

# 聚焦手部：仅在 ROI 内计算/可视化（x1,y1,x2,y2）
python run_farneback_flow.py --video input/hand.avi --roi 200,100,500,400 --viz both

# 调整箭头密度
python run_farneback_flow.py --video input/hand.avi --quiver_step 24
```
输出文件会写入 `--out_dir`：
- `flow_hsv.mp4`：HSV 伪彩色（方向→Hue，幅值→Value/Saturation）。
- `flow_quiver.mp4`：原始帧上叠加箭头，显示流向和大致幅值。

## 参数说明与影响
- `--pyr_scale`（默认 0.5）：金字塔尺度因子。越小，金字塔层间差异越大，能捕捉大位移但易噪声。
- `--levels`（默认 3）：金字塔层数。越大越能处理大位移，计算更慢。
- `--winsize`（默认 15）：窗口大小。越大越平滑、对噪声更鲁棒，但会抹掉细节；越小对细节敏感但易噪声。
- `--iterations`（默认 3）：每层迭代次数。越大结果更稳定但更慢。
- `--poly_n`（默认 5）：多项式邻域大小。大值更平滑、抗噪声，小值保留细节。
- `--poly_sigma`（默认 1.2）：多项式平滑的高斯核标准差。大值平滑、噪声少，过大会损失纹理；小值保留细节但可能噪声。
- `--resize`：按最长边缩放后再计算，减小尺寸可显著提速，但过小会导致精度下降。
- `--roi`：仅在指定区域计算/可视化，便于聚焦手部且减少计算量。
- `--viz`：选择输出 HSV、箭头或二者，控制输出视频类型。
- `--quiver_step`：箭头网格步长。值越大箭头越稀疏、文件更小；越小箭头更密集但易重叠。
- `--playback_rate`：调节输出视频的播放倍率（相对输入 FPS）。`0.5` 可放慢为半速观察细节，`1.0` 为原速，`2.0` 加速。

## 输出与统计
- 每帧打印光流幅值的 `mean_mag` 和 `max_mag`（可用 `--print_every` 调整打印频率）。
- 全视频结束后汇总：处理帧数、全局平均幅值、最大幅值。
- 视频编码优先使用 `mp4v`，若失败自动回退 `MJPG`，尽量保证在 Windows 也可写入。

## 常见问题（FAQ）
- **找不到视频**：提示 `Input video not found` 时，确认文件存在且路径正确；默认期待 `input/hand.avi`。
- **无法写 mp4**：若 `mp4v` 不可用脚本会回退 `MJPG`。仍失败时，可尝试安装完整的 OpenCV 或改用 `--viz quiver` 生成 AVI（MJPG）。
- **OpenCV 安装失败**：优先升级 pip；如网络受限，可下载离线轮子；或在 Conda 环境中安装 `conda install -c conda-forge opencv`。
- **输出过慢**：使用 `--resize` 减小分辨率、增大 `--quiver_step`、减小 `--levels`/`--winsize`。
- **需要慢放观察**：用 `--playback_rate 0.5`（或更小）降低输出 FPS；若想更快浏览，可设置大于 1 的倍率。
- **结果太平滑/细节丢失**：减小 `--winsize`、`--poly_n`、`--poly_sigma`，或增加 `--levels` 捕捉细节。
- **噪声太多/箭头抖动**：增大 `--winsize`、`--poly_n`、`--poly_sigma`，减少 `--levels`，或使用 ROI 聚焦主体。

## 提示
- 脚本不使用 GUI，不会弹出窗口；所有结果写入输出目录。
- ROI 只影响计算和可视化区域，区域外的 HSV 输出为黑色，箭头图会用矩形标记 ROI。
