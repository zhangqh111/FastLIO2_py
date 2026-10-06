# FastLIO2_py

本项目为使用 Python 实现的 FastLIO2。

## 项目简介

本仓库提供了一个纯 Python 版本的 FastLIO2 实现方案。为了方便数据处理和算法运行，我们将 ROS bag 包中的 IMU 和 LiDAR 数据预先提取并转换为二进制（bin）文件，从而加速后续的算法运行流程。

## 📂 目录结构

```text
FastLIO2_py/
├── IMU2bin.py          # IMU数据转换脚本
├── LiDAR2bin.py        # LiDAR数据转换脚本
└── scripts/
    └── main.py         # 主运行脚本
```

## 🛠️ 运行流程

请按照以下顺序执行脚本：

### 1️⃣ 转换 IMU 数据
使用 `IMU2bin.py` 读取 bag 包中的 IMU 话题，并将其转换为二进制的 `.bin` 文件。

```bash
python IMU2bin.py
```

### 2️⃣ 转换 LiDAR 数据
使用 `LiDAR2bin.py` 读取 bag 包中的 LiDAR 话题，并将其转换为二进制的 `.bin` 文件。

```bash
python LiDAR2bin.py
```

### 3️⃣ 运行主流程
数据转换完成后，运行 `scripts/main.py` 来跑整个 FastLIO2 流程。

```bash
python scripts/main.py
```

## ⚙️ 环境依赖

*   Python 3.x
*   依赖库：`numpy`、`rosbag` 等（请根据实际环境自行安装）

💡 **提示**：在运行转换脚本前，请确保在代码中正确配置了你的 bag 包路径以及需要提取的话题名称。


### 💡 设计说明：
1. **标题与图标**：使用了 `🚀`、`📖`、`📂` 等 Emoji 增加视觉层次感，这是 GitHub 开源项目常用的风格。
2. **代码块**：使用了 `bash` 代码块包裹运行命令，用户可以一键复制命令；使用 `text` 展示了清晰的目录树。
3. **分步介绍**：用序号（1️⃣、2️⃣、3️⃣）和粗体明确标识了运行顺序，非常直观。
4. **额外提示**：在末尾加了一个温馨提示，因为对于处理 bag 包的 Python 脚本，最常出问题的地方就是路径和话题名配置。
