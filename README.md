# 基于智能体的声学仿真模拟器

一个使用 Python 实现的 2D 声场仿真程序，包含：
- 波动方程有限差分求解
- 点声源与线声源激励
- 障碍物反射与吸收
- 吸收/反射/周期边界条件
- 频谱分析与 SPL 计算
- 房间声学与波束形成示例

## 1. 环境准备（Conda）

推荐使用 Conda，已提供环境文件：environment.yml。

### 1.1 首次创建环境

在项目目录执行：

```powershell
conda env create -f environment.yml
```

### 1.2 激活环境

```powershell
conda activate acousticsim
```

### 1.3 验证依赖

```powershell
python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
```

## 2. 运行程序

### 2.1 运行默认主程序

```powershell
python shengxuefangzhen.py
```

主程序默认会执行基础示例 run_basic_example()，并进行可视化。

### 2.2 运行其他示例

在 shengxuefangzhen.py 文件底部取消注释即可：
- run_multiple_sources_example()
- run_room_acoustic_example()
- run_interactive_simulation()

## 3. 常见问题

### 3.1 报错：无法解析导入 matplotlib

原因：当前解释器不是 conda 环境 acousticsim，或依赖未安装。

处理：
1. 在 VS Code 中切换解释器为：
   C:\Users\ASUS\miniconda3\envs\acousticsim\python.exe
2. 在终端执行：

```powershell
conda activate acousticsim
python -c "import matplotlib"
```

### 3.2 运行时出现 CFL 警告

这是数值稳定性保护逻辑，程序会自动减小时间步长以满足稳定性条件，不属于错误。

### 3.3 无图形窗口或窗口不弹出

可能原因：
- 远程会话或无 GUI 环境
- Matplotlib 后端不可用

可先做无图形验证：

```powershell
python -c "from shengxuefangzhen import create_default_simulation; s=create_default_simulation(); s.simulate(); print(s.time_steps, s.current_time)"
```

## 4. 项目文件

- shengxuefangzhen.py：仿真核心与示例入口
- environment.yml：Conda 环境复现文件
- README.md：当前说明文档

## 5. 后续建议

- 增加单元测试（边界条件、频谱、SPL）
- 将仿真参数拆分为配置文件（YAML/JSON）
- 增加结果导出（CSV/NPY/GIF）
