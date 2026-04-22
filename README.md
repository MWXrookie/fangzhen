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


# 声学仿真参数收集表

以下内容按模块整理，包含参数名、说明、类型、是否必填、默认值和取值示例。

## 第一部分：仿真类型选择（必填，三选一）

| 参数名 | 中文说明 | 类型 | 是否必填 | 可选值 / 示例 |
| --- | --- | --- | --- | --- |
| simulation_type | 仿真模式 | 字符串 | ✅ 必填 | general（通用声场）/ room（房间声学）/ beamforming（波束形成） |

## 第二部分：通用仿真配置（AcousticConfig）

适用于 general 和 room 模式。

| 参数名 | 中文说明 | 类型 | 是否必填 | 默认值 | 取值范围 / 示例 |
| --- | --- | --- | --- | --- | --- |
| spatial_domain_x | 仿真空间宽度（x 方向） | 浮点数 | ⬜ 选填 | 10.0 | 正数，单位：米，如 5.0 |
| spatial_domain_y | 仿真空间高度（y 方向） | 浮点数 | ⬜ 选填 | 10.0 | 正数，单位：米，如 8.0 |
| grid_resolution_x | x 方向网格数量 | 整数 | ⬜ 选填 | 200 | 建议 50~500，如 100 |
| grid_resolution_y | y 方向网格数量 | 整数 | ⬜ 选填 | 200 | 建议 50~500，如 100 |
| time_step | 时间步长 | 浮点数 | ⬜ 选填 | 0.001 | 单位：秒，如 0.0005（越小越精确） |
| total_time | 总仿真时长 | 浮点数 | ⬜ 选填 | 2.0 | 单位：秒，如 0.5 |
| boundary_type | 边界条件类型 | 字符串 | ⬜ 选填 | absorbing | absorbing（吸收）/ reflecting（反射）/ periodic（周期） |
| damping | 阻尼系数 | 浮点数 | ⬜ 选填 | 0.01 | 范围 0.0~1.0，如 0.05 |
| medium_material | 传播介质材料 | 字符串 | ⬜ 选填 | AIR | AIR（空气）/ WATER（水）/ CONCRETE（混凝土）/ WOOD（木材）/ GLASS（玻璃） |
| max_frequency | 最高仿真频率（用于自动优化网格） | 浮点数 | ⬜ 选填 | 无 | 单位：Hz，如 2000.0（>1000Hz 时自动细化网格） |

## 第三部分：声源参数（可添加多个）

适用于 general 和 room 模式；每个声源填写一行。

| 参数名 | 中文说明 | 类型 | 是否必填 | 默认值 | 取值范围 / 示例 |
| --- | --- | --- | --- | --- | --- |
| source_position_x | 声源 x 坐标 | 浮点数 | ✅ 必填 | 无 | 0 ~ spatial_domain_x，单位：米 |
| source_position_y | 声源 y 坐标 | 浮点数 | ✅ 必填 | 无 | 0 ~ spatial_domain_y，单位：米 |
| source_frequency | 声源频率 | 浮点数 | ⬜ 选填 | 440.0 | 单位：Hz，如 200.0、1000.0 |
| source_amplitude | 声源振幅 | 浮点数 | ⬜ 选填 | 1.0 | 正数，如 0.5、2.0 |
| source_phase | 声源初相位 | 浮点数 | ⬜ 选填 | 0.0 | 单位：弧度，如 0.0、3.14 |
| source_type | 声源类型 | 字符串 | ⬜ 选填 | POINT | POINT（点源）/ LINE（线源）/ PLANE（面源）/ DIRECTIONAL（定向源） |
| source_direction_x | 定向声源方向 x 分量 | 浮点数 | ⬜ 条件必填 | 无 | 仅定向/线源/面源需要，如 1.0（向右） |
| source_direction_y | 定向声源方向 y 分量 | 浮点数 | ⬜ 条件必填 | 无 | 仅定向/线源/面源需要，如 0.0 |

## 第四部分：障碍物参数（可添加多个，选填）

适用于 general 和 room 模式；每个障碍物填写一行。

| 参数名 | 中文说明 | 类型 | 是否必填 | 默认值 | 取值范围 / 示例 |
| --- | --- | --- | --- | --- | --- |
| obstacle_position_x | 障碍物中心 x 坐标 | 浮点数 | ✅ 必填 | 无 | 0 ~ spatial_domain_x，单位：米 |
| obstacle_position_y | 障碍物中心 y 坐标 | 浮点数 | ✅ 必填 | 无 | 0 ~ spatial_domain_y，单位：米 |
| obstacle_width | 障碍物宽度 | 浮点数 | ✅ 必填 | 无 | 正数，单位：米，如 0.5 |
| obstacle_height | 障碍物高度 | 浮点数 | ✅ 必填 | 无 | 正数，单位：米，如 2.0 |
| obstacle_material | 障碍物材料 | 字符串 | ✅ 必填 | 无 | AIR / WATER / CONCRETE / WOOD / GLASS |
| obstacle_is_reflecting | 是否反射声波 | 布尔值 | ⬜ 选填 | True | True / False |
| obstacle_is_absorbing | 是否吸收声波 | 布尔值 | ⬜ 选填 | False | True / False |

## 第五部分：房间声学专用参数（仅 room 模式）

| 参数名 | 中文说明 | 类型 | 是否必填 | 默认值 | 取值范围 / 示例 |
| --- | --- | --- | --- | --- | --- |
| room_size_x | 房间宽度 | 浮点数 | ✅ 必填 | 无 | 正数，单位：米，如 10.0 |
| room_size_y | 房间高度 | 浮点数 | ✅ 必填 | 无 | 正数，单位：米，如 8.0 |

## 第六部分：波束形成专用参数（仅 beamforming 模式）

| 参数名 | 中文说明 | 类型 | 是否必填 | 默认值 | 取值范围 / 示例 |
| --- | --- | --- | --- | --- | --- |
| mic_positions | 麦克风坐标列表 | 列表[[x,y],...] | ✅ 必填 | 无 | 至少 2 个麦克风，如 [[0,0],[1,0],[2,0]] |
| beamforming_sample_rate | 波束形成采样率 | 整数 | ⬜ 选填 | 44100 | 单位：Hz，如 22050、44100 |
| beamforming_speed_of_sound | 声速 | 浮点数 | ⬜ 选填 | 343.0 | 单位：m/s，空气中约 343 |

## 填写说明

- ✅ 必填：不填写将导致程序报错或无法运行。
- ⬜ 选填：有默认值，不填则使用默认值。
- 声源和障碍物都可以不填（0 个）或填写多组。
- boundary_type 推荐：封闭空间选 reflecting，开放场地选 absorbing。
- grid_resolution 越大越精确，但计算时间显著增加；建议从 (100,100) 开始。

---

## 纯净参数填写表（可直接填）

> 说明：本表仅保留“参数名 + 你的值”，便于你快速填写与汇总。

| 参数名 | 你的值 |
| --- | --- |
| simulation_type |  |
| spatial_domain_x |  |
| spatial_domain_y |  |
| grid_resolution_x |  |
| grid_resolution_y |  |
| time_step |  |
| total_time |  |
| boundary_type |  |
| damping |  |
| medium_material |  |
| max_frequency |  |
| source_position_x |  |
| source_position_y |  |
| source_frequency |  |
| source_amplitude |  |
| source_phase |  |
| source_type |  |
| source_direction_x |  |
| source_direction_y |  |
| obstacle_position_x |  |
| obstacle_position_y |  |
| obstacle_width |  |
| obstacle_height |  |
| obstacle_material |  |
| obstacle_is_reflecting |  |
| obstacle_is_absorbing |  |
| room_size_x |  |
| room_size_y |  |
| mic_positions |  |
| beamforming_sample_rate |  |
| beamforming_speed_of_sound |  |
