"""
基于智能体的声学仿真模拟器
Acoustic Simulation Engine Based on Agent

功能特性:
- 2D声场波动方程有限差分求解
- 多种声源类型（点源、线源、面源、定向声源）
- 声波传播与吸收模拟
- 障碍物反射与散射
- 吸收/反射/周期边界条件
- 实时可视化与频域分析
- 房间声学（RT60）
- 波束形成（近场距离模型）
- AcousticAgent 智能体（感知 / 移动 / 发声）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class SoundSourceType(Enum):
    """声源类型枚举"""
    POINT = "point"           # 点声源
    LINE = "line"             # 线声源
    PLANE = "plane"           # 面声源
    DIRECTIONAL = "directional" # 定向声源


class MediumType(Enum):
    """介质类型枚举"""
    AIR = "air"               # 空气
    WATER = "water"           # 水
    CONCRETE = "concrete"     # 混凝土
    WOOD = "wood"             # 木材
    GLASS = "glass"           # 玻璃


@dataclass
class MaterialProperties:
    """材料声学属性"""
    name: str
    density: float = 1.21           # 密度 (kg/m³)
    speed_of_sound: float = 343.0   # 声速 (m/s)
    absorption_coeff: float = 0.05   # 吸收系数
    scattering_coeff: float = 0.01   # 散射系数
    porosity: float = 0.0            # 孔隙率
    
    @property
    def acoustic_impedance(self) -> float:
        """声阻抗 Z = ρ * c"""
        return self.density * self.speed_of_sound
    
    @classmethod
    def get_material(cls, material_type: MediumType) -> 'MaterialProperties':
        """获取预设材料属性"""
        materials = {
            MediumType.AIR: cls("空气", 1.21, 343.0, 0.01, 0.001),
            MediumType.WATER: cls("水", 1000.0, 1480.0, 0.001, 0.0001),
            MediumType.CONCRETE: cls("混凝土", 2300.0, 3200.0, 0.05, 0.1),
            MediumType.WOOD: cls("木材", 600.0, 4000.0, 0.1, 0.15),
            MediumType.GLASS: cls("玻璃", 2500.0, 5300.0, 0.03, 0.02),
        }
        return materials.get(material_type, materials[MediumType.AIR])


@dataclass
class SoundSource:
    """声源类"""
    position: np.ndarray          # 位置 (x, y) 或 (x, y, z)
    frequency: float = 440.0      # 频率 (Hz)
    amplitude: float = 1.0         # 振幅
    phase: float = 0.0             # 初相位 (弧度)
    source_type: SoundSourceType = SoundSourceType.POINT
    direction: Optional[np.ndarray] = None  # 定向声源方向
    active: bool = True
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        if self.direction is not None:
            self.direction = np.array(self.direction, dtype=float)
            norm = np.linalg.norm(self.direction)
            if norm > 0:
                self.direction = self.direction / norm
            else:
                self.direction = None
    
    def get_signal(self, time: float) -> float:
        """获取时刻t的声压值"""
        if not self.active:
            return 0.0
        omega = 2 * np.pi * self.frequency
        return self.amplitude * np.sin(omega * time + self.phase)
    
    def get_wave_packet(self, time: float, bandwidth: float = 50.0) -> float:
        """获取高斯波包信号"""
        if not self.active:
            return 0.0
        omega = 2 * np.pi * self.frequency
        envelope = np.exp(-(bandwidth * (time - 1.0))**2 / 2)
        return self.amplitude * envelope * np.sin(omega * time + self.phase)


@dataclass
class Obstacle:
    """障碍物类"""
    position: np.ndarray        # 位置
    size: np.ndarray            # 尺寸
    material: MaterialProperties
    is_absorbing: bool = False  # 是否为吸收边界
    is_reflecting: bool = True   # 是否反射声波
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.size = np.array(self.size, dtype=float)


@dataclass
class AcousticConfig:
    """声学仿真配置"""
    sample_rate: int = 2000              # 采样率
    grid_resolution: Tuple[int, int] = (200, 200)  # 网格分辨率
    spatial_domain: Tuple[float, float] = (10.0, 10.0)  # 空间域 (m)
    time_step: float = 0.001             # 时间步长 (s)
    total_time: float = 2.0               # 总仿真时间 (s)
    cfl_number: float = 0.5               # CFL稳定性条件数
    boundary_type: str = "absorbing"     # 边界类型: 'absorbing', 'reflecting', 'periodic'
    damping: float = 0.01                # 阻尼系数
    material: MaterialProperties = field(default_factory=lambda: MaterialProperties.get_material(MediumType.AIR))

    def __post_init__(self):
        """保持采样率与时间步长一致，避免频谱坐标歧义"""
        if self.time_step > 0:
            inferred_sample_rate = int(round(1.0 / self.time_step))
            if inferred_sample_rate > 0 and self.sample_rate != inferred_sample_rate:
                self.sample_rate = inferred_sample_rate


class AcousticSimulator:
    """
    声学仿真引擎
    实现2D波动方程的有限差分法求解
    """
    
    def __init__(self, config: AcousticConfig):
        self.config = config
        self.nx, self.ny = config.grid_resolution
        self.dx = config.spatial_domain[0] / self.nx
        self.dy = config.spatial_domain[1] / self.ny
        self.dt = config.time_step
        self.c = config.material.speed_of_sound
        
        # 验证CFL稳定性条件
        self._check_cfl_condition()
        
        # 初始化声压场
        self.pressure = np.zeros((self.nx, self.ny))
        self.pressure_prev = np.zeros((self.nx, self.ny))
        self.pressure_next = np.zeros((self.nx, self.ny))
        
        # 声源列表
        self.sources: List[SoundSource] = []
        
        # 障碍物列表
        self.obstacles: List[Obstacle] = []
        
        # 障碍物掩码（预计算）
        self.obstacle_mask = np.zeros((self.nx, self.ny))
        self.absorption_mask = np.zeros((self.nx, self.ny))
        
        # 时间追踪
        self.current_time = 0.0
        self.time_steps = 0
        
        # 声场历史记录（deque 自动丢弃最旧帧，避免 O(n) pop(0)）
        self.max_history = 500
        self.history: Deque[np.ndarray] = deque(maxlen=self.max_history)
        
        # 反射系数场
        self.reflection_coeff = np.ones((self.nx, self.ny))

        # 预计算网格坐标，供声源激励使用（避免每步重复 meshgrid）
        self._grid_x, self._grid_y = np.meshgrid(
            np.arange(self.nx), np.arange(self.ny), indexing='ij'
        )
        
    def _check_cfl_condition(self):
        """检查CFL稳定性条件"""
        cfl_x = self.c * self.dt / self.dx
        cfl_y = self.c * self.dt / self.dy
        cfl = np.sqrt(cfl_x**2 + cfl_y**2)
        
        if cfl > self.config.cfl_number:
            print(f"警告: CFL数 {cfl:.3f} 超过设定值 {self.config.cfl_number}")
            print("调整时间步长以满足稳定性条件...")
            max_dt = self.config.cfl_number * min(self.dx, self.dy) / self.c
            self.dt = max_dt * 0.9
            # 将调整后的时间步长同步回配置，保持 sample_rate 一致
            self.config.time_step = self.dt
            self.config.sample_rate = int(round(1.0 / self.dt))
            print(f"新时间步长: {self.dt:.6f} s")
        
    def add_source(self, source: SoundSource):
        """添加声源"""
        self.sources.append(source)
        
    def add_obstacle(self, obstacle: Obstacle):
        """添加障碍物"""
        self.obstacles.append(obstacle)
        self._update_obstacle_masks()
        
    def _update_obstacle_masks(self):
        """更新障碍物掩码"""
        self.obstacle_mask.fill(0)
        self.absorption_mask.fill(0)
        self.reflection_coeff.fill(1.0)
        
        for obs in self.obstacles:
            x_pos, y_pos = obs.position
            half_size = obs.size / 2
            
            x_min = max(0, int((x_pos - half_size[0]) / self.dx))
            x_max = min(self.nx, int((x_pos + half_size[0]) / self.dx))
            y_min = max(0, int((y_pos - half_size[1]) / self.dy))
            y_max = min(self.ny, int((y_pos + half_size[1]) / self.dy))
            
            if x_max > x_min and y_max > y_min:
                self.obstacle_mask[x_min:x_max, y_min:y_max] = 1
                
                if obs.is_absorbing:
                    self.absorption_mask[x_min:x_max, y_min:y_max] = obs.material.absorption_coeff
                    self.reflection_coeff[x_min:x_max, y_min:y_max] = 1 - obs.material.absorption_coeff
                elif obs.is_reflecting:
                    self.reflection_coeff[x_min:x_max, y_min:y_max] = 1 - obs.material.scattering_coeff
                    
    def _apply_source_excitation(self):
        """施加声源激励"""
        for source in self.sources:
            if not source.active:
                continue
                
            # 计算声源位置对应的网格索引
            px = int(source.position[0] / self.dx)
            py = int(source.position[1] / self.dy)
            
            if 0 <= px < self.nx and 0 <= py < self.ny:
                if source.source_type == SoundSourceType.POINT:
                    # 点声源
                    signal = source.get_wave_packet(self.current_time)
                    self.pressure[px, py] += signal
                    
                elif source.source_type == SoundSourceType.LINE:
                    # 线声源：沿给定方向延伸；无方向时默认水平线（沿 x 轴）
                    # SoundSource.__post_init__ 保证 direction 已归一化
                    signal = source.get_signal(self.current_time)
                    if source.direction is not None:
                        dir_x, dir_y = source.direction
                        # 法向量（垂直于线方向）
                        perp_x, perp_y = -dir_y, dir_x
                        # 使用预计算的网格坐标
                        perp_dist = np.abs(
                            (self._grid_x - px) * perp_x + (self._grid_y - py) * perp_y
                        )
                        self.pressure[perp_dist < 0.5] += signal
                    else:
                        # 默认：沿 x 轴的水平线声源
                        self.pressure[:, py] += signal

                elif source.source_type == SoundSourceType.DIRECTIONAL:
                    # 定向声源：主方向前向增强，反方向抑制，形成心形指向性
                    # SoundSource.__post_init__ 保证 direction 已归一化
                    signal = source.get_wave_packet(self.current_time)
                    self.pressure[px, py] += signal
                    if source.direction is not None:
                        dir_x, dir_y = source.direction
                        # 前向相邻格点施加同相激励
                        fwd_px = int(round(px + dir_x))
                        fwd_py = int(round(py + dir_y))
                        if 0 <= fwd_px < self.nx and 0 <= fwd_py < self.ny:
                            self.pressure[fwd_px, fwd_py] += signal * 0.5
                        # 后向相邻格点施加反相激励，抑制后向辐射
                        bwd_px = int(round(px - dir_x))
                        bwd_py = int(round(py - dir_y))
                        if 0 <= bwd_px < self.nx and 0 <= bwd_py < self.ny:
                            self.pressure[bwd_px, bwd_py] -= signal * 0.5

                elif source.source_type == SoundSourceType.PLANE:
                    # 面声源（2D 中为平面波前）：激励垂直于传播方向的直线
                    # SoundSource.__post_init__ 保证 direction 已归一化
                    signal = source.get_wave_packet(self.current_time)
                    if source.direction is not None:
                        dir_x, dir_y = source.direction
                        # 使用预计算的网格坐标；投影距离 < 0.5 即在法平面上
                        proj = (self._grid_x - px) * dir_x + (self._grid_y - py) * dir_y
                        self.pressure[np.abs(proj) < 0.5] += signal
                    else:
                        # 默认：沿 x 方向传播的平面波（激励 y 轴线）
                        self.pressure[:, py] += signal
                        
    def _apply_boundary_conditions(self, target_field: Optional[np.ndarray] = None):
        """应用边界条件"""
        if target_field is None:
            target_field = self.pressure_next
        nx, ny = self.nx, self.ny
        
        if self.config.boundary_type == "absorbing":
            # 完美匹配层 (PML) 吸收边界
            pml_width = 10
            sigma = np.zeros((nx, ny))
            
            # 边界区域
            sigma[:pml_width, :] = np.linspace(0, 1, pml_width)[:, np.newaxis] * 2
            sigma[-pml_width:, :] = np.linspace(1, 0, pml_width)[:, np.newaxis] * 2
            sigma[:, :pml_width] = np.maximum(sigma[:, :pml_width], 
                                               np.linspace(0, 1, pml_width) * 2)
            sigma[:, -pml_width:] = np.maximum(sigma[:, -pml_width:], 
                                                np.linspace(1, 0, pml_width) * 2)
            
            # 应用阻尼
            damping_factor = np.exp(-sigma * self.config.damping)
            target_field *= damping_factor
            
        elif self.config.boundary_type == "reflecting":
            # 反射边界
            target_field[0, :] = target_field[1, :]
            target_field[-1, :] = target_field[-2, :]
            target_field[:, 0] = target_field[:, 1]
            target_field[:, -1] = target_field[:, -2]
            
        elif self.config.boundary_type == "periodic":
            # 周期边界
            target_field[0, :] = target_field[-2, :]
            target_field[-1, :] = target_field[1, :]
            target_field[:, 0] = target_field[:, -2]
            target_field[:, -1] = target_field[:, 1]
            
    def _apply_obstacle_conditions(self, target_field: Optional[np.ndarray] = None):
        """应用障碍物条件"""
        if target_field is None:
            target_field = self.pressure_next
        # 障碍物区域设为零（硬边界）
        target_field[self.obstacle_mask > 0] = 0
        
    def _finite_difference_step(self):
        """有限差分一步计算"""
        c2 = self.c ** 2
        dt2 = self.dt ** 2
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2

        # 每步先清空，避免边界残留
        self.pressure_next.fill(0.0)
        
        # 拉普拉斯算子
        laplacian = (
            (self.pressure[2:, 1:-1] - 2 * self.pressure[1:-1, 1:-1] + self.pressure[:-2, 1:-1]) / dx2 +
            (self.pressure[1:-1, 2:] - 2 * self.pressure[1:-1, 1:-1] + self.pressure[1:-1, :-2]) / dy2
        )
        
        # 波动方程: ∂²p/∂t² = c²∇²p
        self.pressure_next[1:-1, 1:-1] = (
            2 * self.pressure[1:-1, 1:-1] - 
            self.pressure_prev[1:-1, 1:-1] + 
            c2 * dt2 * laplacian
        )
        
        # 应用阻尼
        damping = 1 - self.config.damping
        self.pressure_next[1:-1, 1:-1] *= damping
        
        # 应用障碍物反射（直接乘以反射系数，物理上对应能量衰减）
        reflection_effect = self.reflection_coeff[1:-1, 1:-1]
        self.pressure_next[1:-1, 1:-1] *= reflection_effect
        
    def step(self) -> bool:
        """
        执行一步仿真
        返回: 是否继续仿真
        """
        # 施加声源
        self._apply_source_excitation()
        
        # 有限差分计算
        self._finite_difference_step()
        
        # 应用边界条件
        self._apply_boundary_conditions(self.pressure_next)
        
        # 应用障碍物
        self._apply_obstacle_conditions(self.pressure_next)
        
        # 更新场
        self.pressure_prev = self.pressure.copy()
        self.pressure = self.pressure_next.copy()
        
        # 更新时间
        self.current_time += self.dt
        self.time_steps += 1
        
        # 记录历史（deque 达到 maxlen 时自动丢弃最旧帧）
        self.history.append(self.pressure.copy())
        
        # 检查是否到达总仿真时间
        return self.current_time < self.config.total_time
        
    def simulate(self) -> np.ndarray:
        """
        运行完整仿真
        返回: 最终声压场
        """
        while self.step():
            pass
        return self.pressure
    
    def get_snapshot(self, time_index: int) -> Optional[np.ndarray]:
        """获取指定时刻的声场快照"""
        if 0 <= time_index < len(self.history):
            return self.history[time_index]
        return None
    
    def compute_frequency_spectrum(self, position: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算指定位置的频谱
        返回: (频率数组, 幅值数组)
        """
        if len(self.history) < 10:
            return np.array([]), np.array([])
        
        # 提取位置的时间序列
        x, y = position
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return np.array([]), np.array([])
        pressure_time = np.array([h[x, y] for h in self.history])
        
        # FFT计算
        n = len(pressure_time)
        fft_result = np.fft.fft(pressure_time)
        frequencies = np.fft.fftfreq(n, self.dt)[:n//2]
        amplitudes = np.abs(fft_result)[:n//2] * 2 / n
        
        return frequencies, amplitudes
    
    def compute_spl(self, position: Tuple[int, int]) -> float:
        """
        计算指定位置的声压级 (dB SPL)
        """
        if len(self.history) < 10:
            return 0.0
        
        x, y = position
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return 0.0
        pressure_rms = np.sqrt(np.mean([h[x, y]**2 for h in self.history]))
        
        if pressure_rms > 0:
            # 基准声压 20 μPa
            spl = 20 * np.log10(pressure_rms / 2e-5)
            return max(0, spl)
        return 0.0
    
    def get_pressure_at_point(self, position: np.ndarray) -> float:
        """获取指定位置的瞬时声压"""
        px = int(position[0] / self.dx)
        py = int(position[1] / self.dy)
        
        if 0 <= px < self.nx and 0 <= py < self.ny:
            return self.pressure[px, py]
        return 0.0


class AcousticVisualizer:
    """
    声学仿真可视化器
    """
    
    def __init__(self, simulator: AcousticSimulator):
        self.simulator = simulator
        self.fig = None
        self.ax = None
        self.im = None
        
        # 自定义颜色映射
        self.cmap = LinearSegmentedColormap.from_list(
            'acoustic', 
            ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        )
        
    def setup_figure(self, figsize: Tuple[int, int] = (12, 5)):
        """设置图形"""
        self.fig, self.ax = plt.subplots(1, 2, figsize=figsize)
        
    def plot_static(self, title: str = "声压场分布"):
        """静态绘图"""
        if self.fig is None:
            self.setup_figure()
            
        pressure = self.simulator.pressure
        
        # 声压场
        self.ax[0].clear()
        im = self.ax[0].imshow(
            pressure.T, 
            extent=[0, self.simulator.config.spatial_domain[0],
                   0, self.simulator.config.spatial_domain[1]],
            cmap=self.cmap,
            origin='lower',
            aspect='auto'
        )
        self.ax[0].set_xlabel('X (m)')
        self.ax[0].set_ylabel('Y (m)')
        self.ax[0].set_title('声压场')
        plt.colorbar(im, ax=self.ax[0], label='声压 (Pa)')
        
        # 绘制障碍物
        for obs in self.simulator.obstacles:
            x, y = obs.position
            dx, dy = obs.size
            rect = plt.Rectangle(
                (x - dx/2, y - dy/2), dx, dy,
                fill=True, facecolor='gray', alpha=0.5, edgecolor='black'
            )
            self.ax[0].add_patch(rect)
        
        # 绘制声源
        for src in self.simulator.sources:
            x, y = src.position[:2]
            self.ax[0].plot(x, y, 'r*', markersize=15, label=f'{src.frequency}Hz')
        
        self.ax[0].legend()
        
        # 频谱图
        if len(self.simulator.history) > 10:
            center = (self.simulator.nx // 2, self.simulator.ny // 2)
            freqs, amps = self.simulator.compute_frequency_spectrum(center)
            
            self.ax[1].clear()
            self.ax[1].plot(freqs, amps)
            self.ax[1].set_xlabel('频率 (Hz)')
            self.ax[1].set_ylabel('幅值')
            self.ax[1].set_title('中心点频谱')
            self.ax[1].set_xlim([0, self.simulator.config.sample_rate / 2])
            self.ax[1].grid(True, alpha=0.3)
        
        self.fig.suptitle(title)
        plt.tight_layout()
        
    def animate(self, interval: int = 50, save_path: Optional[str] = None):
        """动画展示"""
        if self.fig is None:
            self.setup_figure()
            
        def update(frame):
            if frame < len(self.simulator.history):
                pressure = self.simulator.history[frame]
                self.ax[0].clear()
                self.ax[0].imshow(
                    pressure.T,
                    extent=[0, self.simulator.config.spatial_domain[0],
                           0, self.simulator.config.spatial_domain[1]],
                    cmap=self.cmap,
                    origin='lower',
                    aspect='auto',
                    vmin=-0.5,
                    vmax=0.5
                )
                
                # 绘制障碍物
                for obs in self.simulator.obstacles:
                    x, y = obs.position
                    dx, dy = obs.size
                    rect = plt.Rectangle(
                        (x - dx/2, y - dy/2), dx, dy,
                        fill=True, facecolor='gray', alpha=0.5
                    )
                    self.ax[0].add_patch(rect)
                
                # 绘制声源
                for src in self.simulator.sources:
                    self.ax[0].plot(src.position[0], src.position[1], 'r*', markersize=15)
                
                self.ax[0].set_title(f'声压场 t = {frame * self.simulator.dt:.3f}s')
                self.ax[0].set_xlabel('X (m)')
                self.ax[0].set_ylabel('Y (m)')
                
        anim = FuncAnimation(
            self.fig, update,
            frames=len(self.simulator.history),
            interval=interval,
            repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            
        plt.show()
        return anim
    
    def plot_pressure_profile(self, axis: str = 'x', position: float = None):
        """绘制声压剖面图"""
        if self.fig is None:
            self.setup_figure()
            
        pressure = self.simulator.pressure
        nx, ny = self.simulator.nx, self.simulator.ny
        
        self.ax[0].clear()
        
        if axis == 'x':
            if position is None:
                position = ny // 2
            x = np.linspace(0, self.simulator.config.spatial_domain[0], nx)
            self.ax[0].plot(x, pressure[:, position])
            self.ax[0].set_xlabel('X (m)')
        else:
            if position is None:
                position = nx // 2
            y = np.linspace(0, self.simulator.config.spatial_domain[1], ny)
            self.ax[0].plot(y, pressure[position, :])
            self.ax[0].set_xlabel('Y (m)')
            
        self.ax[0].set_ylabel('声压 (Pa)')
        self.ax[0].set_title(f'声压剖面 ({axis}轴, position={position})')
        self.ax[0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class RoomAcousticSimulator(AcousticSimulator):
    """
    房间声学仿真器
    继承自基础声学仿真器，增加房间特有的功能
    """
    
    def __init__(self, config: AcousticConfig, room_size: Tuple[float, float]):
        super().__init__(config)
        self.room_size = room_size
        
        # 房间声学参数
        self.reverberation_time = 0.0
        self.clarity = 0.0
        self.definiteness = 0.0
        
    def add_wall(self, position: np.ndarray, size: np.ndarray, 
                 material: MaterialProperties, absorption: float = 0.1):
        """添加墙壁"""
        wall = Obstacle(
            position=position,
            size=size,
            material=material,
            is_absorbing=True,
            is_reflecting=True
        )
        self.add_obstacle(wall)
        
    def compute_rt60(self) -> float:
        """
        计算混响时间RT60（声压衰减60dB所需时间）
        """
        if len(self.history) < 100:
            return 0.0
            
        # 计算能量衰减曲线
        center = (self.nx // 2, self.ny // 2)
        energy = np.array([h[center[0], center[1]]**2 for h in self.history])
        
        max_energy = np.max(energy)
        if max_energy <= 0:
            return 0.0

        threshold_60db = max_energy * 1e-6  # 60dB衰减
        
        # 只对正值取对数，避免 log(0) 警告
        decay_mask = energy > threshold_60db
        if np.sum(decay_mask) < 10:
            return 0.0
            
        t = np.arange(len(energy)) * self.dt
        nonzero_energy = np.where(energy > 0, energy, np.nan)
        log_energy = 10 * np.log10(nonzero_energy / max_energy)
        
        # 同时过滤掉 NaN（零值产生）
        valid_mask = decay_mask & np.isfinite(log_energy)
        valid_t = t[valid_mask]
        valid_energy = log_energy[valid_mask]
        
        if len(valid_t) > 2:
            coeffs = np.polyfit(valid_t, valid_energy, 1)
            slope = coeffs[0]
            
            if slope < 0:
                rt60 = 60 / abs(slope)
                self.reverberation_time = rt60
                return rt60
                
        return 0.0


class BeamformingSimulator:
    """
    波束形成仿真器
    用于麦克风阵列声源定位
    """
    
    def __init__(self, array_config: Dict):
        self.array_config = array_config
        self.mic_positions = np.array(array_config['mic_positions'])
        self.num_mics = len(self.mic_positions)
        self.sample_rate = array_config.get('sample_rate', 44100)
        self.c = array_config.get('speed_of_sound', 343.0)
        
    def delay_and_sum(self, signals: np.ndarray, scan_grid: np.ndarray,
                      steering_angle: float = 0) -> np.ndarray:
        """
        延迟求和波束形成（近场距离模型）
        
        参数:
            signals: 麦克风信号 (num_mics, num_samples)
            scan_grid: 扫描网格 (num_points, 2)
            steering_angle: 保留参数（近场模式下不使用）
        
        返回:
            波束图（每个扫描点的估计能量）
        """
        num_samples = signals.shape[1]
        scan_grid = np.asarray(scan_grid, dtype=float)  # 统一在循环外转换，避免重复开销
        num_points = len(scan_grid)
        beamforming_map = np.zeros(num_points)
        
        # 阵列中心作为参考点
        array_center = np.mean(self.mic_positions, axis=0)
        
        for i, point in enumerate(scan_grid):
            # 近场：用各麦克风到扫描点的实际距离计算传播延迟
            distances = np.linalg.norm(self.mic_positions - point, axis=1)  # (num_mics,)
            ref_distance = np.linalg.norm(array_center - point)
            # 相对延迟（正值：信号到达该麦克风更晚）
            delays = (distances - ref_distance) / self.c
            delay_samples = np.round(delays * self.sample_rate).astype(int)
            
            # 对整段信号做延迟对齐后再求和
            aligned = np.zeros_like(signals)
            for m in range(self.num_mics):
                d = delay_samples[m]
                if d > 0 and d < num_samples:
                    aligned[m, d:] = signals[m, :num_samples - d]
                elif d < 0 and -d < num_samples:
                    aligned[m, :num_samples + d] = signals[m, -d:]
                else:
                    aligned[m, :] = signals[m, :]

            beam = np.mean(aligned, axis=0)
            beamforming_map[i] = np.mean(beam**2)
            
        return beamforming_map


class AcousticAgent:
    """
    声学智能体

    代表仿真环境中具有自主行为的实体，能够：
    - 感知（perceive）：读取当前位置的声压和声压级
    - 移动（move）：在仿真空间中改变位置
    - 发声（emit_sound）：在当前位置激活/更新声源
    - 静默（silence）：停止发出声音

    使用示例::

        agent = AcousticAgent(position=np.array([5.0, 5.0]), name="Agent-1")
        agent.attach(simulator)
        agent.emit_sound(frequency=440.0, amplitude=1.0)
        simulator.step()
        state = agent.perceive()
        agent.move(np.array([6.0, 5.0]))
    """

    def __init__(self, position: np.ndarray, name: str = "Agent"):
        self.position = np.array(position, dtype=float)
        self.name = name
        self.simulator: Optional[AcousticSimulator] = None
        self.perceived_pressure: float = 0.0
        self.perceived_spl: float = 0.0
        self.action_history: List[Dict] = []
        self._sound_source: Optional[SoundSource] = None

    def attach(self, simulator: 'AcousticSimulator') -> None:
        """将智能体绑定到仿真器"""
        self.simulator = simulator

    def perceive(self) -> Dict:
        """
        感知当前位置的声场状态

        返回:
            包含 position / pressure / spl / time 的字典
        """
        if self.simulator is None:
            return {}
        self.perceived_pressure = self.simulator.get_pressure_at_point(self.position)
        # 将位置裁剪到有效网格范围，防止越界
        x = int(np.clip(self.position[0] / self.simulator.dx, 0, self.simulator.nx - 1))
        y = int(np.clip(self.position[1] / self.simulator.dy, 0, self.simulator.ny - 1))
        self.perceived_spl = self.simulator.compute_spl((x, y))
        return {
            'position': self.position.copy(),
            'pressure': self.perceived_pressure,
            'spl': self.perceived_spl,
            'time': self.simulator.current_time,
        }

    def move(self, new_position: np.ndarray) -> None:
        """
        移动到新位置

        如果智能体当前有声源，声源位置也会同步更新。
        """
        new_position = np.array(new_position, dtype=float)
        self.action_history.append({
            'type': 'move',
            'from': self.position.copy(),
            'to': new_position.copy(),
            'time': self.simulator.current_time if self.simulator else 0.0,
        })
        self.position = new_position
        if self._sound_source is not None:
            self._sound_source.position = new_position.copy()

    def emit_sound(
        self,
        frequency: float,
        amplitude: float,
        source_type: SoundSourceType = SoundSourceType.POINT,
    ) -> SoundSource:
        """
        在当前位置发出声音

        首次调用时向仿真器注册声源；后续调用更新频率和振幅。

        返回:
            关联的 SoundSource 实例
        """
        if self.simulator is None:
            raise RuntimeError("Agent not attached to a simulator. Call attach() first.")
        if self._sound_source is None:
            self._sound_source = SoundSource(
                position=self.position.copy(),
                frequency=frequency,
                amplitude=amplitude,
                source_type=source_type,
            )
            self.simulator.add_source(self._sound_source)
        else:
            self._sound_source.frequency = frequency
            self._sound_source.amplitude = amplitude
            self._sound_source.source_type = source_type
            self._sound_source.active = True
        self.action_history.append({
            'type': 'emit',
            'frequency': frequency,
            'amplitude': amplitude,
            'time': self.simulator.current_time,
        })
        return self._sound_source

    def silence(self) -> None:
        """停止当前声源"""
        if self._sound_source is not None:
            self._sound_source.active = False
            self.action_history.append({
                'type': 'silence',
                'time': self.simulator.current_time if self.simulator else 0.0,
            })


# ============== 示例和测试 ==============

def create_default_simulation():
    """创建默认仿真配置"""
    
    # 配置参数
    config = AcousticConfig(
        sample_rate=2000,
        grid_resolution=(200, 200),
        spatial_domain=(10.0, 10.0),
        time_step=0.001,
        total_time=2.0,
        boundary_type="absorbing",
        damping=0.02
    )
    
    # 创建仿真器
    simulator = AcousticSimulator(config)
    
    # 添加声源
    source1 = SoundSource(
        position=np.array([5.0, 5.0]),
        frequency=200.0,
        amplitude=1.0,
        source_type=SoundSourceType.POINT
    )
    simulator.add_source(source1)
    
    # 添加障碍物
    wall = Obstacle(
        position=np.array([7.0, 5.0]),
        size=np.array([0.5, 3.0]),
        material=MaterialProperties.get_material(MediumType.CONCRETE),
        is_reflecting=True,
        is_absorbing=False
    )
    simulator.add_obstacle(wall)
    
    return simulator


def run_basic_example():
    """基础示例"""
    print("=" * 50)
    print("声学仿真模拟器 - 基础示例")
    print("=" * 50)
    
    # 创建仿真
    simulator = create_default_simulation()
    
    # 运行仿真
    print("\n运行仿真...")
    final_pressure = simulator.simulate()
    
    # 输出统计信息
    print(f"\n仿真完成!")
    print(f"总时间步数: {simulator.time_steps}")
    print(f"最终时间: {simulator.current_time:.3f}s")
    print(f"声压范围: [{final_pressure.min():.4f}, {final_pressure.max():.4f}] Pa")
    print(f"声压RMS: {np.sqrt(np.mean(final_pressure**2)):.4f} Pa")
    
    # 中心点SPL
    center_spl = simulator.compute_spl((100, 100))
    print(f"中心点声压级: {center_spl:.1f} dB SPL")
    
    # 可视化
    visualizer = AcousticVisualizer(simulator)
    visualizer.setup_figure()
    visualizer.plot_static("基础声学仿真")
    plt.show()


def run_multiple_sources_example():
    """多声源示例"""
    print("\n" + "=" * 50)
    print("声学仿真模拟器 - 多声源示例")
    print("=" * 50)
    
    config = AcousticConfig(
        sample_rate=2000,
        grid_resolution=(250, 250),
        spatial_domain=(12.0, 12.0),
        time_step=0.001,
        total_time=3.0,
        damping=0.015
    )
    
    simulator = AcousticSimulator(config)
    
    # 添加多个不同频率的声源
    frequencies = [100, 200, 400, 800]
    positions = [(3, 6), (9, 3), (6, 9), (3, 3)]
    
    for i, (freq, pos) in enumerate(zip(frequencies, positions)):
        source = SoundSource(
            position=np.array(pos),
            frequency=freq,
            amplitude=0.8,
            source_type=SoundSourceType.POINT
        )
        simulator.add_source(source)
    
    # 添加障碍物
    for i in range(3):
        wall = Obstacle(
            position=np.array([6 + i*2, 6]),
            size=np.array([0.3, 2.0]),
            material=MaterialProperties.get_material(MediumType.CONCRETE)
        )
        simulator.add_obstacle(wall)
    
    print("\n运行多声源仿真...")
    final_pressure = simulator.simulate()
    
    print(f"仿真完成! 时间步: {simulator.time_steps}")
    
    # 可视化
    visualizer = AcousticVisualizer(simulator)
    visualizer.setup_figure()
    visualizer.plot_static("多声源声学仿真")
    plt.show()


def run_room_acoustic_example():
    """房间声学示例"""
    print("\n" + "=" * 50)
    print("声学仿真模拟器 - 房间声学示例")
    print("=" * 50)
    
    config = AcousticConfig(
        sample_rate=1500,
        grid_resolution=(200, 200),
        spatial_domain=(10.0, 10.0),
        time_step=0.001,
        total_time=4.0,
        boundary_type="reflecting",
        damping=0.005
    )
    
    room_sim = RoomAcousticSimulator(config, room_size=(10.0, 10.0))
    
    # 添加声源
    source = SoundSource(
        position=np.array([5.0, 5.0]),
        frequency=150.0,
        amplitude=1.5,
        source_type=SoundSourceType.POINT
    )
    room_sim.add_source(source)
    
    # 添加墙壁
    wall_material = MaterialProperties.get_material(MediumType.CONCRETE)
    walls = [
        (np.array([5.0, 0.2]), np.array([10.0, 0.4])),  # 下墙
        (np.array([5.0, 9.8]), np.array([10.0, 0.4])),  # 上墙
        (np.array([0.2, 5.0]), np.array([0.4, 10.0])),  # 左墙
        (np.array([9.8, 5.0]), np.array([0.4, 10.0])),  # 右墙
    ]
    
    for pos, size in walls:
        room_sim.add_wall(pos, size, wall_material)
    
    print("\n运行房间声学仿真...")
    final_pressure = room_sim.simulate()
    
    # 计算混响时间
    rt60 = room_sim.compute_rt60()
    print(f"混响时间RT60: {rt60:.3f}s" if rt60 > 0 else "混响时间计算中...")
    
    # 可视化
    visualizer = AcousticVisualizer(room_sim)
    visualizer.setup_figure()
    visualizer.plot_static("房间声学仿真")
    plt.show()


def run_interactive_simulation():
    """交互式仿真"""
    print("\n" + "=" * 50)
    print("声学仿真模拟器 - 交互式仿真")
    print("=" * 50)
    
    config = AcousticConfig(
        sample_rate=2000,
        grid_resolution=(150, 150),
        spatial_domain=(8.0, 8.0),
        time_step=0.001,
        total_time=5.0,
        damping=0.01
    )
    
    simulator = AcousticSimulator(config)
    
    # 添加声源
    source = SoundSource(
        position=np.array([4.0, 4.0]),
        frequency=250.0,
        amplitude=1.0
    )
    simulator.add_source(source)
    
    visualizer = AcousticVisualizer(simulator)
    visualizer.setup_figure(figsize=(14, 5))
    
    print("\n开始交互式仿真 (按Ctrl+C停止)...")
    print("注意: 这是一个连续运行的示例\n")
    
    # 初始化绘图
    plt.ion()
    
    try:
        for step in range(500):
            simulator.step()
            
            if step % 10 == 0:
                visualizer.plot_static(f"交互式仿真 Step {step}")
                plt.draw()
                plt.pause(0.01)
                
    except KeyboardInterrupt:
        print("\n仿真已停止")
    
    plt.ioff()
    plt.show()


# ============== 主程序 ==============

if __name__ == "__main__":
    import sys
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         基于智能体的声学仿真模拟器 (Acoustic Simulation)      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  功能:                                                       ║
    ║  - 2D声场波动方程求解                                        ║
    ║  - 多种声源类型（点、线、面、定向）                           ║
    ║  - 障碍物反射与吸收                                          ║
    ║  - 边界条件（PML吸收、反射、周期）                            ║
    ║  - 频域分析                                                  ║
    ║  - 房间声学仿真                                              ║
    ║  - 波束形成                                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 运行基础示例
    run_basic_example()
    
    # 可选择运行其他示例（取消注释即可运行）
    # run_multiple_sources_example()
    # run_room_acoustic_example()
    # run_interactive_simulation()
