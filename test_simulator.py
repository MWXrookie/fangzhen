import json
import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from shengxuefangzhen import (
    AcousticAgent,
    AcousticConfig,
    AcousticSimulator,
    AcousticSimulator3D,
    AcousticVisualizer,
    AcousticVisualizer3D,
    BaseAcousticSimulator,
    BeamformingSimulator,
    MaterialProperties,
    MediumType,
    Obstacle,
    RoomAcousticSimulator,
    SoundSource,
    SoundSourceType,
    create_simulator,
)


class TestAcousticSimulator(unittest.TestCase):
    def test_out_of_bounds_frequency_returns_empty(self):
        config = AcousticConfig(
            grid_resolution=(40, 40),
            spatial_domain=(4.0, 4.0),
            time_step=0.0002,
            total_time=0.01,
        )
        simulator = AcousticSimulator(config)
        source = SoundSource(
            position=np.array([2.0, 2.0]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.POINT,
        )
        simulator.add_source(source)
        simulator.simulate()

        freqs, amps = simulator.compute_frequency_spectrum((1000, 1000))
        self.assertEqual(freqs.size, 0)
        self.assertEqual(amps.size, 0)

    def test_out_of_bounds_spl_returns_zero(self):
        config = AcousticConfig(
            grid_resolution=(40, 40),
            spatial_domain=(4.0, 4.0),
            time_step=0.0002,
            total_time=0.01,
        )
        simulator = AcousticSimulator(config)
        source = SoundSource(
            position=np.array([2.0, 2.0]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.POINT,
        )
        simulator.add_source(source)
        simulator.simulate()

        spl = simulator.compute_spl((1000, 1000))
        self.assertEqual(spl, 0.0)

    def test_obstacle_masks_next_field(self):
        config = AcousticConfig(
            grid_resolution=(50, 50),
            spatial_domain=(5.0, 5.0),
            time_step=0.0002,
            total_time=0.005,
            boundary_type="reflecting",
        )
        simulator = AcousticSimulator(config)

        source = SoundSource(
            position=np.array([2.5, 2.5]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.POINT,
        )
        simulator.add_source(source)

        obstacle = Obstacle(
            position=np.array([2.5, 2.5]),
            size=np.array([0.4, 0.4]),
            material=MaterialProperties.get_material(MediumType.CONCRETE),
            is_reflecting=True,
            is_absorbing=False,
        )
        simulator.add_obstacle(obstacle)

        simulator.step()
        masked_values = simulator.pressure[simulator.obstacle_mask > 0]
        self.assertTrue(np.allclose(masked_values, 0.0))

    def test_sample_rate_sync_with_time_step(self):
        config = AcousticConfig(sample_rate=2000, time_step=0.002)
        self.assertEqual(config.sample_rate, 500)

    def test_zero_direction_becomes_none(self):
        source = SoundSource(
            position=np.array([1.0, 1.0]),
            source_type=SoundSourceType.DIRECTIONAL,
            direction=np.array([0.0, 0.0]),
        )
        self.assertIsNone(source.direction)


class TestCreateSimulator(unittest.TestCase):
    """工厂函数 create_simulator 的单元测试"""

    def test_mic_positions_returns_beamforming(self):
        """含 mic_positions 参数 → BeamformingSimulator"""
        params = {
            'mic_positions': [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        }
        sim = create_simulator(params)
        self.assertIsInstance(sim, BeamformingSimulator)
        self.assertIsInstance(sim, BaseAcousticSimulator)

    def test_room_size_returns_room_simulator(self):
        """含 room_size 参数 → RoomAcousticSimulator"""
        params = {
            'room_size': (8.0, 6.0),
            'grid_resolution': (40, 40),
            'spatial_domain': (8.0, 6.0),
            'time_step': 0.001,
            'total_time': 0.01,
        }
        sim = create_simulator(params)
        self.assertIsInstance(sim, RoomAcousticSimulator)
        self.assertIsInstance(sim, AcousticSimulator)

    def test_walls_key_returns_room_simulator(self):
        """含 walls 参数（无 room_size）→ RoomAcousticSimulator，room_size 降级为 spatial_domain"""
        params = {
            'walls': [],
            'spatial_domain': (5.0, 5.0),
            'grid_resolution': (40, 40),
            'time_step': 0.001,
            'total_time': 0.005,
        }
        sim = create_simulator(params)
        self.assertIsInstance(sim, RoomAcousticSimulator)

    def test_default_returns_acoustic_simulator(self):
        """无特殊键 → AcousticSimulator（非 Room / Beamforming）"""
        params = {
            'grid_resolution': (40, 40),
            'spatial_domain': (4.0, 4.0),
            'time_step': 0.0002,
            'total_time': 0.005,
        }
        sim = create_simulator(params)
        self.assertIsInstance(sim, AcousticSimulator)
        self.assertNotIsInstance(sim, RoomAcousticSimulator)

    def test_high_frequency_auto_grid(self):
        """高频（>1kHz）且未指定网格 → 自动推算精细网格，分辨率 ≥ 100"""
        params = {
            'max_frequency': 2000.0,
            'spatial_domain': (5.0, 5.0),
            'time_step': 0.00005,
            'total_time': 0.001,
        }
        sim = create_simulator(params)
        self.assertIsInstance(sim, AcousticSimulator)
        # 每波长 ≥10 点：2000Hz, c=343 → λ=0.1715m → dx≤0.01715 → nx≥291
        self.assertGreaterEqual(sim.nx, 100)
        self.assertGreaterEqual(sim.ny, 100)

    def test_explicit_grid_not_overridden_by_high_freq(self):
        """显式指定 grid_resolution 时不应被自动调参覆盖"""
        params = {
            'max_frequency': 5000.0,
            'grid_resolution': (60, 60),
            'spatial_domain': (3.0, 3.0),
            'time_step': 0.00002,
            'total_time': 0.001,
        }
        sim = create_simulator(params)
        self.assertEqual(sim.nx, 60)
        self.assertEqual(sim.ny, 60)

    def test_beamforming_interface(self):
        """BeamformingSimulator 的 BaseAcousticSimulator 接口行为"""
        params = {'mic_positions': [[0.0, 0.0], [1.0, 0.0]]}
        sim = create_simulator(params)
        self.assertFalse(sim.step())
        self.assertIsNone(sim.simulate())
        self.assertEqual(sim.get_pressure_at_point(np.array([0.5, 0.0])), 0.0)

    def test_config_fields_passed_through(self):
        """AcousticConfig 字段应正确透传"""
        params = {
            'grid_resolution': (50, 50),
            'spatial_domain': (5.0, 5.0),
            'time_step': 0.0002,
            'total_time': 0.01,
            'boundary_type': 'reflecting',
            'damping': 0.05,
        }
        sim = create_simulator(params)
        self.assertEqual(sim.config.boundary_type, 'reflecting')
        self.assertAlmostEqual(sim.config.damping, 0.05)


class TestAcousticAgentFromParams(unittest.TestCase):
    """AcousticAgent.from_params 类方法的单元测试"""

    def test_from_params_default_uses_acoustic_simulator(self):
        """默认参数 → AcousticSimulator 绑定到 agent"""
        params = {
            'grid_resolution': (40, 40),
            'spatial_domain': (4.0, 4.0),
            'time_step': 0.0002,
            'total_time': 0.005,
        }
        agent = AcousticAgent.from_params(position=[2.0, 2.0], params=params)
        self.assertIsInstance(agent.simulator, AcousticSimulator)
        self.assertNotIsInstance(agent.simulator, RoomAcousticSimulator)

    def test_from_params_room_uses_room_simulator(self):
        """room_size 参数 → RoomAcousticSimulator 绑定到 agent"""
        params = {
            'room_size': (6.0, 6.0),
            'grid_resolution': (40, 40),
            'spatial_domain': (6.0, 6.0),
            'time_step': 0.0002,
            'total_time': 0.005,
        }
        agent = AcousticAgent.from_params(position=[3.0, 3.0], params=params, name="RoomAgent")
        self.assertIsInstance(agent.simulator, RoomAcousticSimulator)
        self.assertEqual(agent.name, "RoomAgent")

    def test_from_params_beamforming(self):
        """mic_positions 参数 → BeamformingSimulator 绑定到 agent"""
        params = {'mic_positions': [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]}
        agent = AcousticAgent.from_params(position=[1.0, 1.0], params=params)
        self.assertIsInstance(agent.simulator, BeamformingSimulator)

    def test_from_params_position_stored(self):
        """agent.position 应与传入值一致"""
        params = {'grid_resolution': (40, 40), 'time_step': 0.0002, 'total_time': 0.005}
        agent = AcousticAgent.from_params(position=[1.5, 2.5], params=params)
        np.testing.assert_array_almost_equal(agent.position, [1.5, 2.5])

    def test_from_params_beamforming_perceive(self):
        """BeamformingSimulator 绑定时 perceive() 应返回 pressure=0, spl=0"""
        params = {'mic_positions': [[0.0, 0.0], [1.0, 0.0]]}
        agent = AcousticAgent.from_params(position=[0.5, 0.0], params=params)
        state = agent.perceive()
        self.assertEqual(state['pressure'], 0.0)
        self.assertEqual(state['spl'], 0.0)
        self.assertEqual(state['time'], 0.0)


# ---------------------------------------------------------------------------
# 边界条件测试
# ---------------------------------------------------------------------------

class TestBoundaryConditions(unittest.TestCase):
    """不同边界条件下的物理行为验证"""

    def _make_sim(self, boundary_type: str) -> AcousticSimulator:
        config = AcousticConfig(
            grid_resolution=(40, 40),
            spatial_domain=(4.0, 4.0),
            time_step=0.0002,
            total_time=0.004,
            boundary_type=boundary_type,
            damping=0.0,
        )
        sim = AcousticSimulator(config)
        sim.add_source(SoundSource(
            position=np.array([2.0, 2.0]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.POINT,
        ))
        return sim

    def test_absorbing_boundary_reduces_energy(self):
        """吸收边界：足够时间后场内总能量应低于反射边界"""
        sim_abs = self._make_sim("absorbing")
        sim_ref = self._make_sim("reflecting")
        sim_abs.simulate()
        sim_ref.simulate()
        energy_abs = np.sum(sim_abs.pressure ** 2)
        energy_ref = np.sum(sim_ref.pressure ** 2)
        self.assertLessEqual(energy_abs, energy_ref)

    def test_periodic_boundary_corner_matches_opposite(self):
        """周期边界：一侧边界的声压应等于对侧边界"""
        sim = self._make_sim("periodic")
        # 跑几步后检查周期性
        for _ in range(5):
            sim.step()
        p = sim.pressure
        # 周期边界条件 target_field[0,:] = target_field[-2,:] 等
        np.testing.assert_array_almost_equal(p[0, :], p[-2, :],
                                             err_msg="周期边界：x=0 行应等于 x=N-2 行（边界条件使用 N-2 而非 N-1 避免重复计数）")
        np.testing.assert_array_almost_equal(p[:, 0], p[:, -2],
                                             err_msg="周期边界：y=0 列应等于 y=N-2 列（边界条件使用 N-2 而非 N-1 避免重复计数）")

    def test_reflecting_boundary_nonzero_pressure(self):
        """反射边界：运行后场内应存在非零声压"""
        config = AcousticConfig(
            grid_resolution=(40, 40),
            spatial_domain=(4.0, 4.0),
            time_step=0.0002,
            total_time=0.004,
            boundary_type="reflecting",
            damping=0.0,
        )
        sim = AcousticSimulator(config)
        # LINE 声源使用 get_signal（纯正弦），避免高斯包络在 t=0 附近为 0 的问题
        sim.add_source(SoundSource(
            position=np.array([2.0, 2.0]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.LINE,
            direction=None,
        ))
        # 运行足够步数，让 sin 越过零点产生非零激励
        for _ in range(5):
            sim.step()
        self.assertGreater(np.max(np.abs(sim.pressure)), 0.0)


# ---------------------------------------------------------------------------
# 声源类型测试
# ---------------------------------------------------------------------------

class TestSourceTypes(unittest.TestCase):
    """不同声源类型的激励行为验证"""

    def _base_config(self) -> AcousticConfig:
        return AcousticConfig(
            grid_resolution=(50, 50),
            spatial_domain=(5.0, 5.0),
            time_step=0.0002,
            total_time=0.002,
            damping=0.0,
            boundary_type="absorbing",
        )

    def test_directional_source_forward_stronger(self):
        """定向声源：沿方向的声压幅值应不弱于反方向"""
        config = self._base_config()
        sim = AcousticSimulator(config)
        src = SoundSource(
            position=np.array([2.5, 2.5]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.DIRECTIONAL,
            direction=np.array([1.0, 0.0]),
        )
        sim.add_source(src)
        for _ in range(5):
            sim.step()
        px, py = int(2.5 / sim.dx), int(2.5 / sim.dy)
        # 前向（+x）
        fwd = abs(sim.pressure[min(px + 2, sim.nx - 1), py])
        # 后向（-x）
        bwd = abs(sim.pressure[max(px - 2, 0), py])
        # 前向激励 ≥ 后向激励（定向声源抑制后向辐射）
        self.assertGreaterEqual(fwd, bwd)

    def test_line_source_no_direction_excites_entire_column(self):
        """线声源（无方向）应激励指定 y 列上的所有网格点"""
        config = self._base_config()
        sim = AcousticSimulator(config)
        src = SoundSource(
            position=np.array([2.5, 2.5]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.LINE,
            direction=None,
        )
        sim.add_source(src)
        # 运行多步使 sin 信号越过 t=0 的零点，从而产生非零激励
        for _ in range(3):
            sim.step()
        py = int(2.5 / sim.dy)
        # 整列 y=py 应有非零声压（线声源沿 x 轴延伸）
        col_pressure = sim.pressure[:, py]
        self.assertGreater(np.sum(np.abs(col_pressure)), 0.0,
                           "线声源应在整列产生非零声压")

    def test_point_source_zero_before_activate(self):
        """未激活声源不应产生激励"""
        config = self._base_config()
        sim = AcousticSimulator(config)
        src = SoundSource(
            position=np.array([2.5, 2.5]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.POINT,
            active=False,
        )
        sim.add_source(src)
        sim.step()
        self.assertTrue(np.allclose(sim.pressure, 0.0),
                        "未激活声源不应产生任何声压")


# ---------------------------------------------------------------------------
# 频谱与 SPL 测试
# ---------------------------------------------------------------------------

class TestFrequencyAndSPL(unittest.TestCase):
    """频谱分析与声压级计算验证"""

    def _run_sim(self, frequency: float = 200.0, steps: int = 60) -> AcousticSimulator:
        config = AcousticConfig(
            grid_resolution=(40, 40),
            spatial_domain=(4.0, 4.0),
            time_step=0.0002,
            total_time=steps * 0.0002,
            boundary_type="absorbing",
            damping=0.0,
        )
        sim = AcousticSimulator(config)
        # LINE 声源使用 get_signal（纯正弦），在短仿真时间内即可产生非零信号；
        # POINT 声源使用高斯波包（中心在 t=1.0s），在 total_time < 0.5s 时近似为 0。
        sim.add_source(SoundSource(
            position=np.array([2.0, 2.0]),
            frequency=frequency,
            amplitude=1.0,
            source_type=SoundSourceType.LINE,
            direction=None,
        ))
        sim.simulate()
        return sim

    def test_spl_positive_at_source(self):
        """声源位置的 SPL 应大于 0"""
        sim = self._run_sim()
        py = int(2.0 / sim.dy)
        # LINE 声源沿整列 x 轴激励，在 y=py 列的任意一列点均应有 SPL > 0
        px = sim.nx // 2
        spl = sim.compute_spl((px, py))
        self.assertGreater(spl, 0.0, "声源列的 SPL 应为正值")

    def test_frequency_spectrum_has_data(self):
        """运行后中心点频谱应包含非空数据"""
        sim = self._run_sim(steps=60)
        center = (sim.nx // 2, sim.ny // 2)
        freqs, amps = sim.compute_frequency_spectrum(center)
        self.assertGreater(freqs.size, 0, "频谱频率数组不应为空")
        self.assertGreater(amps.size, 0, "频谱幅值数组不应为空")

    def test_spl_zero_at_out_of_bounds(self):
        """越界坐标的 SPL 应返回 0"""
        sim = self._run_sim()
        spl = sim.compute_spl((9999, 9999))
        self.assertEqual(spl, 0.0)


# ---------------------------------------------------------------------------
# 结果导出测试
# ---------------------------------------------------------------------------

class TestExport(unittest.TestCase):
    """NPY / CSV 导出功能验证"""

    def _make_sim(self) -> AcousticSimulator:
        config = AcousticConfig(
            grid_resolution=(20, 20),
            spatial_domain=(2.0, 2.0),
            time_step=0.0002,
            total_time=0.004,
        )
        sim = AcousticSimulator(config)
        sim.add_source(SoundSource(
            position=np.array([1.0, 1.0]),
            frequency=200.0,
            amplitude=1.0,
            source_type=SoundSourceType.POINT,
        ))
        sim.simulate()
        return sim

    def test_export_pressure_npy_roundtrip(self):
        """NPY 声压场导出/导入应数值一致"""
        sim = self._make_sim()
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            sim.export_pressure_npy(path)
            loaded = np.load(path)
            np.testing.assert_array_equal(sim.pressure, loaded)
        finally:
            os.unlink(path)

    def test_export_pressure_csv_roundtrip(self):
        """CSV 声压场导出/导入应数值一致"""
        sim = self._make_sim()
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            sim.export_pressure_csv(path)
            loaded = np.loadtxt(path, delimiter=',')
            np.testing.assert_array_almost_equal(sim.pressure, loaded)
        finally:
            os.unlink(path)

    def test_export_history_npy_shape(self):
        """历史 NPY 导出形状应为 (帧数, nx, ny)"""
        sim = self._make_sim()
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            sim.export_history_npy(path)
            loaded = np.load(path)
            self.assertEqual(loaded.ndim, 3)
            self.assertEqual(loaded.shape[1], sim.nx)
            self.assertEqual(loaded.shape[2], sim.ny)
            self.assertGreater(loaded.shape[0], 0)
        finally:
            os.unlink(path)

    def test_save_figure_creates_file(self):
        """save_figure 应生成非空 PNG 文件"""
        sim = self._make_sim()
        viz = AcousticVisualizer(sim)
        viz.setup_figure()
        viz.plot_static()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            viz.save_figure(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)
            plt.close('all')

    def test_save_figure_raises_without_plot(self):
        """未创建图形时调用 save_figure 应抛出 RuntimeError"""
        sim = self._make_sim()
        viz = AcousticVisualizer(sim)
        with self.assertRaises(RuntimeError):
            viz.save_figure('/tmp/should_not_exist.png')


# ---------------------------------------------------------------------------
# JSON 配置序列化测试
# ---------------------------------------------------------------------------

class TestAcousticConfigJson(unittest.TestCase):
    """AcousticConfig JSON 序列化 / 反序列化验证"""

    def test_json_roundtrip_default(self):
        """默认配置经 JSON 往返后应与原始值一致"""
        original = AcousticConfig(
            grid_resolution=(60, 80),
            spatial_domain=(6.0, 8.0),
            time_step=0.0005,
            total_time=0.05,
            boundary_type='reflecting',
            damping=0.03,
        )
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            original.to_json(path)
            restored = AcousticConfig.from_json(path)
            self.assertEqual(restored.grid_resolution, original.grid_resolution)
            self.assertEqual(restored.spatial_domain, original.spatial_domain)
            self.assertAlmostEqual(restored.time_step, original.time_step)
            self.assertAlmostEqual(restored.total_time, original.total_time)
            self.assertEqual(restored.boundary_type, original.boundary_type)
            self.assertAlmostEqual(restored.damping, original.damping)
        finally:
            os.unlink(path)

    def test_json_roundtrip_sample_rate_sync(self):
        """JSON 往返后 sample_rate 应与 time_step 保持同步"""
        original = AcousticConfig(time_step=0.001)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            original.to_json(path)
            restored = AcousticConfig.from_json(path)
            self.assertEqual(restored.sample_rate, int(round(1.0 / restored.time_step)))
        finally:
            os.unlink(path)

    def test_from_json_chinese_material(self):
        """from_json 应能正确解析中文介质名（如 '空气'）"""
        data = {
            'grid_resolution': [40, 40],
            'spatial_domain': [4.0, 4.0],
            'time_step': 0.001,
            'total_time': 0.01,
            'boundary_type': 'absorbing',
            'damping': 0.01,
            'cfl_number': 0.5,
            'material': '空气',
        }
        with tempfile.NamedTemporaryFile(
            suffix='.json', delete=False, mode='w', encoding='utf-8'
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            path = f.name
        try:
            config = AcousticConfig.from_json(path)
            self.assertEqual(config.material.name, '空气')
        finally:
            os.unlink(path)

    def test_from_json_missing_material_defaults_to_air(self):
        """from_json 缺少 material 字段时应默认使用空气介质"""
        data = {
            'grid_resolution': [40, 40],
            'spatial_domain': [4.0, 4.0],
            'time_step': 0.001,
            'total_time': 0.01,
        }
        with tempfile.NamedTemporaryFile(
            suffix='.json', delete=False, mode='w', encoding='utf-8'
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            config = AcousticConfig.from_json(path)
            self.assertEqual(config.material.name, '空气')
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
