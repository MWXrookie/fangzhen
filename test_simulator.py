import unittest

import numpy as np

from shengxuefangzhen import (
    AcousticAgent,
    AcousticConfig,
    AcousticSimulator,
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


if __name__ == "__main__":
    unittest.main()
