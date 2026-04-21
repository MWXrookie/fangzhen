import unittest

import numpy as np

from shengxuefangzhen import (
    AcousticConfig,
    AcousticSimulator,
    MaterialProperties,
    MediumType,
    Obstacle,
    SoundSource,
    SoundSourceType,
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


if __name__ == "__main__":
    unittest.main()
