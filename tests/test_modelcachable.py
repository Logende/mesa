from mesa.modelcachable import Model, ModelCachable, CacheState

from unittest.mock import MagicMock
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path


class ModelFibonacci(Model):
    previous = 0
    current = 1

    def step(self):
        new_value = self.previous + self.current
        self.previous = self.current
        self.current = new_value
        if new_value > 1000:
            self.running = False

    def custom_model_function(self):
        return self.current


class TestModelCachable(unittest.TestCase):

    def test_model_attribute_access_over_wrapper(self):
        model = ModelFibonacci()
        model = ModelCachable(model, "irrelevant_cache_file_path", CacheState.WRITE)
        assert model.running is True
        assert model.previous == 0
        assert model.custom_model_function() == 1

    def test_cache_read_fail_when_invalid_input(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file_not_existing")

            model = ModelFibonacci()

            # No exception when constructing ModelCachable with CacheState.WRITE because does not try to read cache
            ModelCachable(model, cache_file_path, CacheState.WRITE)

            # Exception when trying to construct ModelCachable with CacheState.READ and non-existing cache file
            self.assertRaises(Exception, ModelCachable, model, cache_file_path, CacheState.READ)

    def test_compare_replay_with_simulation(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")
            step_count = 100
            values_simulate = []
            values_replay = []

            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            for i in range(step_count):
                model_simulate.step()
                values_simulate.append(model_simulate.current)
            model_simulate.write_cache_file()

            # Replay
            model_replay = ModelFibonacci()
            # Mock the actual 'step' function to show that we do not simulate again, but instead the replay does work
            mock_function = MagicMock(name='step')
            mock_function.side_effect = Exception('Should not be called because we do not simulate but instead replay.')
            model_replay.step = mock_function
            model_replay = ModelCachable(model_replay, cache_file_path, CacheState.READ)
            for i in range(step_count):
                model_replay.step()
                values_replay.append(model_replay.current)

            # Assert that values are identical
            assert values_replay == values_simulate

    def test_cache_size(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")
            step_count = 100

            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            for i in range(step_count):
                model_simulate.step()
            model_simulate.write_cache_file()

            # Load from cache and check cache size
            model_replay = ModelFibonacci()
            model_replay = ModelCachable(model_replay, cache_file_path, CacheState.READ)
            assert len(model_replay.cache) == step_count

    def test_automatic_save_after_run_finished(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            mock_function = MagicMock(name='write_cache_file')
            model_simulate.write_cache_file = mock_function

            assert mock_function.call_count == 0

            model_simulate.run_model()

            assert mock_function.call_count == 1

    def test_replay_finish_identical_to_simulation_finish(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            model_simulate.run_model()
            final_value_simulation = model_simulate.current
            final_step_simulation = model_simulate.step_count

            model_replay = ModelFibonacci()
            model_replay = ModelCachable(model_replay, cache_file_path, CacheState.READ)
            model_replay.run_model()
            final_value_replay = model_simulate.current
            final_step_replay = model_replay.step_count

            assert final_step_replay == final_step_simulation
            assert final_value_replay == final_value_simulation
